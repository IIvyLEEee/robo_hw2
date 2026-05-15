import torch
import hydra

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    UnboundedContinuous,
)
import builtins
from dataclasses import fields, is_dataclass

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer
from collections import OrderedDict

from abc import abstractmethod
from typing import Dict, List
import time

import active_adaptation.envs.mdp as mdp

class Env(EnvBase):
    def __init__(self, cfg):
        super().__init__(
            device=cfg.sim.device,
            batch_size=[cfg.scene.num_envs],
            run_type_checks=False,
        )
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False
        self.enable_render = False
        self.is_env_initializing = True

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        try:
            import omni.replicator.core as rep
            # create render product
            self._render_product = rep.create.render_product(
                "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
            )
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        except ModuleNotFoundError as e:
            print(e)
            print("Set enable_cameras=true to use cameras.")

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self.cfg.scene)
            for k, v in self.scene.articulations.items():
                v._env = self
        print("[INFO]: Scene manager: ", self.scene)

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start"):
                self.sim.reset()
        for _ in range(4):
            self.sim.step(render=True)
        
        self.max_episode_length = self.cfg.max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.step_dt = self.physics_dt * self.cfg.decimation

        # parse obs and reward functions
        self.done_spec = (
            Composite(
                {
                    "done": Binary(1, dtype=bool),
                    "terminated": Binary(1, dtype=bool),
                    "truncated": Binary(1, dtype=bool),
                },
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.reward_spec = Composite(
            {
                "task_success": UnboundedContinuous([self.num_envs, 1]),
            },
            shape=[self.num_envs]
        ).to(self.device)

        import inspect
        members = dict(inspect.getmembers(self.__class__, inspect.isclass))

        RAND_FUNCS = mdp.RAND_FUNCS
        RAND_FUNCS.update(mdp.get_obj_by_class(members, mdp.Randomization))
        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update(mdp.get_obj_by_class(members, mdp.Observation))
        TERM_FUNCS = mdp.TERM_FUNCS
        TERM_FUNCS.update(mdp.get_obj_by_class(members, mdp.Termination))

        self.randomizations = OrderedDict()
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()
        self._post_init_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []
        
        #### action manager ####
        self.action_manager: mdp.ActionManager = hydra.utils.instantiate(self.cfg.action, env=self)
        self._update_callbacks.append(self.action_manager.update)
        self._reset_callbacks.append(self.action_manager.reset)
        self._debug_draw_callbacks.append(self.action_manager.debug_draw)
        
        self.action_spec = Composite(
            {
                "action": UnboundedContinuous((self.num_envs, self.action_dim))
            },
            shape=[self.num_envs]
        ).to(self.device)

        #### randomization ####
        for key, params in self.cfg.randomization.items():
            rand = RAND_FUNCS[key](self, **params if params is not None else {})
            self.randomizations[key] = rand
            rand.startup()
            self._reset_callbacks.append(rand.reset)
            self._debug_draw_callbacks.append(rand.debug_draw)
            self._pre_step_callbacks.append(rand.step)
            self._update_callbacks.append(rand.update)

        #### observation ####
        for group_key, params in self.cfg.observation.items():
            # Extract group-level config: `no_flatten` (default True)
            group_cfg = dict(params) if params is not None else {}
            no_flatten = group_cfg.pop("no_flatten", True)

            funcs = OrderedDict()
            for key, kwargs in group_cfg.items():
                # instantiate each observation func under this group
                obs = OBS_FUNCS[key](self, **(kwargs if kwargs is not None else {}))
                funcs[key] = obs
                obs.startup()
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
                self._post_step_callbacks.append(obs.post_step)
                if hasattr(obs, "__post_init__"):
                    self._post_init_callbacks.append(obs.__post_init__)

            # Validate: if no_flatten, only one obs in this group is allowed
            if no_flatten and len(funcs) != 1:
                raise ValueError(
                    f"Observation group '{group_key}' has {len(funcs)} items, but no_flatten=True requires exactly one. "
                    f"Set `no_flatten: false` in the config for this group to allow multiple and flatten+concat."
                )

            self.observation_funcs[group_key] = ObsGroup(group_key, funcs, no_flatten=no_flatten)

        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            group_spec = group.spec
            observation_spec.update(group_spec.expand(self.num_envs).to(self.device))

        self.observation_spec = Composite(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )

        #### termination ####
        self.termination_funcs = OrderedDict()
        termination_cfg = dict(self.cfg.termination) if self.cfg.termination is not None else {}
        for key, params in termination_cfg.items():
            term_func = TERM_FUNCS[key](self, **params)
            self.termination_funcs[key] = term_func
            self._update_callbacks.append(term_func.update)
            self._reset_callbacks.append(term_func.reset)
            self._debug_draw_callbacks.append(term_func.debug_draw)
            if hasattr(term_func, "__post_init__"):
                self._post_init_callbacks.append(term_func.__post_init__)

        self.timestamp = 0
    
        self.input_tensordict = None

        self.lookat_env_i = 0

        self.extra = {}
        self.has_lang_ins = self.cfg.has_lang_ins
        if self.has_lang_ins:
            self.extra["text_prompts"] = [""] * self.num_envs
        
        self.is_env_initializing = False

    def __post_init__(self):
        for callback in self._post_init_callbacks:
            callback()

    @property
    def action_dim(self) -> int:
        return self.action_manager.action_dim

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs
    
    @property
    def physics_dt(self) -> float:
        return self.sim.get_physics_dt()
    
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        if len(env_ids):
            self._reset_idx(env_ids)

        if self.has_lang_ins and env_ids.numel() > 0:
            idx_list = env_ids.cpu().tolist()
            prompts = self.lang_ins(idx_list)  # List[str]，len == len(idx_list)
            for i, p in zip(idx_list, prompts):
                self.extra["text_prompts"][i] = p

        self.episode_length_buf[env_ids] = 0
        for callback in self._reset_callbacks:
            callback(env_ids)

        tensordict = TensorDict({}, self.num_envs, device=self.device)
        self._compute_observation(tensordict)

        return tensordict
    
    def lang_ins(self, env_ids: List[int]) -> List[str]:
        plist = [""] * len(env_ids)
        print("Language instruction should be implemented in subclass")
        return plist

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.input_tensordict = tensordict
        self.action_manager(tensordict, substep)

    def _compute_observation(self, tensordict: TensorDictBase):
        try:
            for group_key, obs_group in self.observation_funcs.items():
                obs_group.compute(tensordict, self.timestamp)
        except Exception as e:
            print(f"Error in computing observation for {group_key}: {e}")
            raise e
    
    def _compute_termination(self, tensordict: TensorDictBase) -> TensorDictBase:
        flags = []
        for key, func in self.termination_funcs.items():
            # check if key is for tack success check
            if key.startswith("success"):
                success_rate = func()
                success_rate = success_rate.float()
                tensordict.set("task_success", success_rate)
                flag = (success_rate >= 0.99).float()
            # or else for other termination conditions
            else:
                flag = func()
                flag = flag.float()
            flags.append(flag)
        flags = torch.cat(flags, dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        for callback in self._update_callbacks:
            callback()
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        self.episode_length_buf.add_(1)
        self.timestamp += 1

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for substep in range(self.cfg.decimation):
            self.apply_action(tensordict, substep)
            for callback in self._pre_step_callbacks:
                callback(substep)
            self.scene.write_data_to_sim()
            self.pre_step()

            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

            for callback in self._post_step_callbacks:
                callback(substep)
            self.post_step()

        self._update()
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        self._compute_observation(tensordict)
        terminated = self._compute_termination(tensordict)
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        
        if self.sim.has_gui() and hasattr(self, "debug_draw"):
            self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
            self.debug_vis()
            
        return tensordict
    
    def pre_step(self):
        pass

    def post_step(self):
        pass
    
    def _set_seed(self, seed: int = -1):
        torch.manual_seed(seed)
    
    def debug_vis(self):
        pass

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # update closing status
            super().close()

    def state_dict(self):
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        return dict(self.extra)

class ObsGroup:
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        no_flatten: bool = True,
    ):
        self.name = name
        self.funcs = funcs
        self.no_flatten = bool(no_flatten)
        self.raw_obs_t = OrderedDict()
        self.timestamp = -1

    @property
    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        if not hasattr(self, "_spec"):
            spec = Composite()
            if self.no_flatten:
                # Only one obs is allowed by construction when no_flatten=True
                obs_key, func = next(iter(self.funcs.items()))
                sample = func()
                # `sample` must be shaped [N, ...]; remove batch dim from spec
                if sample.ndim < 2:
                    raise ValueError(
                        f"no_flatten group '{self.name}' expects observation with at least 2 dims [envs, ...], got shape {tuple(sample.shape)}"
                    )
                trailing_shape = tuple(sample.shape[1:])
                spec[self.name] = UnboundedContinuous(trailing_shape)
            else:
                total_dim = 0
                for _, func in self.funcs.items():
                    if hasattr(func, "obs_dim"):
                        total_dim += int(func.obs_dim)
                    else:
                        foo = func()
                        total_dim += int(foo.reshape(foo.shape[0], -1).shape[-1])
                spec[self.name] = UnboundedContinuous(total_dim)
            self._spec = spec
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> torch.Tensor:
        # update only if outdated
        if timestamp > self.timestamp:
            self.raw_obs_t = OrderedDict()
            for obs_key, func in self.funcs.items():
                tensor = func()
                self.raw_obs_t[obs_key] = tensor
            self.timestamp = timestamp

        if self.no_flatten:
            # pass-through the single observation tensor without flattening
            out = next(iter(self.raw_obs_t.values()))
            tensordict[self.name] = out
        else:
            # flatten each tensor per env, then concat on last dim
            flat_tensors = [t.reshape(t.shape[0], -1) for t in self.raw_obs_t.values()]
            tensordict[self.name] = torch.cat(flat_tensors, dim=-1)
        return tensordict

from .basic import BasicEnv
import torch
import active_adaptation.envs.mdp as mdp
from isaaclab.assets import RigidObject, RigidObjectCollection
from .placement_strategies import create_placement_strategy
from collections.abc import Mapping
from typing import Optional

class ManipPAPScene(BasicEnv):
    def __init__(
        self,
        cfg,
        task_objects,
        placement_strategy,
        table_center_offset=(0.0, 0.0, 0.0),
    ):
        super().__init__(cfg)
        self.objs : RigidObjectCollection = self.scene["task_objects"]

        if isinstance(placement_strategy, Mapping):
            strategy_name = placement_strategy.get("name")
            strategy_params = placement_strategy.get("params", {})
        else:
            strategy_name = placement_strategy
            strategy_params = {}
        if not strategy_name:
            raise ValueError("placement_strategy must define a non-empty name.")
        
        self.process_task_objects(task_objects)
        # default root state for each object
        self.default_root_state = self.objs.data.default_object_state # [num_instances, num_objects, 13]
        self.default_root_state[:, :, :3] += self.scene.env_origins.unsqueeze(1)
        self.root_state = self.default_root_state.clone()
        
        # table data
        self.table: RigidObject = self.scene["table"]
        self.table_center = self.table.data.default_root_state[:, :3] + self.scene.env_origins + torch.tensor(table_center_offset, device=self.device).unsqueeze(0)# [1, 3]
        # control whether the object need simulation
        self.sim_inactive = torch.zeros((self.objs.num_instances, self.objs.num_objects, 1), device=self.device, dtype=torch.bool) # [num_instances, num_objects, 1]
        
        ### it is better to use physx to disable the simulation for inactive objects, however, it is not working currently
        # https://github.com/isaac-sim/IsaacLab/discussions/1736
        # self.physx_indice_map = self.objs.reshape_view_to_data(torch.arange(self.objs.num_instances * self.objs.num_objects, device=self.device).unsqueeze(-1)) # [num_instances, num_objects, 1]
        # self.physx_view = self.objs.root_physx_view
        # self.sim_inactive_physx = self.physx_view.get_disable_gravities().clone() # [num_objects*num_instances, 1]
        
        # which object is the top object
        self.target_t = torch.zeros((self.objs.num_instances, 1), device=self.device, dtype=torch.long) # [num_instances, 1]
        # which object is the bottom object
        self.target_b = torch.zeros((self.objs.num_instances, 1), device=self.device, dtype=torch.long) # [num_instances, 1]
        
        context = {
            "table_center": self.table_center,
            "objs_id_map": self.objs_id_map,
            "top_objs": self.top_objs,
            "btm_objs": self.btm_objs,
            "all_objs": self.all_objs,
            "object_names": list(self.objs.object_names),
        }
        self.placement_strategy_name = strategy_name
        self.placement_strategy = create_placement_strategy(strategy_name, strategy_params, context)
        self._last_env0_plan: Optional[dict] = None

        self.__post_init__()

    def _apply_placement_plan(self, plan):
        default_relative = bool(getattr(self.placement_strategy, "positions_relative", False))
        for item in plan:
            env_id = item["env_id"]
            self.target_b[env_id, 0] = item["target_b"]
            self.target_t[env_id, 0] = item["target_t"]
            use_relative = bool(item.get("relative", default_relative))
            center = self.table_center[env_id] if use_relative else None
            for placement in item["placements"]:
                obj_id = placement["obj_id"]
                x_rel, y_rel, z_rel = placement["pos"]
                if center is None:
                    self.root_state[env_id, obj_id, 0] = float(x_rel)
                    self.root_state[env_id, obj_id, 1] = float(y_rel)
                    self.root_state[env_id, obj_id, 2] = float(z_rel)
                else:
                    self.root_state[env_id, obj_id, 0] = float(x_rel) + float(center[0])
                    self.root_state[env_id, obj_id, 1] = float(y_rel) + float(center[1])
                    self.root_state[env_id, obj_id, 2] = float(z_rel) + float(center[2])
                self.sim_inactive[env_id, obj_id, 0] = 0

    def _obj_id_to_name(self, obj_id) -> str:
        try:
            idx = int(obj_id)
        except (TypeError, ValueError):
            return str(obj_id)
        meta = self.objs_id_map.get(idx, {})
        if isinstance(meta, dict):
            name = meta.get("name")
            if name:
                return name
        return self.objs.object_names[idx]

    def _pos_to_list(self, pos):
        if isinstance(pos, torch.Tensor):
            return pos.detach().cpu().tolist()
        if isinstance(pos, (list, tuple)):
            out = []
            for val in pos:
                if isinstance(val, torch.Tensor):
                    out.append(float(val.detach().cpu().item()))
                else:
                    out.append(float(val))
            return out
        return pos

    def _translate_plan_item(self, item: dict) -> dict:
        default_relative = bool(getattr(self.placement_strategy, "positions_relative", False))
        placements = []
        for placement in item.get("placements", []):
            obj_id = placement.get("obj_id")
            placements.append(
                {
                    "obj_id": self._obj_id_to_name(obj_id),
                    "pos": self._pos_to_list(placement.get("pos")),
                }
            )
        return {
            "env_id": int(item.get("env_id", -1)),
            "target_b": self._obj_id_to_name(item.get("target_b")),
            "target_t": self._obj_id_to_name(item.get("target_t")),
            "placements": placements,
            "relative": bool(item.get("relative", default_relative)),
        }

    def _update_last_env0_plan(self, plan) -> None:
        for item in plan:
            if int(item.get("env_id", -1)) == 0:
                translated = self._translate_plan_item(item)
                self._last_env0_plan = translated
                break

    def get_env0_plan(self):
        return self._last_env0_plan
    
    def process_task_objects(self, task_objects):
        if hasattr(self, "objs_cfg"):
            return
        # object config
        self.objs_cfg = task_objects
        self.top_objs = []
        self.btm_objs = []
        self.all_objs = []
        self.objs_id_map = {}
        
        for obj_cfg in self.objs_cfg:
            obj_name = obj_cfg[0]
            obj_role = obj_cfg[2]["role"]
            obj_id = self.objs.object_names.index(obj_name)
            
            self.objs_id_map[obj_id] = obj_cfg[2]
            
            if obj_role == "top":
                self.top_objs.append(obj_id)
            elif obj_role == "btm":
                self.btm_objs.append(obj_id)
            elif obj_role == "both":
                self.top_objs.append(obj_id)
                self.btm_objs.append(obj_id)
            elif obj_role == "other":
                pass
            else:
                raise ValueError(f"Invalid object role: {obj_role}")
            self.all_objs.append(obj_id)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # reset state and activity flags
        self.root_state[env_ids] = self.default_root_state[env_ids]
        self.sim_inactive[env_ids] = 1

        plan = self.placement_strategy.plan(env_ids)
        print(f"-----------------{self.placement_strategy_name} placement------------------------")
        self._update_last_env0_plan(plan)

        self._apply_placement_plan(plan)
        self.objs.write_object_state_to_sim(self.root_state[env_ids], env_ids=env_ids)
    
    def post_step(self):
        # reset state for inactive objects
        data = self.objs.data.object_state_w # [num_instances, num_objects, 13]
        # mask the inactive objects with default state
        data_new = torch.where(self.sim_inactive, self.root_state, data)
        self.objs.write_object_state_to_sim(data_new)
    
    def lang_ins(self, env_ids: torch.Tensor):
        prompts = []
        for env_id in env_ids:
            bottom_idx = self.target_b[env_id, 0].item()  
            top_idx = self.target_t[env_id, 0].item()  

            bottom_name = self.objs_id_map[bottom_idx].get("name", self.objs.object_names[bottom_idx])
            top_name = self.objs_id_map[top_idx].get("name", self.objs.object_names[top_idx])

            prompt = f"pick up the {top_name} and put it in the {bottom_name}."
            print(prompt)
            prompts.append(prompt)
        
        return prompts
    
    class success_stack(mdp.Termination):
        def __init__(
            self,
            env,
            k_xy: float = 0.5,      # XY 允许误差系数（越小越严格）
            dz_min: float = 0.0,  # 期望高差下界    ≈ 1.5 cm
            dz_max: float = 0.10,   # 期望高差上界    ≈ 10 cm
            v_thresh: float = 0.02  # 速度阈值 m/s
        ):
            super().__init__(env)
            self.k_xy     = k_xy
            self.dz_min   = dz_min
            self.dz_max   = dz_max
            self.v_thresh = v_thresh

        def __post_init__(self):
            self.objs = self.env.objs
            self.sizes = torch.zeros(self.env.objs.num_objects, device=self.env.device)
            self.env.process_task_objects(self.env.objs_cfg)
            for oid, cfg in self.env.objs_id_map.items():
                self.sizes[oid] = float(cfg.get("size", 0.06))

        def __call__(self) -> torch.Tensor:
            # === 取出位置 & 速度 =========================================================
            state_w = self.objs.data.object_state_w      # [N, n_obj, 13]
            pos     = state_w[..., :3]                       # 质心世界坐标
            vel     = state_w[..., 7:10]                     # 质心线速度

            N        = pos.shape[0]
            env_ids  = torch.arange(N, device=pos.device)

            idx_top  = self.env.target_t.squeeze(-1)         # [N]
            idx_bot  = self.env.target_b.squeeze(-1)         # [N]

            p_top    = pos[env_ids, idx_top]                 # [N, 3]
            p_bot    = pos[env_ids, idx_bot]                 # [N, 3]
            v_top    = vel[env_ids, idx_top]                 # [N, 3]
            v_bot    = vel[env_ids, idx_bot]                 # [N, 3]

            # === (1) XY 位置误差 =========================================================
            loc_err = torch.norm(p_top[:, :2] - p_bot[:, :2], dim=1, keepdim=True)  # [N,1]

            size_bot   = self.sizes[idx_bot].unsqueeze(1)                       # [N,1]
            xy_thresh  = self.k_xy * size_bot                                       # [N,1]

            cond_xy = loc_err < xy_thresh

            # === (2) Z 高差 =============================================================
            dz = (p_top[:, 2:3] - p_bot[:, 2:3])                                    # [N,1]
            cond_z = (dz > self.dz_min) & (dz < self.dz_max)

            # === (3) 速度 ===============================================================
            speed_top = torch.norm(v_top, dim=1, keepdim=True)
            speed_bot = torch.norm(v_bot, dim=1, keepdim=True)
            cond_vel  = (speed_top < self.v_thresh) & (speed_bot < self.v_thresh)

            # === 综合判定 ===============================================================
            success = (cond_xy & cond_z & cond_vel).float()                         # [N,1]
            return success

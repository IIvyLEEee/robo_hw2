import torch
import abc
from typing import Tuple, TYPE_CHECKING
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env

class Observation:
    def __init__(self, env):
        """
        For each episode, with probability mask_ratio, the observation will be masked.
        Note that `True` means the observation is masked.

        """
        self.env: Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def lerp(self, obs_tm1: torch.Tensor, obs_t: torch.Tensor, t) -> torch.Tensor:
        return torch.lerp(obs_tm1, obs_t, t)
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        if hasattr(self, "obs_dim"):
            # check if the tensor has the correct shape
            if tensor.shape[-1] != self.obs_dim:
                raise ValueError(f"The last dimension of the obs {self.__class__.__name__} must be {self.obs_dim}, but got {tensor.shape[-1]}")
        return tensor
    
    def startup(self):
        pass
    
    def post_step(self, substep: int):
        pass

    def update(self):
        """Called at each step **after** simulation"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass

def observation_func(func):

    class ObsFunc(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def compute(self):
            return func(self.env, **self.params)
    
    return ObsFunc

class Buffer:
    def __init__(self, shape, size, device):
        self.data = torch.zeros(*shape, size, device=device)
        self.time_stamp = 0

    def reset(self, env_ids: torch.Tensor, value=0.):
        self.data[env_ids] = value
    
    def update(self, value: torch.Tensor, time_stamp: int):
        if time_stamp > self.time_stamp:
            self.data[..., :-1] = self.data[..., 1:]
            self.data[..., -1] = value
            self.time_stamp = time_stamp

class BufferedObs(Observation):
    def __init__(self, env, shape, size):
        super().__init__(env)
        self.buffer = Buffer(shape, size, self.env.device)
        setattr(self.env, f"_{self.__class__.__name__}")
    
    def compute(self):
        return self.buffer.data.reshape(self.env.num_envs, -1)

    def reset(self, env_ids: torch.Tensor):
        self.buffer.reset(env_ids)

class command(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager

    def compute(self):
        return self.command_manager.command
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return self.command_manager.fliplr(obs)

class command_hidden(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def compute(self):
        return self.command_manager.command_hidden

class action_state(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
    def compute(self):
        return self.env.action_manager.action_state
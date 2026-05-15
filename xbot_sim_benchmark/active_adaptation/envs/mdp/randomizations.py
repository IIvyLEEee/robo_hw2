import torch
import numpy as np

class Randomization:
    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass

import torch
import abc

class Termination:
    def __init__(self, env):
        self.env = env
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def debug_draw(self):
        pass


def termination_func(func):
    class TermFunc(Termination):
        def __call__(self):
            return func(self.env)
    return TermFunc

class dummy(Termination):
    def __init__(self, env):
        super().__init__(env)
        self.always = torch.zeros(self.num_envs, 1, device=self.env.device, dtype=bool)
    
    def __call__(self):
        return self.always
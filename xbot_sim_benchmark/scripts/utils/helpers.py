from omegaconf import OmegaConf, DictConfig

def make_env(cfg: DictConfig):
    from active_adaptation.scenes.task import setup_task_env
    from torchrl.envs.transforms import TransformedEnv

    base_env = setup_task_env(cfg.task)
    
    env = TransformedEnv(base_env)
    env.set_seed(cfg.seed)
    return env
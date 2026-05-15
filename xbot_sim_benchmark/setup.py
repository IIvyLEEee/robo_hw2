from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "einops",
        "av", # for moviepy
        "pandas",
        "termcolor",
        "pygame", # for game controller
        "mink==1.1.0",
        "mujoco",
        "gsplat",
        "zmq",
        "setproctitle",
        "tensordict==0.7.2",
        "torchrl==0.7.2",
    ],
)

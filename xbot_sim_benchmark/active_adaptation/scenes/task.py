from active_adaptation.assets import ASSET_PATH, ROBOTS
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollectionCfg
)
from dataclasses import MISSING
from omegaconf.dictconfig import DictConfig
from typing import Dict, List

from .task_scene_cfg import TASK_SCENE_CFG_LIST
from .task_env import TASK_ENV_LIST

@configclass
class EnvCfg:

    max_episode_length: int = 1000

    history_length: int = 32
    
    has_lang_ins: bool = False

    viewer: ViewerCfg = ViewerCfg(eye=(-6.0, 0.0, 10.0), lookat=(0.0, 0.0, 3.0))
    scene : InteractiveSceneCfg = MISSING
    gsplat: Dict = MISSING

    decimation: int = 4
    sim = sim_utils.SimulationCfg(dt=0.005, render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_global_illumination=True, enable_reflections=True, enable_ambient_occlusion=True, enable_direct_lighting=True, enable_translucency=True, samples_per_pixel=4))
    # sim = sim_utils.SimulationCfg(dt=0.005, render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_global_illumination=True, rendering_mode="quality", enable_reflections=True, enable_ambient_occlusion=True, enable_direct_lighting=True, enable_translucency=True))

    action: Dict = MISSING
    observation: Dict[str, List] = MISSING
    termination: List = MISSING
    randomization: List = MISSING

def unfold_object_collection(scene, collections):
    for name, collection in collections.items():
        objs = {}
        height = 5.0
        for obj_data in collection:
            obj_name = obj_data[0]
            obj_path = obj_data[1]
            objs[obj_name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/"+obj_name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ASSET_PATH}/{obj_path}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, height)),
            )
            height += 1.0
        setattr(scene, name, RigidObjectCollectionCfg(rigid_objects=objs))

def setup_task_env(task_cfg):
    robot_name = task_cfg.robot.lower()
    robot_cfg = ROBOTS[robot_name]
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

    scene_cfg_class = TASK_SCENE_CFG_LIST[task_cfg.scene]
    
    scene = scene_cfg_class(
            num_envs=task_cfg.num_envs,
            robot=robot_cfg,
            replicate_physics=True,
        )

    if hasattr(task_cfg, "object_collection"):
        unfold_object_collection(scene, task_cfg.object_collection)
    
    if hasattr(task_cfg, "3dgs"):
        gsplat = task_cfg["3dgs"]
    else:
        gsplat = None

    env_cfg = EnvCfg(
        max_episode_length=task_cfg.max_episode_length,
        decimation=task_cfg.decimation,
        has_lang_ins=task_cfg.has_lang_ins,
        scene=scene,
        action=task_cfg.action,
        gsplat=gsplat,
        observation=task_cfg.observation,
        termination=task_cfg.termination if hasattr(task_cfg, "termination") else {},
        randomization=task_cfg.randomization,
        viewer=ViewerCfg(eye=task_cfg.eye, lookat=task_cfg.lookat)
    )
    if hasattr(task_cfg, "sim_device"):
        env_cfg.sim.device = task_cfg.sim_device

    use_camera = False
    for group in task_cfg.observation.values():
        if "camera" in group.keys():
            use_camera = True
    if not use_camera:
        env_cfg.scene.camera = None

    if isinstance(task_cfg.task, str):
        task_name = task_cfg.task
        task_params = {}
    elif isinstance(task_cfg.task, dict) or isinstance(task_cfg.task, DictConfig):
        task_name = task_cfg.task["name"]
        task_params = task_cfg.task["params"]
    else:
        raise ValueError(f"Invalid task configuration: {task_cfg.task}, type: {type(task_cfg.task)}")

    env_class = TASK_ENV_LIST[task_name]
    env = env_class(env_cfg, **task_params)
    return env

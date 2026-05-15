from active_adaptation.assets import ASSET_PATH
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg
)
from isaaclab.sensors import (
    TiledCameraCfg,
)
import isaaclab.sim as sim_utils

from dataclasses import MISSING


@configclass
class XbotManipSceneCfg(InteractiveSceneCfg):

    num_envs: int = 4096
    env_spacing: float = 3.0

    robot: ArticulationCfg = MISSING

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/xbot_3dgs/xbot_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            semantic_tags=[("class", "table")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5664, 0.0522, -0.126),
        ),
    )
    
    head_cam: TiledCameraCfg = TiledCameraCfg(
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[432.5301575043416, 0.0, 424.9090025057237, 0.0, 432.0592384152491, 244.4039466097479, 0.0, 0.0, 1.0],
            width=848,
            height=480,
        ),
        prim_path="{ENV_REGEX_NS}/Robot/stereo_link/head_cam/cam",
        update_period=0.02,
        width=848,
        height=480,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
    )

    left_wrist_cam: TiledCameraCfg = TiledCameraCfg(
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[266.32843426552085, 0.0, 318.4817991748623, 0.0, 266.817859462642, 260.78867275908215, 0.0, 0.0, 1.0],
            width=640,
            height=480,
        ),
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_camera_link/cam/cam",
        update_period=0.02,
        width=640,
        height=480,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
    )

    right_wrist_cam: TiledCameraCfg = TiledCameraCfg(
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[268.24702130688024, 0.0, 311.09858862062146, 0.0, 269.8357309111451, 257.71128243434185, 0.0, 0.0, 1.0],
            width=640,
            height=480,
        ),
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_camera_link/cam/cam",
        update_period=0.02,
        width=640,
        height=480,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
    )

import copy
import os

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab_assets import H1_CFG

ASSET_PATH = os.path.dirname(__file__)

L7_FIXED_LOWER_BODY_CFG = copy.deepcopy(H1_CFG)
L7_FIXED_LOWER_BODY_CFG.spawn.usd_path = f"{ASSET_PATH}/robots/rel7_u/urdf/M7/M7_flatten.usd"
L7_FIXED_LOWER_BODY_CFG.spawn.rigid_props.disable_gravity = True
L7_FIXED_LOWER_BODY_CFG.spawn.articulation_props.enabled_self_collisions = False
L7_FIXED_LOWER_BODY_CFG.actuators = {
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist.*"],
        effort_limit_sim=220.0,
        velocity_limit_sim=4.19,
        stiffness={
            "waist.*": 200,
        },
        damping={
            "waist.*": 5,
        },
        friction=0.0,
        armature=0.01,
    ),
    "neck_head": ImplicitActuatorCfg(
        joint_names_expr=["neck_yaw_joint", "neck_pitch_joint"],
        effort_limit_sim=35.0,
        velocity_limit_sim=4.19,
        stiffness={
            "neck.*": 20,
        },
        damping={
            "neck.*": 2,
        },
        friction=0.0,
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*(shoulder|elbow|wrist|arm).*"],
        effort_limit_sim=110.0,
        velocity_limit_sim=4.19,
        stiffness={
            ".*shoulder.*": 50,
            ".*elbow.*": 50,
            ".*arm.*": 50,
            ".*wrist.*": 50,
        },
        damping={
            ".*shoulder.*": 2,
            ".*elbow.*": 2,
            ".*arm.*": 2,
            ".*wrist.*": 2,
        },
        friction=0.0,
        armature=0.01,
    ),
    "fingers": ImplicitActuatorCfg(
        joint_names_expr=[".*(hand_thumb|hand_index|hand_mid|hand_ring|hand_pinky).*"],
        effort_limit_sim=20.0,
        velocity_limit_sim=5.0,
        stiffness={
            ".*": 20,
        },
        damping={
            ".*": 2,
        },
        friction=0.0,
        armature=0.01,
    ),
}
L7_FIXED_LOWER_BODY_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "^(?!left_elbow_pitch_joint$|right_elbow_pitch_joint$|left_elbow_yaw_joint$|right_elbow_yaw_joint$).*": 0.0,
        "^left_elbow_pitch_joint$": -1.5707963267948966 * 1.3,
        "^right_elbow_pitch_joint$": -1.5707963267948966 * 1.3,
        "^left_elbow_yaw_joint$": -1.5707963267948966,
        "^right_elbow_yaw_joint$": 1.5707963267948966,
    },
    joint_vel={".*": 0.0},
)

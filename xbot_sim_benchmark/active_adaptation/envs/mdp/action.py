import torch
from typing import TYPE_CHECKING
from tensordict import TensorDictBase
from isaaclab.assets import Articulation
from active_adaptation.assets import ASSET_PATH
from active_adaptation.utils.math import quaternion_to_rot6d, rot6d_to_quaternion

if TYPE_CHECKING:
    from active_adaptation.envs.base import Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass
    
    def update(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

class XbotActionManager(ActionManager):
    NECK_YAW_LIMITS = (-1.05, 1.05)
    NECK_PITCH_LIMITS = (-0.17, 0.47)
    DATASET_STATE_DIM = 57
    DATASET_WAIST_NAMES = (
        "waist_roll_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
    )
    DATASET_NECK_NAMES = (
        "neck_yaw_joint",
        "neck_pitch_joint",
    )

    DEFAULT_HAND_Q = (
        1.5,
        0.3,
        0.0,
        0.0,
        0.7,
        0.4,
        0.7,
        0.4,
        0.7,
        0.4,
        0.7,
        0.4,
    )

    LEFT_HAND_JOINT_NAMES = (
        "left_hand_thumb_bend_joint",
        "left_hand_thumb_rota_joint1",
        "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint",
        "left_hand_index_joint1",
        "left_hand_index_joint2",
        "left_hand_mid_joint1",
        "left_hand_mid_joint2",
        "left_hand_ring_joint1",
        "left_hand_ring_joint2",
        "left_hand_pinky_joint1",
        "left_hand_pinky_joint2",
    )
    RIGHT_HAND_JOINT_NAMES = (
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    )

    def __init__(
        self,
        env,
        urdf_path: str = f"{ASSET_PATH}/robots/rel7_u/urdf/M7.urdf",
        model_xml_path: str = f"{ASSET_PATH}/robots/rel7_u/rel7_u_mujoco.xml",
    ):
        super().__init__(env)
        if self.num_envs != 1:
            raise ValueError("XbotActionManager with Mink dual-arm IK only supports num_envs == 1.")

        from active_adaptation.envs.mdp.ik.mink_rel7_u_upper_body import Rel7UUpperBodyMinkIK

        self.right_arm_dim = 21
        self.left_arm_dim = 21
        self.action_dim = self.right_arm_dim + self.left_arm_dim

        self.dual_arm_ik = Rel7UUpperBodyMinkIK(
            model_path=model_xml_path,
        )

        self.waist_names = list(self.dual_arm_ik.WAIST_JOINT_NAMES)
        self.left_arm_names = list(self.dual_arm_ik.LEFT_ARM_JOINT_NAMES)
        self.right_arm_names = list(self.dual_arm_ik.RIGHT_ARM_JOINT_NAMES)
        self.upper_body_names = list(self.dual_arm_ik.CONTROLLED_JOINT_NAMES)
        self.waist_ids = [self._find_joint_id(name) for name in self.waist_names]
        self.left_arm_ids = [self._find_joint_id(name) for name in self.left_arm_names]
        self.right_arm_ids = [self._find_joint_id(name) for name in self.right_arm_names]
        self.upper_body_ids = [self._find_joint_id(name) for name in self.upper_body_names]
        self.left_hand_ids = [self._find_joint_id(name) for name in self.LEFT_HAND_JOINT_NAMES]
        self.right_hand_ids = [self._find_joint_id(name) for name in self.RIGHT_HAND_JOINT_NAMES]
        self.dataset_waist_ids = [self._find_joint_id(name) for name in self.DATASET_WAIST_NAMES]
        self.dataset_neck_ids = [self._find_joint_id(name) for name in self.DATASET_NECK_NAMES]

        self.neck_yaw_id = self._find_joint_id("neck_yaw_joint")
        self.neck_pitch_id = self._find_joint_id("neck_pitch_joint")
        self.waist_pitch_id = self._find_joint_id("waist_pitch_joint")
        self.waist_pitch_upper_body_index = self.dual_arm_ik.joint_index("waist_pitch_joint")

        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.idle_upper_body_q = self.default_joint_pos[:, self.upper_body_ids].clone()
        self.dual_arm_ik.set_posture_target(self.idle_upper_body_q[0].detach().cpu().numpy())
        self.default_hand_q = torch.tensor(self.DEFAULT_HAND_Q, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.default_joint_pos[:, self.right_hand_ids] = self.default_hand_q
        self.default_joint_pos[:, self.left_hand_ids] = self.default_hand_q
        self.default_joint_pos[:, self.neck_pitch_id] = 0.45

        self.idle_action_state = self._compose_action_state(self.default_joint_pos)
        self.action_state = self.idle_action_state.clone()

        self.head_pitch_scale = 1.0

        self.interp_reach_substep = 10
        self._prev_upper_body_q = None
        self._target_upper_body_q = None
        self._applied_upper_body_q = None
        self._prev_right_hand = None
        self._target_right_hand = None
        self._applied_right_hand = None
        self._prev_left_hand = None
        self._target_left_hand = None
        self._applied_left_hand = None
        self._prev_neck = None  # (B,2) yaw,pitch
        self._target_neck = None
        self._applied_neck = None

    def reset(self, env_ids: torch.Tensor):
        self.action_state[env_ids] = self.idle_action_state[env_ids]
        idle_upper_body_q = self.idle_upper_body_q.repeat(self.num_envs, 1)
        idle_neck = torch.tensor([[0.0, 0.45]], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        self._prev_upper_body_q = idle_upper_body_q.clone()
        self._target_upper_body_q = idle_upper_body_q.clone()
        self._applied_upper_body_q = idle_upper_body_q.clone()
        self._prev_right_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._target_right_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._applied_right_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._prev_left_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._target_left_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._applied_left_hand = self.default_hand_q.repeat(self.num_envs, 1)
        self._prev_neck = idle_neck.clone()
        self._target_neck = idle_neck.clone()
        self._applied_neck = idle_neck.clone()

    def update(self):
        self.action_state[:, :] = self._compose_action_state(self.asset.data.joint_pos)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            action = tensordict["action"].clone()
            idx = 0
            right_arm_target = action[:, idx: idx+9].clone()
            idx += 9
            right_hand_target = action[:, idx: idx+12].clone()
            idx += 12
            right_target_pos = right_arm_target[:, :3].clone()
            right_target_quat = rot6d_to_quaternion(right_arm_target[:, 3:9])
            left_arm_target = action[:, idx: idx+9].clone()
            idx += 9
            left_hand_target = action[:, idx: idx+12].clone()
            idx += 12
            left_target_pos = left_arm_target[:, :3].clone()
            left_target_quat = rot6d_to_quaternion(left_arm_target[:, 3:9])

            q_current_upper_body = self._get_upper_body_q(self.asset.data.joint_pos)
            q_target_upper_body = self._solve_dual_arm_ik(
                q_current_upper_body=q_current_upper_body,
                left_target_pos=left_target_pos,
                left_target_quat=left_target_quat,
                right_target_pos=right_target_pos,
                right_target_quat=right_target_quat,
            )

            if self._applied_upper_body_q is None:
                self._prev_upper_body_q = q_current_upper_body.clone()
            else:
                self._prev_upper_body_q = self._applied_upper_body_q.clone()
            self._target_upper_body_q = q_target_upper_body.clone()

            if self._applied_right_hand is None:
                self._prev_right_hand = self.asset.data.joint_pos[:, self.right_hand_ids].clone()
            else:
                self._prev_right_hand = self._applied_right_hand.clone()
            self._target_right_hand = right_hand_target.clone()

            if self._applied_left_hand is None:
                self._prev_left_hand = self.asset.data.joint_pos[:, self.left_hand_ids].clone()
            else:
                self._prev_left_hand = self._applied_left_hand.clone()
            self._target_left_hand = left_hand_target.clone()

            ee_sum_y = right_target_pos[:, 1] + left_target_pos[:, 1]
            ee_diff_x = left_target_pos[:, 0] - right_target_pos[:, 0]
            target_neck_yaw = ee_diff_x + ee_sum_y
            target_neck_pitch = (
                -q_target_upper_body[:, self.waist_pitch_upper_body_index] + 0.45
            ) * self.head_pitch_scale
            target_neck_yaw = target_neck_yaw.clamp(*self.NECK_YAW_LIMITS)
            target_neck_pitch = target_neck_pitch.clamp(*self.NECK_PITCH_LIMITS)
            target_neck = torch.stack([target_neck_yaw, target_neck_pitch], dim=1)
            if self._applied_neck is None:
                self._prev_neck = self.asset.data.joint_pos[:, [self.neck_yaw_id, self.neck_pitch_id]].clone()
            else:
                self._prev_neck = self._applied_neck.clone()
            self._target_neck = target_neck

        t = float(substep)
        denom = float(self.interp_reach_substep)
        w = 1.0 if t >= denom else (t / denom)

        if self._prev_upper_body_q is None:
            q_applied_upper_body = self._get_upper_body_q(self.asset.data.joint_pos)
        else:
            q_applied_upper_body = (
                (1.0 - w) * self._prev_upper_body_q + w * self._target_upper_body_q
            )

        joint_target = self.default_joint_pos.clone()
        joint_target[:, self.upper_body_ids] = q_applied_upper_body

        if (self._prev_right_hand is not None) and (self._target_right_hand is not None):
            right_hand_applied = (1.0 - w) * self._prev_right_hand + w * self._target_right_hand
            joint_target[:, self.right_hand_ids] = right_hand_applied
            self._applied_right_hand = right_hand_applied
        if (self._prev_left_hand is not None) and (self._target_left_hand is not None):
            left_hand_applied = (1.0 - w) * self._prev_left_hand + w * self._target_left_hand
            joint_target[:, self.left_hand_ids] = left_hand_applied
            self._applied_left_hand = left_hand_applied

        if (self._prev_neck is not None) and (self._target_neck is not None):
            neck_applied = (1.0 - w) * self._prev_neck + w * self._target_neck
            joint_target[:, self.neck_yaw_id] = neck_applied[:, 0]
            joint_target[:, self.neck_pitch_id] = neck_applied[:, 1]
            self._applied_neck = neck_applied

        if "arm_qpos" in self.env.extra:
            arm_qpos = torch.tensor(self.env.extra["arm_qpos"], device=self.device)
            joint_target[:, self.right_arm_ids] = arm_qpos[:7].reshape(1, -1)
            joint_target[:, self.left_arm_ids] = arm_qpos[7:].reshape(1, -1)
        if "waist_qpos" in self.env.extra:
            waist_qpos = torch.tensor(self.env.extra["waist_qpos"], device=self.device)
            joint_target[:, self.waist_ids] = waist_qpos.reshape(1, -1)
        if "neck_qpos" in self.env.extra:
            neck_qpos = torch.tensor(self.env.extra["neck_qpos"], device=self.device)
            if neck_qpos.numel() >= 2:
                joint_target[:, self.neck_yaw_id] = neck_qpos[0].reshape(1)
                joint_target[:, self.neck_pitch_id] = neck_qpos[1].reshape(1)
            else:
                joint_target[:, self.neck_yaw_id] = neck_qpos.reshape(1)

        self.asset.set_joint_position_target(joint_target)
        self.asset.write_data_to_sim()

        self._applied_upper_body_q = q_applied_upper_body

    def debug_draw(self):
        return

    def _find_joint_id(self, joint_name: str) -> int:
        joint_ids, _ = self.asset.find_joints(fr"^{joint_name}$")
        if len(joint_ids) == 0:
            raise KeyError(f"Joint '{joint_name}' not found in robot asset.")
        return int(joint_ids[0])

    def _get_upper_body_q(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return joint_pos[:, self.upper_body_ids].clone()

    def _solve_dual_arm_ik(
        self,
        q_current_upper_body: torch.Tensor,
        left_target_pos: torch.Tensor,
        left_target_quat: torch.Tensor,
        right_target_pos: torch.Tensor,
        right_target_quat: torch.Tensor,
    ) -> torch.Tensor:
        left_target_pose = torch.cat((left_target_quat, left_target_pos), dim=1)
        right_target_pose = torch.cat((right_target_quat, right_target_pos), dim=1)
        q_target_np = self.dual_arm_ik.solve(
            controlled_q=q_current_upper_body[0].detach().cpu().numpy(),
            left_flange_pose_in_root=left_target_pose[0].detach().cpu().numpy(),
            right_flange_pose_in_root=right_target_pose[0].detach().cpu().numpy(),
        )
        return torch.from_numpy(q_target_np).to(device=self.device, dtype=torch.float32).unsqueeze(0)

    def _flange_pose_tensors_from_q(self, upper_body_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_pose_np, right_pose_np = self.dual_arm_ik.flange_pose_arrays_from_q(
            upper_body_q[0].detach().cpu().numpy()
        )
        left_pose = torch.from_numpy(left_pose_np).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        right_pose = torch.from_numpy(right_pose_np).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        return left_pose, right_pose

    def _compose_action_state(self, joint_pos: torch.Tensor) -> torch.Tensor:
        upper_body_q = self._get_upper_body_q(joint_pos)
        left_pose, right_pose = self._flange_pose_tensors_from_q(upper_body_q)

        right_quat = right_pose[:, :4]
        right_pos = right_pose[:, 4:]
        left_quat = left_pose[:, :4]
        left_pos = left_pose[:, 4:]

        right_rot6d = quaternion_to_rot6d(right_quat)
        left_rot6d = quaternion_to_rot6d(left_quat)

        right_hand = joint_pos[:, self.right_hand_ids].clone()
        left_hand = joint_pos[:, self.left_hand_ids].clone()

        return torch.cat(
            (
                right_pos,
                right_rot6d,
                right_hand,
                left_pos,
                left_rot6d,
                left_hand,
            ),
            dim=1,
        )

    def compose_dataset_state(self, joint_pos: torch.Tensor | None = None) -> torch.Tensor:
        if joint_pos is None:
            joint_pos = self.asset.data.joint_pos

        upper_body_q = self._get_upper_body_q(joint_pos)
        left_pose, right_pose = self._flange_pose_tensors_from_q(upper_body_q)

        right_arm = joint_pos[:, self.right_arm_ids].clone()
        left_arm = joint_pos[:, self.left_arm_ids].clone()

        right_quat_wxyz = right_pose[:, :4]
        right_pos = right_pose[:, 4:]
        left_quat_wxyz = left_pose[:, :4]
        left_pos = left_pose[:, 4:]

        right_quat_xyzw = torch.cat((right_quat_wxyz[:, 1:], right_quat_wxyz[:, :1]), dim=1)
        left_quat_xyzw = torch.cat((left_quat_wxyz[:, 1:], left_quat_wxyz[:, :1]), dim=1)

        right_hand = joint_pos[:, self.right_hand_ids].clone()
        left_hand = joint_pos[:, self.left_hand_ids].clone()
        waist = joint_pos[:, self.dataset_waist_ids].clone()
        neck = joint_pos[:, self.dataset_neck_ids].clone()

        state = torch.cat(
            (
                right_arm,
                left_arm,
                right_pos,
                right_quat_xyzw,
                left_pos,
                left_quat_xyzw,
                right_hand,
                left_hand,
                waist,
                neck,
            ),
            dim=1,
        )
        if state.shape[1] != self.DATASET_STATE_DIM:
            raise RuntimeError(
                f"Expected dataset state dim {self.DATASET_STATE_DIM}, got {state.shape[1]}"
            )
        return state

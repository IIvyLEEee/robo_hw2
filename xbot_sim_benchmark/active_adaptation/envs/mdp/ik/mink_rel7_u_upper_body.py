from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import mink


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
REL7_U_DIR = PACKAGE_ROOT / "assets" / "robots" / "rel7_u"
MODEL_XML = REL7_U_DIR / "rel7_u_mujoco.xml"


class Rel7UUpperBodyMinkIK:
    ROOT_BODY = "waist_base_link"
    LEFT_FLANGE_BODY = "left_hand_task_link"
    RIGHT_FLANGE_BODY = "right_hand_task_link"

    WAIST_JOINT_NAMES = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    LEFT_ARM_JOINT_NAMES = (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
    )
    RIGHT_ARM_JOINT_NAMES = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_arm_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    )
    CONTROLLED_JOINT_NAMES = (
        *WAIST_JOINT_NAMES,
        *LEFT_ARM_JOINT_NAMES,
        *RIGHT_ARM_JOINT_NAMES,
    )
    WAIST_Q_LIMITS = {
        "waist_yaw_joint": (-0.45, 0.45),
        "waist_roll_joint": (-0.12, 0.12),
        "waist_pitch_joint": (-0.05, 0.35),
    }

    def __init__(
        self,
        model_path: str | Path = MODEL_XML,
        *,
        solver: str = "daqp",
        position_cost: float = 200.0,
        orientation_cost: float = 10.0,
        arm_posture_cost: float = 1.0,
        waist_yaw_posture_cost: float = 5.0,
        waist_pitch_posture_cost: float = 5.0,
        waist_roll_posture_cost: float = 40.0,
        damping: float = 1e-4,
        default_dt: float = 0.01,
        max_velocity: float = 2.0,
        waist_max_velocity: float = 0.05,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path.as_posix())
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data

        self.solver = solver
        self.default_dt = float(default_dt)
        self.damping = float(damping)
        self._waist_q_limits = {
            joint_name: tuple(float(v) for v in self.WAIST_Q_LIMITS[joint_name])
            for joint_name in self.WAIST_JOINT_NAMES
        }

        self.left_flange_task = mink.RelativeFrameTask(
            frame_name=self.LEFT_FLANGE_BODY,
            frame_type="body",
            root_name=self.ROOT_BODY,
            root_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1e-3,
        )
        self.right_flange_task = mink.RelativeFrameTask(
            frame_name=self.RIGHT_FLANGE_BODY,
            frame_type="body",
            root_name=self.ROOT_BODY,
            root_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1e-3,
        )

        all_joint_names = tuple(self.model.joint(i).name for i in range(self.model.njnt))
        self.frozen_joint_names = tuple(
            joint_name
            for joint_name in all_joint_names
            if joint_name not in self.CONTROLLED_JOINT_NAMES
        )

        posture_weights = np.zeros(self.model.nv, dtype=np.float64)
        for joint_name in (*self.LEFT_ARM_JOINT_NAMES, *self.RIGHT_ARM_JOINT_NAMES):
            posture_weights[self._joint_dof_index(joint_name)] = arm_posture_cost
        posture_weights[self._joint_dof_index("waist_yaw_joint")] = waist_yaw_posture_cost
        posture_weights[self._joint_dof_index("waist_pitch_joint")] = waist_pitch_posture_cost
        posture_weights[self._joint_dof_index("waist_roll_joint")] = waist_roll_posture_cost
        self.posture_task = mink.PostureTask(self.model, cost=posture_weights)

        frozen_dofs = [self._joint_dof_index(joint_name) for joint_name in self.frozen_joint_names]
        self.freeze_task = mink.DofFreezingTask(self.model, dof_indices=frozen_dofs)

        self.tasks = [
            self.left_flange_task,
            self.right_flange_task,
            self.posture_task,
        ]
        self.constraints = [self.freeze_task]
        velocity_limits = {joint_name: max_velocity for joint_name in self.CONTROLLED_JOINT_NAMES}
        for joint_name in self.WAIST_JOINT_NAMES:
            velocity_limits[joint_name] = float(waist_max_velocity)
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(
                self.model,
                velocity_limits,
            ),
        ]

        self._controlled_qpos_indices = np.array(
            [self._joint_qpos_index(joint_name) for joint_name in self.CONTROLLED_JOINT_NAMES],
            dtype=np.int32,
        )
        self._controlled_joint_index = {
            joint_name: index for index, joint_name in enumerate(self.CONTROLLED_JOINT_NAMES)
        }

        self._nominal_q = self._build_nominal_configuration()
        self._posture_target_q = self._nominal_q.copy()
        self._posture_target_q[self._joint_qpos_index("left_elbow_pitch_joint")] = -np.pi / 2
        self._posture_target_q[self._joint_qpos_index("right_elbow_pitch_joint")] = -np.pi / 2
        self._posture_target_q[self._joint_qpos_index("left_elbow_yaw_joint")] = -np.pi / 2
        self._posture_target_q[self._joint_qpos_index("right_elbow_yaw_joint")] = np.pi / 2
        self._frozen_joint_qpos = {
            joint_name: self._nominal_q[self._joint_qpos_index(joint_name)]
            for joint_name in self.frozen_joint_names
        }

        self.reset()

    def reset(self) -> None:
        self.configuration.update(self._posture_target_q.copy())
        self._project_to_nominal_frozen_pose()
        self.posture_task.set_target(self._posture_target_q)
        self.left_flange_task.set_target(self.left_flange_pose())
        self.right_flange_task.set_target(self.right_flange_pose())

    def posture_target_vector(self) -> np.ndarray:
        return self._extract_controlled_q(self._posture_target_q)

    def set_posture_target(self, controlled_q: np.ndarray, *, reset: bool = True) -> None:
        q = self._nominal_q.copy()
        q[self._controlled_qpos_indices] = np.asarray(controlled_q, dtype=np.float64)
        self._posture_target_q = q
        self.posture_task.set_target(self._posture_target_q)
        if reset:
            self.reset()

    def joint_index(self, joint_name: str) -> int:
        return self._controlled_joint_index[joint_name]

    def update_configuration(self, controlled_q: np.ndarray) -> None:
        q = self._nominal_q.copy()
        q[self._controlled_qpos_indices] = np.asarray(controlled_q, dtype=np.float64)
        self.configuration.update(q)
        self._clip_waist_qpos_inplace()
        self._project_to_nominal_frozen_pose()

    def controlled_joint_vector(self) -> np.ndarray:
        return self._extract_controlled_q(self.data.qpos)

    def solve(
        self,
        controlled_q: np.ndarray,
        left_flange_pose_in_root: mink.SE3 | np.ndarray,
        right_flange_pose_in_root: mink.SE3 | np.ndarray,
        *,
        dt: float | None = None,
        max_iters: int = 100,
        position_tol: float = 1e-3,
        orientation_tol: float = 1e-2,
    ) -> np.ndarray:
        self.update_configuration(controlled_q)
        dt = self.default_dt if dt is None else float(dt)
        left_target = self._coerce_se3(left_flange_pose_in_root)
        right_target = self._coerce_se3(right_flange_pose_in_root)

        for _ in range(max_iters):
            self.left_flange_task.set_target(left_target)
            self.right_flange_task.set_target(right_target)
            velocity = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                self.solver,
                damping=self.damping,
                limits=self.limits,
                constraints=self.constraints,
            )
            self.configuration.integrate_inplace(velocity, dt)
            self._clip_waist_qpos_inplace()
            self._project_to_nominal_frozen_pose()
            pos_err, rot_err = self.flange_errors()
            if pos_err <= position_tol and rot_err <= orientation_tol:
                break

        return self.controlled_joint_vector()

    def left_flange_pose(self) -> mink.SE3:
        return self.configuration.get_transform(
            self.LEFT_FLANGE_BODY, "body", self.ROOT_BODY, "body"
        )

    def right_flange_pose(self) -> mink.SE3:
        return self.configuration.get_transform(
            self.RIGHT_FLANGE_BODY, "body", self.ROOT_BODY, "body"
        )

    def flange_pose_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.left_flange_pose().wxyz_xyz.copy()
        right = self.right_flange_pose().wxyz_xyz.copy()
        return left, right

    def flange_pose_arrays_from_q(self, controlled_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.update_configuration(controlled_q)
        return self.flange_pose_arrays()

    def flange_errors(self) -> tuple[float, float]:
        errors = (
            self.left_flange_task.compute_error(self.configuration),
            self.right_flange_task.compute_error(self.configuration),
        )
        position_error = max(float(np.linalg.norm(error[:3])) for error in errors)
        orientation_error = max(float(np.linalg.norm(error[3:])) for error in errors)
        return position_error, orientation_error

    def _extract_controlled_q(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q[self._controlled_qpos_indices], dtype=np.float64).copy()

    def _build_nominal_configuration(self) -> np.ndarray:
        q = self.model.qpos0.copy()
        for joint_name in (*self.CONTROLLED_JOINT_NAMES, *self.frozen_joint_names):
            q[self._joint_qpos_index(joint_name)] = 0.0
        return q

    def _project_to_nominal_frozen_pose(self) -> None:
        q = self.data.qpos.copy()
        for joint_name, value in self._frozen_joint_qpos.items():
            q[self._joint_qpos_index(joint_name)] = value
        self.configuration.update(q)

    def _clip_waist_qpos_inplace(self) -> None:
        q = self.data.qpos.copy()
        for joint_name, (q_min, q_max) in self._waist_q_limits.items():
            q_index = self._joint_qpos_index(joint_name)
            q[q_index] = np.clip(q[q_index], q_min, q_max)
        self.configuration.update(q)

    def _joint_qpos_index(self, joint_name: str) -> int:
        joint_id = self.model.joint(joint_name).id
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dof_index(self, joint_name: str) -> int:
        joint_id = self.model.joint(joint_name).id
        return int(self.model.jnt_dofadr[joint_id])

    @staticmethod
    def _coerce_se3(pose: mink.SE3 | np.ndarray) -> mink.SE3:
        if isinstance(pose, mink.SE3):
            return pose.normalize()
        array = np.asarray(pose, dtype=np.float64)
        if array.shape == (7,):
            return mink.SE3(wxyz_xyz=array).normalize()
        if array.shape == (4, 4):
            return mink.SE3.from_matrix(array).normalize()
        raise ValueError("Pose must be a mink.SE3, a 7D [wxyz_xyz] array, or a 4x4 matrix.")

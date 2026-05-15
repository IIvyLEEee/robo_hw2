import io
import sys
from pathlib import Path
from typing import Callable, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
import zmq
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher
from utils.helpers import make_env
from utils.debugger import (
    SuccessLogger,
    TrajectoryRecorder,
    make_print_every_callback,
)
from utils.vla_runner import run_env_loop


DEFAULT_IMAGE_KEYS = {
    "cam_high": "world_cam",
    "cam_left_wrist": "left_cam",
    "cam_right_wrist": "right_cam",
}


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    rot = R.from_quat(quat_xyzw).as_matrix()
    return rot[:2, :].reshape(6).astype(np.float32)


def _decode_action_38_to_42(action_38: np.ndarray) -> np.ndarray:
    action_38 = np.asarray(action_38, dtype=np.float32)
    if action_38.shape != (38,):
        raise ValueError(f"Expected 38-dim action, got {action_38.shape}")

    right_pos = action_38[0:3]
    right_quat_xyzw = action_38[3:7]
    right_hand = action_38[7:19]
    left_pos = action_38[19:22]
    left_quat_xyzw = action_38[22:26]
    left_hand = action_38[26:38]

    right_rot6d = _quat_xyzw_to_rot6d(right_quat_xyzw)
    left_rot6d = _quat_xyzw_to_rot6d(left_quat_xyzw)
    return np.concatenate(
        [right_pos, right_rot6d, right_hand, left_pos, left_rot6d, left_hand],
        axis=0,
    ).astype(np.float32)


def _decode_action_batch(reply: Any) -> np.ndarray:
    if isinstance(reply, dict):
        if "error" in reply:
            raise RuntimeError(f"Error from VLA server: {reply['error']}")
        if "actions" not in reply:
            raise RuntimeError("VLA server reply dict must contain 'actions'.")
        reply = reply["actions"]

    actions = np.asarray(reply, dtype=np.float32)
    if actions.ndim == 2:
        actions = actions[None, ...]
    if actions.ndim != 3 or actions.shape[-1] != 38:
        raise RuntimeError(f"Expected reply shape [B,T,38] or [T,38], got {actions.shape}")

    decoded = np.empty((actions.shape[0], actions.shape[1], 42), dtype=np.float32)
    for batch_idx in range(actions.shape[0]):
        for step_idx in range(actions.shape[1]):
            decoded[batch_idx, step_idx] = _decode_action_38_to_42(actions[batch_idx, step_idx])
    return decoded


def _ensure_image_keys(cfg) -> Dict[str, str]:
    image_keys = dict(getattr(cfg.task, "vla_image_keys", {}) or {})
    if image_keys:
        return image_keys
    return DEFAULT_IMAGE_KEYS.copy()


def _resolve_traj_out_root(cfg) -> Optional[str]:
    vla_cfg = getattr(cfg, "vla", None)
    if vla_cfg is not None:
        traj_out_root = vla_cfg.get("trajectory_output_dir", "")
        if traj_out_root is not None and str(traj_out_root):
            return str(traj_out_root)
    return HydraConfig.get().runtime.output_dir


def _build_model_state(env, env_ids: torch.Tensor) -> np.ndarray:
    if not hasattr(env.action_manager, "compose_dataset_state"):
        raise RuntimeError("Current action manager does not implement compose_dataset_state().")
    state = env.action_manager.compose_dataset_state()[env_ids]
    return state.detach().cpu().numpy().astype(np.float32)


def _blend_chunk_prefix(
    decoded: np.ndarray,
    env_ids: torch.Tensor,
    previous_overlap: dict[int, np.ndarray],
    *,
    execute_steps: int,
    overlap_start: int,
    overlap_len: int,
    alpha: float,
) -> np.ndarray:
    blended = decoded.copy()
    for batch_index, env_id_t in enumerate(env_ids.tolist()):
        prev = previous_overlap.get(int(env_id_t))
        if prev is None:
            continue
        blend_len = min(execute_steps, overlap_len, prev.shape[0], blended.shape[1])
        if blend_len <= 0:
            continue
        blended[batch_index, :blend_len] = (
            (1.0 - alpha) * prev[:blend_len] + alpha * blended[batch_index, :blend_len]
        )
    return blended


@hydra.main(version_base=None, config_path="../cfg", config_name="base")
def main(cfg):
    cfg.app.headless = False
    cfg.app.enable_cameras = True
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    object_select: Optional[list[int]] = getattr(cfg.task, "object_select", None)
    if object_select is not None:
        origin = cfg.task.object_collection.task_objects
        new = []
        for i in object_select:
            new.append(origin[i])
        cfg.task.object_collection.task_objects = new
        cfg.task.task.params.task_objects = new

    if cfg.app.get("device", None) is not None:
        cfg.task.sim_device = cfg.app.device

    env = make_env(cfg)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    vla_cfg = getattr(cfg, "vla", None)
    server_addr = "tcp://10.7.0.1:8003"
    if vla_cfg is not None and vla_cfg.get("server", None):
        server_addr = str(vla_cfg.get("server"))
    socket.connect(server_addr)

    max_rollouts: int = int(getattr(cfg.task, "max_rollouts", 10))
    completed_rollouts: list[int] = [0]
    stop_requested: list[bool] = [False]
    execute_steps = 5
    overlap_start = 5
    overlap_len = 5
    smoothing_alpha = 0.5
    previous_overlap: dict[int, np.ndarray] = {}

    def reset_callback(done_mask: torch.Tensor):
        if done_mask is None:
            return
        for env_id in torch.nonzero(done_mask, as_tuple=False).squeeze(-1).tolist():
            previous_overlap.pop(int(env_id), None)
        n_done = int(done_mask.sum().item())
        if n_done <= 0:
            return
        completed_rollouts[0] += n_done
        if completed_rollouts[0] >= max_rollouts:
            stop_requested[0] = True

    def vla_query_callback(batch: Dict[str, Any]) -> torch.Tensor:
        if stop_requested[0]:
            print(f"[VLA TEST] Max rollouts reached ({completed_rollouts[0]} >= {max_rollouts}). Exiting loop.")
            return False

        env_ids = batch["env_ids"]
        payload = {
            "images": batch["images"],
            "state": _build_model_state(env, env_ids),
            "text": batch["text"],
        }

        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        socket.send(buf.getvalue())
        reply = socket.recv_pyobj()
        decoded = _decode_action_batch(reply)
        decoded = _blend_chunk_prefix(
            decoded,
            env_ids,
            previous_overlap,
            execute_steps=execute_steps,
            overlap_start=overlap_start,
            overlap_len=overlap_len,
            alpha=smoothing_alpha,
        )
        for batch_index, env_id_t in enumerate(env_ids.tolist()):
            start = overlap_start
            end = min(decoded.shape[1], overlap_start + overlap_len)
            if end > start:
                previous_overlap[int(env_id_t)] = decoded[batch_index, start:end].copy()
            else:
                previous_overlap.pop(int(env_id_t), None)
        return torch.from_numpy(decoded).to(batch["device"])

    image_keys = _ensure_image_keys(cfg)

    traj_out_root = _resolve_traj_out_root(cfg)

    debug_callbacks: list[Callable[[int, Dict[str, Any]], None]] = [
        make_print_every_callback(50),
        SuccessLogger(),
        TrajectoryRecorder(
            image_keys=image_keys,
            out_root=traj_out_root,
            fps=int(round(1.0 / env.step_dt)) if getattr(env, "step_dt", 0) else 10,
        ),
    ]

    try:
        run_env_loop(
            env,
            query_callback=vla_query_callback,
            reset_callback=reset_callback,
            debug_callbacks=debug_callbacks,
            T=execute_steps,
            default_after_reset=5,
            image_keys=image_keys,
        )
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        socket.close(linger=0)
        context.term()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

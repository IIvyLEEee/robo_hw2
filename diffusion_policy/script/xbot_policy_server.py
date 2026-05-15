import argparse
import io
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_key_map(items: List[str]) -> Dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected mapping in obs_key=payload_key form, got: {item}")
        obs_key, payload_key = item.split("=", 1)
        result[obs_key] = payload_key
    return result


def load_policy(checkpoint_path: str, device: str):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.get("use_ema", False):
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def decode_request(message: bytes) -> Dict[str, Any]:
    data = np.load(io.BytesIO(message), allow_pickle=True)
    payload = {}
    for key in data.files:
        value = data[key]
        if value.shape == () and value.dtype == object:
            value = value.item()
        payload[key] = value
    return payload


def prepare_rgb(images: Dict[str, np.ndarray], payload_key: str, shape: List[int]) -> np.ndarray:
    if payload_key not in images:
        raise KeyError(f"Missing image key '{payload_key}'. Available image keys: {list(images.keys())}")

    image = np.asarray(images[payload_key])
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected image '{payload_key}' with shape [B,H,W,C], got {image.shape}")

    channels, out_h, out_w = tuple(shape)
    if image.shape[-1] != channels:
        raise ValueError(f"Expected image '{payload_key}' to have {channels} channels, got {image.shape[-1]}")

    in_h, in_w = image.shape[1:3]
    if (in_h, in_w) != (out_h, out_w):
        transform = get_image_transform(input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=False)
        image = np.stack([transform(frame) for frame in image], axis=0)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    return np.moveaxis(image, -1, 1)


def prepare_low_dim(payload: Dict[str, Any], payload_key: str, shape: List[int]) -> np.ndarray:
    if payload_key not in payload:
        raise KeyError(f"Missing low-dimensional key '{payload_key}'. Available payload keys: {list(payload.keys())}")

    value = np.asarray(payload[payload_key], dtype=np.float32)
    if value.ndim == 1:
        value = value[None]

    expected_dim = int(np.prod(shape))
    if value.shape[-1] < expected_dim:
        raise ValueError(f"Expected '{payload_key}' last dim >= {expected_dim}, got {value.shape}")
    if value.shape[-1] > expected_dim:
        value = value[..., :expected_dim]

    return value.reshape(value.shape[0], *shape)


def build_current_obs(
    payload: Dict[str, Any],
    shape_meta: Dict[str, Any],
    image_key_map: Dict[str, str],
    lowdim_key_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    images = payload.get("images", {})
    obs = {}

    for obs_key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        shape = attr["shape"]
        if obs_type == "rgb":
            payload_key = image_key_map.get(obs_key, obs_key)
            obs[obs_key] = prepare_rgb(images, payload_key, shape)
        else:
            payload_key = lowdim_key_map.get(obs_key, obs_key)
            obs[obs_key] = prepare_low_dim(payload, payload_key, shape)

    return obs


def stack_history(
    current_obs: Dict[str, np.ndarray],
    history: Dict[int, Deque[Dict[str, np.ndarray]]],
    n_obs_steps: int,
) -> Dict[str, np.ndarray]:
    batch_size = next(iter(current_obs.values())).shape[0]
    stacked = defaultdict(list)

    for batch_idx in range(batch_size):
        current_item = {key: value[batch_idx].copy() for key, value in current_obs.items()}
        hist = history[batch_idx]
        if len(hist) == 0:
            for _ in range(n_obs_steps - 1):
                hist.append(current_item)
        hist.append(current_item)

        for key in current_obs:
            stacked[key].append(np.stack([step[key] for step in hist], axis=0))

    return {key: np.stack(value, axis=0) for key, value in stacked.items()}


def pad_action_steps(actions: np.ndarray, min_action_steps: int) -> np.ndarray:
    if actions.ndim != 3:
        raise ValueError(f"Policy action must have shape [B,T,A], got {actions.shape}")
    if actions.shape[1] >= min_action_steps:
        return actions
    pad_count = min_action_steps - actions.shape[1]
    tail = np.repeat(actions[:, -1:, :], pad_count, axis=1)
    return np.concatenate([actions, tail], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Minimal ZMQ policy server for xbot_sim_benchmark.")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to a trained diffusion_policy checkpoint.")
    parser.add_argument("--bind", default="tcp://*:8003", help="ZMQ REP bind address.")
    parser.add_argument("-d", "--device", default="cuda:0", help="Torch device used by the policy.")
    parser.add_argument("--image-key-map", nargs="*", default=["image=cam_high"])
    parser.add_argument("--lowdim-key-map", nargs="*", default=["state=state"])
    parser.add_argument("--min-action-steps", type=int, default=5)
    args = parser.parse_args()

    policy, cfg = load_policy(args.checkpoint, args.device)
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    n_obs_steps = int(cfg.n_obs_steps)
    image_key_map = parse_key_map(args.image_key_map)
    lowdim_key_map = parse_key_map(args.lowdim_key_map)
    history = defaultdict(lambda: deque(maxlen=n_obs_steps))

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)

    print(f"[xbot_policy_server] listening on {args.bind}")
    print(f"[xbot_policy_server] checkpoint: {args.checkpoint}")
    print(f"[xbot_policy_server] device: {args.device}, n_obs_steps: {n_obs_steps}")
    print(f"[xbot_policy_server] image_key_map: {image_key_map}")
    print(f"[xbot_policy_server] lowdim_key_map: {lowdim_key_map}")

    try:
        while True:
            message = socket.recv()
            try:
                payload = decode_request(message)
                current_obs = build_current_obs(payload, shape_meta, image_key_map, lowdim_key_map)
                obs_np = stack_history(current_obs, history, n_obs_steps)
                obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).to(args.device))

                with torch.no_grad():
                    result = policy.predict_action(obs)

                actions = result["action"].detach().cpu().numpy().astype(np.float32)
                actions = pad_action_steps(actions, args.min_action_steps)
                socket.send_pyobj({"actions": actions})
            except Exception as exc:
                socket.send_pyobj({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()

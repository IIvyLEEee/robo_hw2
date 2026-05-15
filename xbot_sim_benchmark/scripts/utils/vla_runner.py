import itertools
from typing import Callable, Optional, Sequence, Dict, Any

import numpy as np
import torch


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert camera tensor to HWC uint8 numpy.
    Accepts [N, C, H, W] or [N, 3*H*W] (flattened, 256x256 fallback).
    """
    if img.ndim == 4:
        # [N, C, H, W] -> [N, H, W, C]
        return img.permute(0, 2, 3, 1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported camera tensor shape: {tuple(img.shape)}")


def build_query_batch(td, env_ids, env, image_keys: Dict[str, str]) -> Dict[str, Any]:
    """Pack tensors from *td* into a dictionary that the policy server (VLA) understands."""
    images = {}
    for key, value in image_keys.items():
        images[key] = _to_hwc_uint8(td[value][env_ids])

    action_state = (
        td["action_state"][env_ids]
        .reshape(-1, env.action_manager.action_dim)
        .detach()
        .cpu()
        .numpy()
    )

    text_prompt = [env.extra["text_prompts"][int(i)] for i in env_ids]

    batch: Dict[str, Any] = {
        "images": images,
        "state": action_state,
        "text": text_prompt,
        "env_ids": env_ids,
        "device": td["action"].device,
    }

    return batch

def run_env_loop(
    env,
    *,
    image_keys: Dict[str, str],
    query_callback: Callable[[Dict[str, Any]], Optional[torch.Tensor]],
    reset_callback: Optional[Callable[[Dict[str, Any]], Optional[torch.Tensor]]] = None,
    debug_callbacks: Optional[Sequence[Callable[[int, Dict[str, Any]], None]]] = None,
    T: int = 5,
    default_after_reset: int = 5,
):
    """Unified environment episode loop with pluggable callbacks."""

    td_ = env.reset()
    td_["action"] = env.action_spec.zeros()

    action_dim = env.action_manager.action_dim
    num_envs = env.num_envs
    device = td_["action"].device

    # ── Buffers ──────────────────────────────────────────────────────────────
    action_buffer = torch.zeros(num_envs, T, action_dim, device=device)
    action_step = torch.zeros(num_envs, dtype=torch.int64, device=device)
    default_step_counter = torch.zeros(num_envs, dtype=torch.int64, device=device)
    needs_default_step = torch.ones(num_envs, dtype=torch.bool, device=device)
    default_action_state = td_["action_state"].reshape(-1, action_dim).clone()

    step_counter = 0

    # Ensure we iterate over an empty list instead of `None` for callbacks
    debug_callbacks = list(debug_callbacks or [])

    for _ in itertools.count():
        # 1) Determine which envs still need blank steps
        needs_default_step[:] = default_step_counter < default_after_reset
        envs_to_query = torch.nonzero((action_step == 0) & ~needs_default_step, as_tuple=False).squeeze(1)

        # 2) Query policy server when needed
        if envs_to_query.numel() > 0:
            batch = build_query_batch(td_, envs_to_query, env, image_keys)
            recv_actions = query_callback(batch)
            if recv_actions is False:
                break
            if len(recv_actions) == 2:
                recv_actions, extra = recv_actions
                for key, value in extra.items():
                    env.extra[key] = value

            if recv_actions is not None:
                action_buffer[envs_to_query] = recv_actions[:, :T, :]

        # 3) Take the correct sub‑action from buffer
        current_actions = action_buffer[torch.arange(num_envs), action_step]

        action_step += 1
        action_step = torch.where(
            torch.logical_or(action_step >= T, needs_default_step),
            torch.zeros_like(action_step),
            action_step,
        )

        # 4) Override with default action when still in blank‑step period
        current_actions[needs_default_step] = default_action_state[needs_default_step]
        default_step_counter[needs_default_step] += 1

        # 5) Step the env
        td_["action"][:] = current_actions
        td, td_ = env.step_and_maybe_reset(td_)

        # 6) Handle resets
        done = td["next", "done"].squeeze(-1)
        if done.any():
            action_step[done] = 0
            default_step_counter[done] = 0
            reset_callback(done)

        # 7) Invoke all debug callbacks
        ctx = locals()  # expose all live variables
        for cb in debug_callbacks:
            cb(step_counter, ctx)

        step_counter += 1
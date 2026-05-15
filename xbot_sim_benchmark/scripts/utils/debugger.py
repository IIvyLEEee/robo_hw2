import json
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.io import write_video

def make_print_every_callback(n: int = 50) -> Callable[[int, Dict[str, Any]], None]:
    """Factory for a simple step counter printer."""

    def _cb(step_idx: int, _ctx: Dict[str, Any]):
        if step_idx % n == 0:
            print(f"Step: {step_idx}")

    return _cb


class ActionStatePlotter:
    """Record action_state/target and dump plots at a given step (replicates the
    original commented-out debugging code)."""

    def __init__(self, action_dim: int, plot_step: int = 50, out_dir: str | os.PathLike = "dbg_plots"):
        self.action_dim = action_dim
        self.plot_step = plot_step
        self.out_dir = Path(out_dir)
        self.states: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, step_idx: int, ctx: Dict[str, Any]):
        td_ = ctx["td_"]
        current_actions = ctx["current_actions"]

        self.states.append(td_["action_state"][0].flatten().detach().cpu().numpy())
        self.targets.append(current_actions[0].flatten().detach().cpu().numpy())

        if step_idx == self.plot_step:
            states = np.asarray(self.states)
            targets = np.asarray(self.targets)
            for i in range(self.action_dim):
                plt.figure()
                plt.plot(states[:, i], label="action_state")
                plt.plot(targets[:, i], label="action_target")
                plt.legend()
                plt.savefig(self.out_dir / f"action_state_{i}.png")
                plt.close()


class WorldCamSaver:
    """Capture first *N* world-cam frames and write a playable MP4."""

    def __init__(
        self,
        cam_name: str = 'world_cam',
        save_first_n: int = 10,
        out_dir: str | Path = "",
        fps: int = 10,
        filename: str = "world_cam_video.mp4",
    ):
        self.save_first_n = save_first_n
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[torch.Tensor] = []
        self.fps = fps
        self.video_path = self.out_dir / filename
        self.cam_name = cam_name

    def __call__(self, step_idx: int, ctx: Dict[str, Any]):
        # When we've gathered enough frames, write out once then disable.
        if step_idx == self.save_first_n:
            if not self.frames:
                return
            video_tensor = torch.stack(self.frames)
            write_video(str(self.video_path), video_tensor, fps=self.fps)
            print(f"[WorldCamSaver] video saved to {self.video_path.resolve()}")
            self.frames.clear()
            return
        elif step_idx > self.save_first_n:
            return  # already saved

        # Collect frames (RGB uint8 expected by write_video)
        td_ = ctx["td_"]
        cam = td_[self.cam_name]
        # Support both non-flattened [N, C, H, W] and flattened [N, C*H*W]
        img_t = cam[0]
        if img_t.ndim == 3:
            img = img_t.permute(1, 2, 0).detach().cpu()
        else:
            raise ValueError(f"Unsupported camera tensor shape: {tuple(img_t.shape)}")

        assert img.dtype == torch.uint8

        self.frames.append(img)


class TrajectoryRecorder:
    """Record env0 trajectory and dump per-camera videos plus JSON files."""

    def __init__(
        self,
        image_keys: Dict[str, str],
        out_root: str | Path | None = None,
        fps: int = 10,
        json_name: str = "traj.json",
        plan_filename: str = "placement_plan.json",
    ):
        self.image_keys = dict(image_keys)
        self.out_root = Path(out_root) if out_root is not None else Path.cwd()
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.json_name = json_name
        self.plan_filename = plan_filename
        self._next_id = self._find_next_id()
        self._reset_buffers()

    def _find_next_id(self) -> int:
        existing = [
            int(p.name)
            for p in self.out_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        return (max(existing) + 1) if existing else 0

    def _reset_buffers(self):
        self.states: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.texts: list[str] = []
        self.frames: Dict[str, list[torch.Tensor]] = {k: [] for k in self.image_keys}
        self._last_text: Optional[str] = None
        self._text_recorded = False
        self._plan_recorded = False
        self._plan_data: Optional[dict] = None
        self._success: Optional[bool] = None

    def _resolve_text(self, env, ctx: Dict[str, Any]) -> Optional[str]:
        text = None
        extra = getattr(env, "extra", None)
        if isinstance(extra, dict):
            if "text" in extra:
                text = extra["text"]
            if isinstance(text, (list, tuple)):
                text = text[0] if text else None
            if text is None and "text_prompts" in extra:
                text = extra["text_prompts"]
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else None

        if text is None:
            batch = ctx.get("batch")
            if isinstance(batch, dict):
                batch_text = batch.get("text")
                if isinstance(batch_text, (list, tuple)):
                    text = batch_text[0] if batch_text else None

        if text is None:
            text = self._last_text
        if text is not None and not isinstance(text, str):
            text = str(text)
        if text is not None:
            self._last_text = text
        return text

    def _append_images(self, td_: Dict[str, Any]):
        for out_key, obs_key in self.image_keys.items():
            cam = td_[obs_key]
            img_t = cam[0]
            if img_t.ndim != 3:
                raise ValueError(f"Unsupported camera tensor shape: {tuple(img_t.shape)}")
            img = img_t.detach().cpu()
            if img.dtype != torch.uint8:
                if torch.is_floating_point(img):
                    img = (img * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    img = img.to(torch.uint8)
            img = img.permute(1, 2, 0)
            self.frames[out_key].append(img)

    def _flush(self):
        if not self.states and not any(self.frames.values()):
            self._reset_buffers()
            return

        while (self.out_root / str(self._next_id)).exists():
            self._next_id += 1
        out_dir = self.out_root / str(self._next_id)
        self._next_id += 1
        out_dir.mkdir(parents=True, exist_ok=False)

        payload = {
            "texts": self.texts,
            "states": self.states,
            "actions": self.actions,
            "success": self._success,
        }
        with open(out_dir / self.json_name, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True)

        if self._plan_data is not None:
            with open(out_dir / self.plan_filename, "w", encoding="utf-8") as fp:
                json.dump(self._plan_data, fp, ensure_ascii=True)

        for key, frames in self.frames.items():
            if not frames:
                continue
            video_tensor = torch.stack(frames)
            write_video(str(out_dir / f"{key}.mp4"), video_tensor, fps=self.fps)

        print(f"[TrajectoryRecorder] trajectory saved to {out_dir.resolve()}")
        self._reset_buffers()

    def __call__(self, step_idx: int, ctx: Dict[str, Any]):
        needs_default_step = ctx.get("needs_default_step")
        if needs_default_step is None:
            return

        if not self._plan_recorded:
            env = ctx.get("env")
            if env is not None and hasattr(env, "get_env0_plan"):
                plan = env.get_env0_plan()
                if plan is not None:
                    self._plan_data = plan
                    self._plan_recorded = True

        td_ = ctx.get("td_")
        current_actions = ctx.get("current_actions")
        if td_ is None or current_actions is None:
            return

        if not bool(needs_default_step[0].item()):
            state = td_["action_state"][0].flatten().detach().cpu().numpy().tolist()
            action = current_actions[0].flatten().detach().cpu().numpy().tolist()
            self.states.append(state)
            self.actions.append(action)

            if not self._text_recorded:
                env = ctx.get("env")
                if env is not None:
                    text = self._resolve_text(env, ctx)
                    self.texts.append(text or "")
                else:
                    self.texts.append("")
                self._text_recorded = True

            self._append_images(td_)

        done = ctx.get("done")
        if done is None:
            return
        if bool(done[0].item()):
            td = ctx.get("td")
            if td is not None and ("next", "terminated") in td:
                terminated = td["next", "terminated"].squeeze(-1)
                self._success = bool(terminated[0].item())
            else:
                self._success = False
            self._flush()

class SuccessLogger:
    """
    Log success/failure when environments finish and maintain cumulative success rate.
    Usage: register as a callback, expects ctx["td_"]["next", "done"] and ctx["td_"]["next", "terminated"].
    """

    def __init__(self):
        self.total = 0      # 总共结束了多少次
        self.success = 0    # 总共成功了多少次

    def __call__(self, step_idx: int, ctx: dict):
        td = ctx["td"] # td, not td_
        done = td["next", "done"].squeeze(-1).cpu().numpy()           # shape: [N]
        terminated = td["next", "terminated"].squeeze(-1).cpu().numpy()  # shape: [N]

        done_env_ids = np.where(done == 1)[0]        # 哪些环境结束了
        if len(done_env_ids) == 0:
            return  # 没有环境刚刚结束

        # 区分成功和失败
        success_envs = done_env_ids[terminated[done_env_ids] == 1]
        fail_envs = done_env_ids[terminated[done_env_ids] == 0]

        # 累计计数
        self.total += len(done_env_ids)
        self.success += len(success_envs)

        # 打印信息
        print(f"[Step {step_idx}]")
        print(f"  Success env ids: {success_envs.tolist()}")
        print(f"  Fail env ids: {fail_envs.tolist()}")
        if self.total > 0:
            succ_rate = self.success / self.total
            print(f"  [Cumulative success rate]: {self.success}/{self.total} = {succ_rate:.3f}")
        else:
            print("  [Cumulative success rate]: N/A")

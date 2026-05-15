import os
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from isaaclab.utils.math import quat_from_matrix

from active_adaptation.assets import ASSET_PATH


class StaticGSRenderer:
    def __init__(self, device: str):
        self.device = device
        self._means: List[torch.Tensor] = []
        self._quats: List[torch.Tensor] = []
        self._scales: List[torch.Tensor] = []
        self._opacities: List[torch.Tensor] = []
        self._colors: List[torch.Tensor] = []
        self.sh_degree: Optional[int] = None
        self.means = None
        self.quats = None
        self.scales = None
        self.opacities = None
        self.colors = None

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _apply_transform(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        transform: torch.Tensor,
        scale: float,
    ):
        means = means.to(self.device)
        quats = quats.to(self.device)
        scales = scales.to(self.device)
        transform = transform.to(self.device)

        # Transform is expected to map GS local coordinates into the robot root frame.
        rot = transform[:3, :3]
        trans = transform[:3, 3]

        means = means * scale
        means = torch.matmul(means, rot.T) + trans
        scales = scales * scale

        rot_quat = quat_from_matrix(rot)
        if rot_quat.dim() == 1:
            rot_quat = rot_quat.unsqueeze(0)
        if rot_quat.shape[0] == 1 and quats.shape[0] > 1:
            rot_quat = rot_quat.expand(quats.shape[0], -1)
        quats = self._quat_mul(rot_quat, quats)
        quats = F.normalize(quats, dim=-1)
        return means, quats, scales

    def _load_ply(self, path: str, sh_layout: str):
        import trimesh

        cloud = trimesh.load(path, process=False)
        data = cloud.metadata["_ply_raw"]["vertex"]["data"]

        means = torch.stack(
            [torch.from_numpy(data["x"]), torch.from_numpy(data["y"]), torch.from_numpy(data["z"])],
            dim=-1,
        ).to(torch.float32)

        opacity_key = "opacity" if "opacity" in data.dtype.names else "opacities"
        opacities = torch.from_numpy(data[opacity_key]).to(torch.float32).unsqueeze(-1)
        opacities = torch.sigmoid(opacities)

        scales = torch.stack(
            [torch.from_numpy(data["scale_0"]), torch.from_numpy(data["scale_1"]), torch.from_numpy(data["scale_2"])],
            dim=-1,
        ).to(torch.float32)
        scales = torch.exp(scales)

        rotations = torch.stack(
            [
                torch.from_numpy(data["rot_0"]),
                torch.from_numpy(data["rot_1"]),
                torch.from_numpy(data["rot_2"]),
                torch.from_numpy(data["rot_3"]),
            ],
            dim=-1,
        ).to(torch.float32)
        rotations = F.normalize(rotations, dim=-1)

        f_dc = torch.stack(
            [torch.from_numpy(data["f_dc_0"]), torch.from_numpy(data["f_dc_1"]), torch.from_numpy(data["f_dc_2"])],
            dim=-1,
        ).to(torch.float32)
        f_rest_keys = [k for k in data.dtype.names if k.startswith("f_rest_")]
        f_rest_keys = sorted(f_rest_keys, key=lambda k: int(k.split("_")[-1]))
        if f_rest_keys:
            f_rest = torch.stack([torch.from_numpy(data[k]) for k in f_rest_keys], dim=-1).to(torch.float32)
            if sh_layout == "channel_first":
                f_rest = f_rest.reshape(f_rest.shape[0], 3, -1).transpose(1, 2)
            elif sh_layout == "interleaved":
                f_rest = f_rest.reshape(f_rest.shape[0], -1, 3)
            else:
                raise ValueError(f"Unknown sh_layout '{sh_layout}' for 3DGS PLY")
            shs = torch.cat([f_dc[:, None, :], f_rest], dim=1)
        else:
            shs = f_dc[:, None, :]

        return means, rotations, scales, opacities, shs

    def _load_pth(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if "splats" in ckpt:
            ckpt = ckpt["splats"]

        means = ckpt["means"].to(torch.float32)
        quats = F.normalize(ckpt["quats"].to(torch.float32), p=2, dim=-1)
        scales = torch.exp(ckpt["scales"].to(torch.float32))
        opacities = torch.sigmoid(ckpt["opacities"].to(torch.float32))
        sh0 = ckpt["sh0"].to(torch.float32)
        shn = ckpt["shN"].to(torch.float32)
        colors = torch.cat([sh0, shn], dim=-2)
        return means, quats, scales, opacities, colors

    def add_from_path(self, path: str, sh_layout: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".pt", ".pth"):
            means, quats, scales, opacities, colors = self._load_pth(path)
        else:
            means, quats, scales, opacities, colors = self._load_ply(path, sh_layout)

        if self.sh_degree is None:
            coeffs = colors.shape[1]
            # Pick the largest SH degree whose coefficient count fits the tensor.
            # This keeps truncated SH sets valid, e.g. DC-only => degree 0.
            self.sh_degree = max(0, math.isqrt(int(coeffs)) - 1)

        self._means.append(means.to(self.device))
        self._quats.append(quats.to(self.device))
        self._scales.append(scales.to(self.device))
        self._opacities.append(opacities.to(self.device))
        self._colors.append(colors.to(self.device))

    def finalize(self):
        if not self._means:
            return
        self.means = torch.cat(self._means, dim=0)
        self.quats = torch.cat(self._quats, dim=0)
        self.scales = torch.cat(self._scales, dim=0)
        self.opacities = torch.cat(self._opacities, dim=0).reshape(-1)
        self.colors = torch.cat(self._colors, dim=0)

    @torch.no_grad()
    def render(self, pose: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
        from gsplat.rendering import rasterization

        pose = pose.to(self.device)
        K = K.to(self.device)
        if pose.dim() == 2:
            pose = pose.unsqueeze(0)
        if K.dim() == 2:
            K = K.unsqueeze(0)

        batch = pose.shape[0]
        background = torch.zeros((batch, 3), device=self.device, dtype=torch.float32)
        viewmats = torch.linalg.inv(pose)
        render_colors, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmats,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            camera_model="pinhole",
            rasterize_mode="antialiased",
            sh_degree=3 if self.sh_degree is None else self.sh_degree,
            near_plane=0.001,
            far_plane=100.0,
            radius_clip=0.0,
            eps2d=0.3,
            render_mode="RGB",
            backgrounds=background,
            with_ut=False,
            with_eval3d=False,
        )
        return (render_colors.clip(0, 1) * 255.0).to(torch.uint8)


class Background3DGSManager:
    def __init__(self, cfg, device: str):
        self.device = device
        self._renderer: Optional[StaticGSRenderer] = None
        self._gs_transform_inv: Optional[torch.Tensor] = None
        self._offset_camera = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._init_from_cfg(cfg)

    @staticmethod
    def _normalize_3dgs_cfg(cfg_3dgs) -> List[DictConfig]:
        if cfg_3dgs is None:
            return []
        if isinstance(cfg_3dgs, (list, tuple, ListConfig)):
            return list(cfg_3dgs)
        if isinstance(cfg_3dgs, (dict, DictConfig)):
            if "path" in cfg_3dgs:
                return [cfg_3dgs]
            return list(cfg_3dgs.values())
        return []

    @staticmethod
    def _resolve_3dgs_path(path: str) -> str:
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(ASSET_PATH, path))

    @staticmethod
    def _parse_3dgs_transform(entry: DictConfig) -> Tuple[torch.Tensor, float]:
        scale = 1.0
        if hasattr(entry, "scale") and entry.scale is not None:
            scale = float(entry.scale)

        y_up = False
        if hasattr(entry, "y_up") and entry.y_up is not None:
            y_up = bool(entry.y_up)

        yup_to_zup = torch.tensor(
            [
                [-1,0,0,0],
                [0,0,-1,0],
                [0,-1,0,0],
                [0,0,0,1],
            ],
            dtype=torch.float32,
        )

        if hasattr(entry, "transform_homogeneous") and entry.transform_homogeneous is not None:
            transform = torch.tensor(entry.transform_homogeneous, dtype=torch.float32)
            if transform.numel() == 16:
                transform = transform.view(4, 4)
            if y_up:
                transform = transform @ yup_to_zup
            return transform, scale

        if hasattr(entry, "transform") and entry.transform is not None:
            transform_cfg = entry.transform
            pos = transform_cfg.get("pos", [0.0, 0.0, 0.0])
            rot = transform_cfg.get("rot", [0.0, 0.0, 0.0])
            scale = float(transform_cfg.get("scale", scale))
            from scipy.spatial.transform import Rotation as R
            rot_mat = R.from_euler("ZYX", rot, degrees=True).as_matrix()
            transform = torch.eye(4, dtype=torch.float32)
            transform[:3, :3] = torch.tensor(rot_mat, dtype=torch.float32)
            transform[:3, 3] = torch.tensor(pos, dtype=torch.float32)
            if y_up:
                transform = transform @ yup_to_zup
            return transform, scale

        transform = torch.eye(4, dtype=torch.float32)
        if y_up:
            transform = transform @ yup_to_zup
        return transform, scale

    def _init_from_cfg(self, cfg):
        cfg_3dgs = getattr(cfg, "gsplat", None)
        entries = self._normalize_3dgs_cfg(cfg_3dgs)
        if not entries:
            return

        renderer = StaticGSRenderer(self.device)
        for entry in entries:
            path = entry.get("path", None)
            if path is None:
                continue
            resolved = self._resolve_3dgs_path(path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"3DGS file not found: {resolved}")
            transform, _ = self._parse_3dgs_transform(entry)
            transform_inv = torch.linalg.inv(transform)
            if self._gs_transform_inv is None:
                self._gs_transform_inv = transform_inv
            else:
                if not torch.allclose(self._gs_transform_inv, transform_inv, atol=1e-5, rtol=1e-4):
                    raise ValueError("Multiple 3DGS transforms found; camera-space transform is ambiguous.")
            sh_layout = entry.get("sh_layout", "channel_first")
            renderer.add_from_path(resolved, sh_layout)

        renderer.finalize()
        self._renderer = renderer

    @torch.no_grad()
    def render(self, pose_root: torch.Tensor, K: torch.Tensor, height: int, width: int) -> Optional[torch.Tensor]:
        if self._renderer is None or self._renderer.means is None:
            return None
        if pose_root.dim() == 2:
            pose_root = pose_root.unsqueeze(0)
        if K.dim() == 2:
            K = K.unsqueeze(0)
        pose_root = pose_root.to(self.device)
        K = K.to(self.device)
        pose = pose_root @ self._offset_camera
        if self._gs_transform_inv is not None:
            pose = self._gs_transform_inv.to(pose.device, pose.dtype) @ pose
        return self._renderer.render(pose, K, width, height)

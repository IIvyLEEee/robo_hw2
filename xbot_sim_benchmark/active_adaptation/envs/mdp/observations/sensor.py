import torch
import einops
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils.math import matrix_from_quat, convert_camera_frame_orientation_convention, quat_conjugate, quat_mul

from .base import Observation

import matplotlib.pyplot as plt
from torchvision.io import write_video
from typing import Tuple, Optional
import torch.nn.functional as F
import numpy as np
import cv2
import os

from active_adaptation.assets import ASSET_PATH

class camera(Observation):
    def __init__(
        self,
        env,
        name: str,
        key: str = "rgb",
        debug: bool = False,
        clip_hw: Tuple[int, int, int, int] = None,
        resize_hw: Tuple[int, int] = None,
        use_3dgs: bool = False,
        color_alignment: Optional[str] = None,
        cache_3dgs: bool = False,
        save_rendering: bool = False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.camera: Camera = self.env.scene[name]
        self.name = name
        self.key = key
        self.debug = debug
        self.frame_count = 0
        self.clip_hw = clip_hw
        self.resize_hw = resize_hw
        self.use_3dgs = use_3dgs
        self.color_alignment_path = color_alignment
        self.color_alignment_B = None
        self.cache_3dgs = cache_3dgs
        self._gs_cache = None
        self.save_rendering = bool(save_rendering)
        self._render_debug = None
        self._render_dump_dir = None
        self._render_save_count = 0
        if self.color_alignment_path:
            self._load_color_alignment(self.color_alignment_path)
        if self.debug:
            plt.ion()  # 开启交互模式
            self.frames = []

    def _resolve_color_alignment_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        asset_path = os.path.join(ASSET_PATH, path)
        if os.path.exists(asset_path):
            return asset_path
        return path

    def _load_color_alignment(self, path: str) -> None:
        resolved = self._resolve_color_alignment_path(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Color alignment file not found: {resolved}")
        mat = np.load(resolved)
        if mat.shape != (10, 3):
            raise ValueError(f"Expected 10x3 matrix for color alignment, got {mat.shape}")
        self.color_alignment_B = torch.from_numpy(mat.astype(np.float32))

    def _apply_color_alignment(self, img: torch.Tensor) -> torch.Tensor:
        if self.color_alignment_B is None:
            return img

        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]
        ones = torch.ones_like(r)
        feats = torch.cat([ones, r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], dim=1)
        feats = feats.permute(0, 2, 3, 1)

        B = self.color_alignment_B.to(img.device, dtype=feats.dtype)
        out = torch.matmul(feats, B).permute(0, 3, 1, 2)
        out = out.clamp(0.0, 1.0)

        return out

    def _get_camera_pose_root(self) -> torch.Tensor:
        pos_w, quat_w = self.camera._view.get_world_poses()
        quat_w = convert_camera_frame_orientation_convention(
            quat_w, origin="opengl", target="world"
        )

        root_pos = self.asset.data.root_pos_w
        root_quat = self.asset.data.root_quat_w
        pos_root = pos_w - root_pos
        rot_root = matrix_from_quat(quat_mul(
            quat_conjugate(root_quat),
            quat_w,
        ))

        pose = torch.zeros(pos_w.shape[0], 4, 4, device=pos_w.device, dtype=pos_w.dtype)
        pose[:, :3, :3] = rot_root
        pose[:, :3, 3] = pos_root
        pose[:, 3, 3] = 1.0
        return pose

    def table_luminance_quantile(self, Y: torch.Tensor, table_mask: torch.Tensor, q: float = 0.95) -> torch.Tensor:
        I0_list = []
        for n in range(Y.shape[0]):
            Yn = Y[n][table_mask[n]].flatten()
            I0_list.append(torch.quantile(Yn, q))
        return torch.stack(I0_list, dim=0).view(-1, 1, 1, 1)

    def _apply_table_shadow(self, img, gs_image, table_mask, gamma=2.2):
        # 1) 转线性
        img_lin = img.clamp(0, 1).pow(gamma)
        gs_lin  = gs_image.clamp(0, 1).pow(gamma)

        # 2) 亮度（线性空间）
        w = img_lin.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        Y = (img_lin * w).sum(dim=1, keepdim=True)  # N,1,H,W

        # print(self.table_luminance_quantile(Y, table_mask, 0.9))
        # # 3) 从桌面区域取无阴影基准 I0
        # sum_Y = (Y * table_mask).sum(dim=(2, 3), keepdim=True)
        # cnt   = table_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)  # 防止除0
        # I0    = (sum_Y / cnt).clamp_min(1e-5)  # N,1,1,1
        I0 = 0.3
        strength = 1.0
        # 阴影倍率图（只保留变暗 <=1）
        s = (Y / I0).clamp(0.0, 1.0)  # N,1,H,W
        # s = s.pow(1.5)

        # 5) 只在桌面应用阴影倍率
        gs_lin_shadow = gs_lin * torch.where(table_mask, s, torch.ones_like(s))

        # 6) 回到 gamma 空间
        out = gs_lin_shadow.clamp(0, 1).pow(1.0/gamma)
        return out

    def _ensure_render_dump_dir(self) -> str:
        if self._render_dump_dir is None:
            root = os.getcwd()
            dump_dir = os.path.join(root, "render_debug")
            os.makedirs(dump_dir, exist_ok=True)
            self._render_dump_dir = dump_dir
        return self._render_dump_dir

    def _prepare_render_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return None
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            return None
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        return self.process_image(tensor)

    def _save_render_tensor(self, label: str, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            return
        dump_dir = self._ensure_render_dump_dir()
        for idx in range(tensor.shape[0]):
            img = tensor[idx]
            if img.dtype == torch.bool:
                img = img.float()
            if torch.is_floating_point(img):
                img = (img * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                img = img.to(torch.uint8)
            if img.shape[0] in (1, 3):
                img_np = img.permute(1, 2, 0).detach().cpu().numpy()
                if img_np.shape[2] == 3:
                    img_np = img_np[..., ::-1]
            else:
                img_np = img.detach().cpu().numpy()
                if img_np.ndim == 3:
                    img_np = img_np[0]
            filename = f"{self.name}_{label}_{self.frame_count:06d}_env{idx}.png"
            cv2.imwrite(os.path.join(dump_dir, filename), img_np)

    def _flush_render_debug(self, img_isaac: torch.Tensor, img_blend: torch.Tensor) -> None:
        if not self.save_rendering:
            return
        if self._render_save_count >= 5:
            self._render_debug = None
            return
        img_isaac = self._prepare_render_tensor(img_isaac)
        self._save_render_tensor("img_isaac", img_isaac)
        if self._render_debug is None:
            if img_blend is not None:
                self._save_render_tensor("img_blend", self._prepare_render_tensor(img_blend))
            self._render_save_count += 1
            return
        self._save_render_tensor("gs_render", self._prepare_render_tensor(self._render_debug.get("gs_render")))
        self._save_render_tensor("gs_aligned", self._prepare_render_tensor(self._render_debug.get("gs_aligned")))
        self._save_render_tensor("background_mask", self._prepare_render_tensor(self._render_debug.get("background_mask")))
        self._save_render_tensor("table_mask", self._prepare_render_tensor(self._render_debug.get("table_mask")))
        self._save_render_tensor("gs_shadow", self._prepare_render_tensor(self._render_debug.get("gs_shadow")))
        self._save_render_tensor("img_blend", self._prepare_render_tensor(img_blend))
        self._render_debug = None
        self._render_save_count += 1

    def _blend_3dgs(self, img: torch.Tensor) -> torch.Tensor:
        if not self.use_3dgs:
            return img
        if not hasattr(self.env, "render_3dgs_background"):
            return img
        H, W = self.camera.data.image_shape
        if self.cache_3dgs and self._gs_cache is not None:
            gs_image = self._gs_cache
            if gs_image.shape[-2:] != (H, W) or gs_image.shape[0] != img.shape[0]:
                print("3DGS cache shape mismatch, recomputing 3DGS background.")
                self._gs_cache = None
                gs_image = None
        else:
            gs_image = None

        if gs_image is None:
            pose_root = self._get_camera_pose_root()
            K = self.camera.data.intrinsic_matrices
            gs_image = self.env.render_3dgs_background(pose_root, K, H, W)
            if gs_image is None:
                return img
            gs_image = einops.rearrange(gs_image, "n h w c -> n c h w")
            if gs_image.dtype == torch.uint8:
                gs_image = gs_image.float() / 255.0
            else:
                print(f"[Warning] 3DGS output dtype is {gs_image.dtype}, expected uint8.")
            if self.cache_3dgs:
                self._gs_cache = gs_image
        gs_render = gs_image

        types = self.camera.data.output.get("semantic_segmentation", None)
        if types is None:
            return img
        background_mask = (types == 0)
        background_mask = einops.rearrange(background_mask, "n h w c -> n c h w")

        id_to_class = self.camera.data.info['semantic_segmentation']["idToLabels"]
        table_type = None
        for id_, segtag in id_to_class.items():
            if segtag.get("class", "") == "table":
                table_type = int(id_)
                break
        if table_type is not None:
            table_mask = (types == table_type)
            table_mask = einops.rearrange(table_mask, "n h w c -> n c h w")
            background_mask = background_mask | table_mask
        else:
            table_mask = None
        
        # table_mask = None

        gs_image = self._apply_color_alignment(gs_image)
        gs_shadow = gs_image
        if table_mask is not None:
            gs_shadow = self._apply_table_shadow(img, gs_image, table_mask)

        if self.save_rendering:
            self._render_debug = {
                "gs_render": gs_render,
                "gs_aligned": gs_image,
                "background_mask": ~background_mask,
                "table_mask": table_mask,
                "gs_shadow": gs_shadow,
            }

        background_mask = background_mask.to(img.device)
        return torch.where(background_mask, gs_shadow, img)
        # return gs_image

    def process_image(self, img: torch.Tensor) -> torch.Tensor:
        if self.clip_hw is not None:
            img = img[:, :, self.clip_hw[0]:self.clip_hw[1], self.clip_hw[2]:self.clip_hw[3]]
        if self.resize_hw is not None:
            # interpolate the image to the new size
            img = F.interpolate(img, size=(self.resize_hw[0], self.resize_hw[1]), mode="bilinear", align_corners=False)
        return img

    def compute(self):
        img_isaac = self.camera.data.output[self.key]
        img_isaac = einops.rearrange(img_isaac, "n h w c -> n c h w")
        if img_isaac.dtype == torch.uint8:
            img_isaac = img_isaac.float() / 255.0
        else:
            print(f"[Warning] Camera output dtype is {img_isaac.dtype}, expected uint8.")
        img_blend = self._blend_3dgs(img_isaac)
        img_final = self.process_image(img_blend)
        if self.save_rendering:
            self._flush_render_debug(img_isaac, img_blend)
        img_final = img_final.clamp(0.0, 1.0)
        img_final_u8 = (img_final * 255.0).round().clamp(0, 255).to(torch.uint8)
        self.frame_count += 1
        if self.debug:
            img_disp = img_final_u8[0].clone()
            self.frames.append(img_disp)

            plt.clf()
            plt.imshow(img_disp.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.pause(0.001)

        # Return channel-first image per env: [N, C, H, W].
        # Flattening (if desired) is handled at the ObsGroup level via `no_flatten`.
        return img_final_u8

class padding_camera(camera):
    def __init__(
        self,
        env,
        name: str,
        key: str = "rgb",
        debug: bool = False,
        resize_hw: Tuple[int, int] = None,
        padding_hw: Tuple[int, int] = None,
        padding_offset: Tuple[int, int] = None,
        final_resize_hw: Tuple[int, int] = None,
        use_3dgs: bool = False,
        color_alignment: Optional[str] = None,
        cache_3dgs: bool = False,
    ):
        super().__init__(
            env,
            name,
            key,
            debug,
            use_3dgs=use_3dgs,
            color_alignment=color_alignment,
            cache_3dgs=cache_3dgs,
        )
        self.resize_hw = resize_hw
        self.padding_hw = padding_hw
        self.final_resize_hw = final_resize_hw
        self.padding_offset = padding_offset

    def process_image(self, img: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        # resize the image to the resize_hw
        img = F.interpolate(img, size=(self.resize_hw[0], self.resize_hw[1]), mode="bilinear", align_corners=False)
        # pad the image to the padding_hw
        img_pad = torch.zeros(img.shape[0], img.shape[1], self.padding_hw[0], self.padding_hw[1], device=img.device, dtype=img.dtype)
        img_pad[:, :, self.padding_offset[0]:self.padding_offset[0]+img.shape[2], self.padding_offset[1]:self.padding_offset[1]+img.shape[3]] = img
        # resize the image to the final_resize_hw
        img = F.interpolate(img_pad, size=(self.final_resize_hw[0], self.final_resize_hw[1]), mode="bilinear", align_corners=False)
        return img

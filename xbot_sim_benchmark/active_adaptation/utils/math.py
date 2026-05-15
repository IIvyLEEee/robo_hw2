import torch
import torch.distributions as D

# @torch.compile
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


# @torch.compile
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis


class MultiUniform(D.Distribution):
    """
    A distribution over the union of multiple disjoint intervals.
    """
    def __init__(self, ranges: torch.Tensor):
        batch_shape = ranges.shape[:-2]
        if not ranges[..., 0].le(ranges[..., 1]).all():
            raise ValueError("Ranges must be non-empty and ordered.")
        super().__init__(batch_shape, validate_args=False)
        self.ranges = ranges
        self.ranges_len = ranges.diff(dim=-1).squeeze(1)
        self.total_len = self.ranges_len.sum(-1)
        self.starts = torch.zeros_like(ranges[..., 0])
        self.starts[..., 1:] = self.ranges_len.cumsum(-1)[..., :-1]

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        uniform = torch.rand(shape, device=self.ranges.device) * self.total_len
        i = torch.searchsorted(self.starts, uniform) - 1
        return self.ranges[i, 0] + uniform - self.starts[i]


import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

def quaternion_to_rot6d(quat):
    """
    quat: shape (..., 4), wxyz
    返回 shape (..., 6)
    """
    quat = torch.as_tensor(quat, dtype=torch.float32)
    assert quat.shape[-1] == 4 and quat.ndim >= 1
    raw_shape = quat.shape[:-1]
    rotmat = matrix_from_quat(quat)  # (..., 3, 3)
    rot6d = rotmat[..., :2, :].reshape(*raw_shape, 6)
    return rot6d

def rot6d_to_quaternion(rot6d):
    """
    rot6d: shape (..., 6)
    返回 shape (..., 4), wxyz
    """
    rot6d = torch.as_tensor(rot6d, dtype=torch.float32)
    assert rot6d.shape[-1] == 6 and rot6d.ndim >= 1
    raw_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)

    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]
    # Gram-Schmidt 正交化
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rotmat = torch.stack([b1, b2, b3], dim=-2)  # (N, 3, 3)
    rotmat = rotmat.reshape(*raw_shape, 3, 3)
    quat = quat_from_matrix(rotmat)
    return quat.reshape(*raw_shape, 4)

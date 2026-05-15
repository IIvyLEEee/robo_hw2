from typing import Dict
import copy
import torch
import numpy as np
import zarr
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class M7Star1Dataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            zarr_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            episode_start=None,
            episode_end=None,
        ):
        super().__init__()
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        zarr_root = zarr.open(zarr_path, mode='r')
        data_group = zarr_root['data']

        rgb_key_map = {
            obs_key: self._resolve_data_key(data_group, obs_key, is_rgb=True)
            for obs_key in rgb_keys
        }
        lowdim_key_map = {
            obs_key: self._resolve_data_key(data_group, obs_key, is_rgb=False)
            for obs_key in lowdim_keys
        }

        replay_keys = list(dict.fromkeys(
            list(rgb_key_map.values()) + list(lowdim_key_map.values()) + ['action']
        ))
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=replay_keys)

        for obs_key, data_key in rgb_key_map.items():
            image_shape = tuple(obs_shape_meta[obs_key]['shape'])
            c, h, w = image_shape
            assert tuple(replay_buffer[data_key].shape[1:]) == (h, w, c)

        for obs_key, data_key in lowdim_key_map.items():
            assert tuple(replay_buffer[data_key].shape[1:]) == tuple(
                obs_shape_meta[obs_key]['shape'])
        assert tuple(replay_buffer['action'].shape[1:]) == tuple(
            shape_meta['action']['shape'])

        key_first_k = dict()
        if n_obs_steps is not None:
            for data_key in rgb_key_map.values():
                key_first_k[data_key] = n_obs_steps
            for data_key in lowdim_key_map.values():
                key_first_k[data_key] = n_obs_steps

        episode_mask = self._make_episode_mask(
            n_episodes=replay_buffer.n_episodes,
            episode_start=episode_start,
            episode_end=episode_end,
        )
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        val_mask = val_mask & episode_mask
        train_mask = episode_mask & ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.rgb_key_map = rgb_key_map
        self.lowdim_key_map = lowdim_key_map
        self.key_first_k = key_first_k
        self.episode_mask = episode_mask
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @staticmethod
    def _resolve_data_key(data_group, obs_key: str, is_rgb: bool) -> str:
        if obs_key in data_group:
            return obs_key
        if is_rgb:
            image_key = f'img_{obs_key}'
            if image_key in data_group:
                return image_key
        elif obs_key == 'state' and 'state' in data_group:
            return 'state'

        available = list(data_group.keys())
        raise KeyError(
            f"Could not map obs key '{obs_key}' to a zarr data key. "
            f"Available zarr data keys: {available}"
        )

    @staticmethod
    def _make_episode_mask(n_episodes: int, episode_start=None, episode_end=None) -> np.ndarray:
        start = 0 if episode_start is None else int(episode_start)
        end = n_episodes - 1 if episode_end is None else int(episode_end)
        if start < 0:
            raise ValueError(f"episode_start must be >= 0, got {start}")
        if end < start:
            raise ValueError(f"episode_end must be >= episode_start, got {end} < {start}")
        if end >= n_episodes:
            raise ValueError(f"episode_end must be < n_episodes ({n_episodes}), got {end}")

        mask = np.zeros(n_episodes, dtype=bool)
        mask[start:end + 1] = True
        return mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=self.key_first_k
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        for obs_key, data_key in self.lowdim_key_map.items():
            normalizer[obs_key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[data_key])
        for obs_key in self.rgb_keys:
            normalizer[obs_key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # only return observed timesteps; future observations are not used.
        T_slice = slice(self.n_obs_steps)
        obs = {}
        for obs_key, data_key in self.rgb_key_map.items():
            image = np.moveaxis(data[data_key][T_slice], -1, 1
                ).astype(np.float32) / 255.0
            obs[obs_key] = torch.from_numpy(image)

        for obs_key, data_key in self.lowdim_key_map.items():
            value = data[data_key][T_slice].astype(np.float32)
            obs[obs_key] = torch.from_numpy(value)

        torch_data = {
            'obs': obs,
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

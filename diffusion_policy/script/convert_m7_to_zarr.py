import os
import json
import argparse
import numpy as np
import pandas as pd
import zarr
import cv2
from huggingface_hub import snapshot_download
from tqdm import tqdm


class M7ZarrConverter:

    def __init__(
        self,
        data_dir,
        out_path,
        img_size=96,
        camera_names=None,
        max_episodes=None,     # set by yourself
        debug=False
    ):
        self.data_dir = data_dir
        self.out_path = out_path
        self.img_size = img_size
        self.camera_names = tuple(camera_names or ("cam_high", "cam_left", "cam_right"))
        self.max_episodes = max_episodes
        self.debug = debug
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera.")

    @property
    def image_keys(self):
        return [f"img_{name}" for name in self.camera_names]

    def video_path(self, camera_name, ep_id):
        return os.path.join(
            self.data_dir,
            f"videos/chunk-000/observation.images.{camera_name}/episode_{ep_id:06d}.mp4"
        )

    def decode_video(self, video_path, length):
        cap = cv2.VideoCapture(video_path)
        frames = np.empty(
            (length, self.img_size, self.img_size, 3),
            dtype=np.uint8
        )

        for t in range(length):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"frame {t} decode failed")

            frame = frame[:, :, ::-1]
            frames[t] = cv2.resize(frame, (self.img_size, self.img_size))

        cap.release()
        return frames

    def _make_image_dataset(self, data_group, key, total_frames):
        return data_group.create_dataset(
            key,
            shape=(total_frames, self.img_size, self.img_size, 3),
            chunks=(256, self.img_size, self.img_size, 3),
            dtype=np.uint8
        )

    def run(self):

        meta = os.path.join(self.data_dir, "meta")

        episodes = [json.loads(l) for l in open(os.path.join(meta, "episodes.jsonl"))]

        # episode number check
        if self.max_episodes is not None:
            episodes = episodes[:self.max_episodes]

        print(f"[INFO] data_dir: {self.data_dir}")
        print(f"[INFO] Converting {len(episodes)} episodes")
        print(f"[INFO] cameras: {', '.join(self.camera_names)}")

        # calculate total frames
        total_frames = sum(ep["length"] for ep in episodes)

        # determine state and action dimensions
        first_ep_id = episodes[0]["episode_index"]
        first_parquet_path = os.path.join(
            self.data_dir,
            f"data/chunk-000/episode_{first_ep_id:06d}.parquet"
        )
        first_df = pd.read_parquet(first_parquet_path, engine="pyarrow")
        state_dim = len(np.asarray(first_df.iloc[0]["observation.state"]))
        action_dim = len(np.asarray(first_df.iloc[0]["action"]))

        resume = os.path.exists(self.out_path)
        root = zarr.open(self.out_path, mode="a" if resume else "w")
        data_group = root.require_group("data")
        meta_group = root.require_group("meta")

        if (
            any(key not in data_group for key in self.image_keys)
            or "state" not in data_group
            or "action" not in data_group
            or "episode_ends" not in meta_group
        ):
            if resume:
                print("[INFO] Existing zarr has no resume metadata, rebuilding it")
                root = zarr.open(self.out_path, mode="w")
                data_group = root.require_group("data")
                meta_group = root.require_group("meta")

            obs_imgs = {
                key: self._make_image_dataset(data_group, key, total_frames)
                for key in self.image_keys
            }

            obs_state = data_group.create_dataset(
                "state",
                shape=(total_frames, state_dim),
                chunks=(256, state_dim),
                dtype=np.float32
            )

            action = data_group.create_dataset(
                "action",
                shape=(total_frames, action_dim),
                chunks=(256, action_dim),
                dtype=np.float32
            )

            episode_ends = meta_group.create_dataset(
                "episode_ends",
                shape=(0,),
                chunks=(min(256, len(episodes)),),
                dtype=np.int64
            )
        else:
            obs_imgs = {key: data_group[key] for key in self.image_keys}
            obs_state = data_group["state"]
            action = data_group["action"]
            episode_ends = meta_group["episode_ends"]

            expected_shapes = {
                **{
                    key: (total_frames, self.img_size, self.img_size, 3)
                    for key in self.image_keys
                },
                "state": (total_frames, state_dim),
                "action": (total_frames, action_dim)
            }
            actual_shapes = {
                **{key: obs_imgs[key].shape for key in self.image_keys},
                "state": obs_state.shape,
                "action": action.shape
            }
            if actual_shapes != expected_shapes:
                raise RuntimeError(
                    f"Existing zarr shape mismatch: {actual_shapes} != {expected_shapes}"
                )

        completed_episodes = len(episode_ends)
        cursor = int(episode_ends[-1]) if completed_episodes > 0 else 0

        if completed_episodes > 0:
            print(f"[INFO] Resuming from episode {completed_episodes}, frames={cursor}")

        for ep_idx, ep in enumerate(tqdm(episodes[completed_episodes:]), completed_episodes):

            ep_id = ep["episode_index"]
            L = ep["length"]
            start = cursor
            end = cursor + L

            parquet_path = os.path.join(
                self.data_dir,
                f"data/chunk-000/episode_{ep_id:06d}.parquet"
            )

            # ⭐ read by pyarrow-safe
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            df = df.iloc[:L]

            obs_state[start:end] = np.asarray(
                df["observation.state"].tolist(),
                dtype=np.float32
            )
            action[start:end] = np.asarray(
                df["action"].tolist(),
                dtype=np.float32
            )

            # debug mode cancel video decode
            if self.debug:
                blank = np.zeros(
                    (L, self.img_size, self.img_size, 3),
                    dtype=np.uint8
                )
                for obs_img in obs_imgs.values():
                    obs_img[start:end] = blank
            else:
                for camera_name, image_key in zip(self.camera_names, self.image_keys):
                    frames = self.decode_video(self.video_path(camera_name, ep_id), L)
                    obs_imgs[image_key][start:end] = frames

            cursor = end
            episode_ends.resize(ep_idx + 1)
            episode_ends[ep_idx] = cursor

            print(f"[DEBUG] finished episode {ep_id}, frames={cursor}")

        print("\n[OK] DONE zarr saved to:", self.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", default="data/roboterax_M7_pickplace_example_initial")
    parser.add_argument("--data-id", default="2031605")
    parser.add_argument("--out-path", default="data/m7.zarr")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--camera-names", nargs="+", default=["cam_high", "cam_left", "cam_right"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    download_dir = args.download_dir
    data_dir = os.path.join(download_dir, args.data_id)

    if not os.path.exists(os.path.join(data_dir, "meta", "episodes.jsonl")):
        snapshot_download(
            repo_id="roboterax/M7_pickplace_example",
            repo_type="dataset",
            allow_patterns=f"{args.data_id}/**",
            local_dir=download_dir,
            local_dir_use_symlinks=False
        )

    converter = M7ZarrConverter(
        data_dir=data_dir,
        out_path=args.out_path,
        img_size=args.img_size,
        camera_names=args.camera_names,
        max_episodes=args.max_episodes,
        debug=args.debug
    )

    converter.run()

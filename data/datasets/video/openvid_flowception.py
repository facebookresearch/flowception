import os, numpy as np, tqdm, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from engine.data_classes import Datapoint
from decord import VideoReader, cpu
from decord import bridge as decord_bridge

# Return torch tensors from decord (so we can .to(torch.float32))
decord_bridge.set_bridge("torch")


class OpenVid1MFlowception(Dataset):
    """OpenVid-1M dataset with Flowception frame alignment (T = 1 + n*ld)."""

    def __init__(
        self,
        annotations_dir: str,
        width: int,
        height: int,
        num_frames: int = 72,
        min_motion_score: float = -1.0,
        sampling_fps: float = 24.0,
        native_fps: float = 30.0,
        num_start_frames: int = 2,
        latent_downsample: int = 8,
        max_retries: int = 20,
    ):
        annots_path = Path(annotations_dir) / "data" / "train" / "OpenVid-1M.csv"
        self.annots_csv = pd.read_csv(annots_path)
        self.video_root = Path(annotations_dir) / "video"

        if min_motion_score > 0:
            self.annots_csv = self.annots_csv[self.annots_csv["motion score"] > min_motion_score]
            print(
                f"OpenVid1MFlowception: filtered to {len(self.annots_csv)} videos "
                f"(motion score > {min_motion_score})"
            )

        self.width = int(width)
        self.height = int(height)

        self.sampling_fps = float(sampling_fps)
        self.native_fps = float(native_fps)
        stride = int(round(self.native_fps / self.sampling_fps))
        self.frame_stride = max(1, stride)

        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)
        self.max_retries = int(max_retries)

        # Align T to VAE rule: T = 1 + n*ld (ceil)
        T_req = int(num_frames)
        ld = self.latent_downsample
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 for (first + groups-of-ld) rule")
        rem = (T_req - 1) % ld
        if rem != 0:
            T_aligned = T_req + (ld - rem)
            print(f"[info] aligning num_frames from {T_req} -> {T_aligned} (1 + n*{ld})")
            T_req = T_aligned
        self.num_frames = T_req

    def __len__(self):
        return len(self.annots_csv)

    def _fetch_one(self, idx):
        item = self.annots_csv.iloc[idx]
        vid_path = str(self.video_root / item["video"])
        caption = str(item["caption"])

        if not os.path.isfile(vid_path):
            raise FileNotFoundError(vid_path)

        # Use the per-video fps if available, fall back to native_fps
        video_fps = float(item["fps"]) if "fps" in item and not pd.isna(item["fps"]) else self.native_fps
        stride = max(1, int(round(video_fps / self.sampling_fps)))

        reader = VideoReader(
            vid_path,
            num_threads=-1,
            ctx=cpu(0),
            width=self.width,
            height=self.height,
        )
        total = len(reader)
        if total < 1:
            raise ValueError("Empty video")

        ld = self.latent_downsample
        k = self.num_start_frames
        s = stride
        T = self.num_frames

        max_valid = 1 + (total - 1) // s
        min_latents = k + 2
        min_valid_needed = 1 + (min_latents - 1) * ld

        if max_valid < min_valid_needed:
            raise RuntimeError("Video too short for min latent length")

        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed

        max_start = total - (L - 1) * s
        if max_start <= 0:
            start = 0
        else:
            start = int(np.random.randint(0, max_start))

        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
        frames_valid = reader.get_batch(idx_valid).to(torch.float32)
        frames_valid = frames_valid / 127.5 - 1.0

        # Pad to fixed T for batching
        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        frame_mask = torch.zeros(T, dtype=torch.bool)
        frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld
        video_length = L

        img_tensor = frames.permute(3, 0, 1, 2).contiguous()
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()

        assert img_tensor.shape[1] == T
        assert frame_mask.numel() == T
        assert frame_indices.shape[0] == T
        assert (L - 1) % ld == 0

        crop_coords = torch.zeros(8)

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": caption,
                "caption_idx": torch.tensor(0),
                "crop_coords": crop_coords,
                "anchor_frame": anchor_tensor,
                "frame_mask": frame_mask,
                "video_length": torch.tensor(video_length),
                "latent_length": torch.tensor(latent_length),
                "stride": torch.tensor(s),
                "frame_indices": torch.from_numpy(frame_indices),
            },
        )

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one(idx)
                if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
                    return dp
            except Exception:
                pass
            idx = np.random.randint(len(self.annots_csv))
        return self._fetch_one(idx)

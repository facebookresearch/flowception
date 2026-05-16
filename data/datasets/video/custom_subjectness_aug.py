import os, joblib, numpy as np, tqdm, torch
from torch.utils.data import Dataset
from typing import Optional
from engine.data_classes import Datapoint
from decord import VideoReader, cpu
from decord import bridge as decord_bridge

# Return torch tensors from decord (so we can .to(torch.float32))
decord_bridge.set_bridge("torch")


def _passes_subjectness_filters(
    entry: dict,
    min_subjectness: Optional[float],
    min_size_ratio: Optional[float],
    max_size_ratio: Optional[float],
) -> bool:
    if min_subjectness is None and min_size_ratio is None and max_size_ratio is None:
        return True  # no subjectness-based filtering requested

    sd = entry.get("subjectness", None)  # your out_key name
    if not isinstance(sd, dict):
        return False  # need the dict present to evaluate thresholds

    # pull values, tolerate missing -> fail that constraint only
    subj = sd.get("subjectness", None)
    sr = sd.get("size_ratio", None)

    if (min_subjectness is not None) and (
        not isinstance(subj, (int, float)) or subj < float(min_subjectness)
    ):
        return False
    if (min_size_ratio is not None) and (not isinstance(sr, (int, float)) or sr < float(min_size_ratio)):
        return False
    if (max_size_ratio is not None) and (not isinstance(sr, (int, float)) or sr > float(max_size_ratio)):
        return False
    return True


class CustomSubjectnessFlowceptionAug(Dataset):
    def __init__(
        self,
        annotations_dir: str,
        width: int,
        height: int,
        num_frames: int = 72,  # requested output length (we will ceil-align to 1 + n*ld)
        min_motion_score: float = 3.0,
        sampling_fps: float = 24.0,
        native_fps: float = 24.0,
        pick_files: int | None = 64,
        # --- VAE/Flowception knobs ---
        num_start_frames: int = 2,  # k
        latent_downsample: int = 8,  # ld
        max_retries: int = 20,
        # --- entropy filter knobs ---
        min_entropy: float | None = None,  # e.g. 5.0 (nats) or 6.5 (bits) – None disables filtering
        entropy_base: float | str = "e",  # 'e' (nats) or 2 (bits)
        return_entropy: bool = False,  # if True, include measured entropy in condition
        # --- NEW: subjectness filters ---
        min_subjectness: float | None = None,
        min_size_ratio: float | None = None,
        max_size_ratio: float | None = None,
        # If your subjectness dict is stored under a different key, set this:
        subjectness_key: str = "subjectness",
        # Optional: override motion key name; kept default as "motion_score"
        motion_key: str = "motion_score",
        decode_scale: float = 1.25,
    ):
        entry_paths = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
        if pick_files is not None and pick_files < len(entry_paths):
            entry_paths = list(np.random.choice(entry_paths, pick_files, replace=False))

        all_entries = []
        kept_motion = kept_subj = 0

        # for jp in tqdm.tqdm(entry_paths, desc="Load+filter shards"):
        for jp in entry_paths:
            data = joblib.load(os.path.join(annotations_dir, jp))

            # rename subjectness key if needed to normalize downstream access
            if subjectness_key != "subjectness":
                for it in data:
                    if "subjectness" not in it and subjectness_key in it:
                        it["subjectness"] = it[subjectness_key]

            # motion filter first (same behavior as before)
            motion_filtered = [
                it
                for it in data
                if isinstance(it.get(motion_key, None), (int, float))
                and float(it[motion_key]) >= float(min_motion_score)
            ]
            kept_motion += len(motion_filtered)

            # NEW: subjectness + size_ratio filters
            subj_filtered = [
                it
                for it in motion_filtered
                if _passes_subjectness_filters(
                    it,
                    min_subjectness=min_subjectness if min_subjectness > 0 else None,
                    min_size_ratio=min_size_ratio if min_size_ratio > 0 else None,
                    max_size_ratio=max_size_ratio if max_size_ratio > 0 else None,
                )
            ]
            kept_subj += len(subj_filtered)
            all_entries.extend(subj_filtered)

        self.all_entries = all_entries
        print(f"Remaining items after motion ≥ {min_motion_score}: {kept_motion}")
        if any(v is not None for v in (min_subjectness, min_size_ratio, max_size_ratio)):
            print(
                f"Remaining items after subjectness filters "
                f"(min_subj={min_subjectness}, size∈[{min_size_ratio},{max_size_ratio}]): {kept_subj}"
            )

        self.width = int(width)
        self.height = int(height)

        self.decode_scale = float(decode_scale)
        self.decode_width = int(round(self.width * self.decode_scale))
        self.decode_height = int(round(self.height * self.decode_scale))

        self.sampling_fps = float(sampling_fps)
        self.native_fps = float(native_fps)
        stride = int(round(self.native_fps / self.sampling_fps))
        self.frame_stride = max(1, stride)

        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)
        self.max_retries = int(max_retries)

        # --- entropy settings
        self.min_entropy = float(min_entropy) if min_entropy is not None else None
        self.entropy_base = entropy_base
        self.return_entropy = bool(return_entropy)

        # ----- Align T to VAE rule: T = 1 + n*ld (ceil) -----
        T_req = int(num_frames)
        ld = self.latent_downsample
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 for (first + groups-of-ld) rule")
        rem = (T_req - 1) % ld
        if rem != 0:
            T_aligned = T_req + (ld - rem)  # ceil to 1 + n*ld
            print(f"[info] aligning num_frames from {T_req} -> {T_aligned} (1 + n*{ld})")
            T_req = T_aligned
        self.num_frames = T_req  # final T

    def __len__(self):
        # streaming-style: iterate endlessly; adjust if you want epoch-sized length
        return max(len(self.all_entries), 1_000_000)

    # --- pure-torch grayscale entropy on a single uint8 frame [H,W,3]
    @staticmethod
    def _frame_entropy_uint8(frame_uint8: torch.Tensor, base: float | str = "e") -> float:
        f = frame_uint8.to(torch.float32)
        gray = (
            (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).round().clamp(0, 255).to(torch.uint8)
        )
        flat = gray.view(-1)
        counts = torch.bincount(flat, minlength=256).to(torch.float32)
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts[counts > 0] / total
        H = -(p * torch.log(p)).sum()  # nats
        if base == "e":
            return float(H.item())
        else:
            base = float(base)
            return float((H / torch.log(torch.tensor(base))).item())

    def _fetch_one(self):
        j = np.random.randint(0, len(self.all_entries))
        item = self.all_entries[j]
        vid_path = item.get("filepath") or item.get("path")
        if not isinstance(vid_path, str) or not os.path.isfile(vid_path):
            raise FileNotFoundError(str(vid_path))

        reader = VideoReader(
            vid_path, num_threads=-1, ctx=cpu(0), width=self.decode_width, height=self.decode_height
        )

        total = len(reader)
        if total < 1:
            raise ValueError("Empty video")

        ld = self.latent_downsample
        k = self.num_start_frames
        s = self.frame_stride
        T = self.num_frames

        # Frames available at this stride from the whole video
        max_valid = 1 + (total - 1) // s

        # Minimum real RGB frames you need (first + groups of ld)
        min_latents = k + 2
        min_valid_needed = 1 + (min_latents - 1) * ld

        if max_valid < min_valid_needed:
            raise RuntimeError("Video too short for min latent length")

        # Choose target frames (≤ T and ≤ max_valid), then ALIGN: L = 1 + q*ld
        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed

        # Choose a random start that fits exactly L frames at stride s
        max_start = total - (L - 1) * s
        start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

        # quick entropy check on anchor frame
        first_frame = reader.get_batch([start])[0]  # [H,W,3], uint8
        anchor_entropy = self._frame_entropy_uint8(first_frame, base=self.entropy_base)
        if self.min_entropy is not None and anchor_entropy < self.min_entropy:
            raise RuntimeError(f"Low entropy anchor ({anchor_entropy:.3f} < {self.min_entropy})")

        # materialize indices & frames
        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
        frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
        frames_valid = frames_valid / 127.5 - 1.0

        # pad to fixed T
        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        frame_mask = torch.zeros(T, dtype=torch.bool)
        frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld
        video_length = L

        img_tensor = frames.permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        # anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()   # [3,1,H,W]
        anchor_tensor = img_tensor[:, :1].contiguous()  # first frame after pad, same spatial res

        assert img_tensor.shape[1] == T
        assert frame_mask.numel() == T
        assert frame_indices.shape[0] == T
        assert (L - 1) % ld == 0

        crop_coords = torch.zeros(8)

        cond = {
            "class_id": item.get("description", "") or "",
            "caption_idx": torch.tensor(0),
            "crop_coords": crop_coords,
            "anchor_frame": anchor_tensor,
            "frame_mask": frame_mask,
            "video_length": torch.tensor(video_length),
            "latent_length": torch.tensor(latent_length),
            "stride": torch.tensor(s),
            "frame_indices": torch.from_numpy(frame_indices),
        }
        if self.return_entropy:
            cond["entropy"] = torch.tensor(anchor_entropy, dtype=torch.float32)

        return Datapoint(pixel_values=img_tensor, condition=cond)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one()
                if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
                    return dp
            except Exception:
                continue
        return self._fetch_one()

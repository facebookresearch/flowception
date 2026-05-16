import os, math, json, random
import joblib
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from engine.data_classes import Datapoint
from torch.utils.data import get_worker_info

from decord import bridge as decord_bridge

decord_bridge.set_bridge("torch")  # call once at import time (e.g., top of file)


class YouCook2Flowception(Dataset):
    """
    Each item in `annotations` is a dict like:
      {
        "filename": "K6Uk5vNi1_Q.mp4",
        "caption": [
            {"start": 0.0, "end": 2.87, "caption": "..."},
            {"start": 2.87, "end": 5.14, "caption": "..."},
            ...
        ]
      }

    One __getitem__():
      - picks a random video entry
      - picks a random segment from its 'caption' list
      - samples frames ONLY from [start, end) (seconds), using integer stride
      - pads with the last valid frame to reach `num_frames`
      - returns [-1,1] pixel_values [C,T,H,W] and Flowception-friendly condition:
        anchor_frame, frame_mask, frame_indices, stride, video_length (valid_len),
        latent_length (ceil(valid_len / latent_downsample)), class_id=segment text

    Notes
    -----
    * `native_fps` is the assumed source FPS; `sampling_fps` → integer stride.
    * If a segment is too short, we fall back to another random segment/video
      up to `max_retries` times, then do a last-ditch attempt.
    """

    def __init__(
        self,
        annotations,  # list[dict] OR path to .pt/.pkl/.joblib/.json
        vid_root: str,
        width: int,
        height: int,
        num_frames: int = 72,
        sampling_fps: float = 24.0,  # the ONLY fps you set
        native_fps: float = 24.0,  # assumed fps of encoded videos
        pick_videos: int | None = None,
        # Flowception/training knobs
        num_start_frames: int = 2,
        latent_downsample: int = 8,
        max_retries: int = 20,
        # ):
        shard_style: str = "contiguous",
    ):  # "interleaved"):  # or "contiguous"
        # Load annotations if a path was provided
        if isinstance(annotations, str):
            if annotations.endswith((".pt", ".pkl", ".joblib")):
                annotations = joblib.load(annotations)
            elif annotations.endswith(".json"):
                with open(annotations, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            else:
                raise ValueError(f"Unsupported annotations file: {annotations}")

        # Optional sub-sample of videos
        if pick_videos is not None and len(annotations) > pick_videos:
            annotations = list(np.random.choice(annotations, pick_videos, replace=False))

        # # Keep only entries with at least one well-formed segment
        # cleaned = []
        # for it in annotations:
        #     segs = it.get("caption", []) or []
        #     segs = [s for s in segs
        #             if isinstance(s.get("start", None), (int, float))
        #             and isinstance(s.get("end", None), (int, float))
        #             and s.get("end", 0) > s.get("start", 0)]
        #     if not segs:
        #         continue
        #     cleaned.append({"filename": it["filename"], "segments": segs})
        # self.entries = cleaned
        self.vid_root = vid_root

        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)

        self.sampling_fps = float(sampling_fps)
        self.native_fps = float(native_fps)
        stride = int(round(self.native_fps / self.sampling_fps))
        self.frame_stride = max(1, stride)

        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)

        # --- enforce T = 1 + n*ld (first frame + groups of ld) ---
        # ld = int(self.latent_downsample)
        # if self.num_frames <= 1:
        #     raise ValueError("num_frames must be >= 2 for first+groups-of-ld rule")
        # if (self.num_frames - 1) % ld != 0:
        #     T_aligned = 1 + ((self.num_frames - 1) // ld) * ld
        #     print(f"[info] aligning num_frames from {self.num_frames} -> {T_aligned} (1 + n*{ld})")
        #     self.num_frames = T_aligned

        ld = int(self.latent_downsample)
        if self.num_frames <= 1:
            raise ValueError("num_frames must be >= 2 for first+groups-of-ld rule")

        # CEIL to keep 73 when user passes 72
        rem = (self.num_frames - 1) % ld
        if rem != 0:
            T_aligned = self.num_frames + (ld - rem)  # == 1 + ceil((T-1)/ld)*ld
            print(f"[info] aligning num_frames from {self.num_frames} -> {T_aligned} (1 + n*{ld})")
            self.num_frames = T_aligned

        # --- stronger segment filter: require min latents at this stride ---
        k = int(self.num_start_frames)
        min_latents = k + 2  # your training requirement
        s = int(self.frame_stride)
        nfps = float(self.native_fps)

        min_valid_needed = 1 + (min_latents - 1) * ld  # RGB frames you must actually sample
        raw_frames_needed = 1 + (min_valid_needed - 1) * s  # raw frames in the encoded video at stride s
        min_seconds = raw_frames_needed / nfps

        self.max_retries = int(max_retries)
        self.blank_value = 0

        # after: self.frame_stride = max(1, stride)
        # ld = int(self.latent_downsample)
        # k  = int(self.num_start_frames)
        # s  = int(self.frame_stride)
        # nfps = float(self.native_fps)

        # we need seg_len_raw_frames >= s*(ld*k) + 1
        # min_seg_frames = s * (ld * k) + 1
        # min_seconds = min_seg_frames / nfps

        kept, dropped = 0, 0
        cleaned = []
        for it in annotations:
            segs_all = it.get("caption", []) or []
            segs = []
            for sgm in segs_all:
                st = sgm.get("start", None)
                en = sgm.get("end", None)
                if isinstance(st, (int, float)) and isinstance(en, (int, float)) and en > st:
                    if (en - st) + 1e-6 >= min_seconds:  # tiny epsilon for rounding
                        segs.append(sgm)
                    else:
                        dropped += 1
                else:
                    dropped += 1
            if segs:
                cleaned.append({"filename": it["filename"], "segments": segs})
                kept += len(segs)

        self.entries = cleaned
        print(
            f"Segmented entries (≥1 valid segment): {len(self.entries)}  | "
            f"segments kept: {kept}, dropped (too short): {dropped}"
        )

        # (optional) sanity: if self.num_frames < (ld*k + 1), you’ll *never* meet the requirement
        min_valid_len_needed = ld * k + 1
        if self.num_frames < min_valid_len_needed:
            print(
                f"[warn] num_frames={self.num_frames} < required valid_len={min_valid_len_needed} "
                f"for (ld={ld}, k={k}). Consider increasing num_frames or relaxing k."
            )

        # --- NEW: will be set lazily inside workers ---
        self._shard_ready = False
        self._my_entries = None
        self._global_worker_id = None
        self._global_num_workers = None
        self._shard_style = shard_style

        print(f"Segmented entries (videos with ≥1 segment): {len(self.entries)}")

    def _setup_worker_shard(self):
        """Compute per-worker/ per-rank video subset exactly once."""
        if self._shard_ready:
            return

        # Worker info (DataLoader)
        wi = get_worker_info()
        wid, wnum = (wi.id, wi.num_workers) if wi is not None else (0, 1)

        print("worker id and num_workers", wid, wnum)

        # DDP rank/world (if initialized)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()

        gnum = world * wnum
        gid = rank * wnum + wid

        entries = self.entries
        n = len(entries)
        if n == 0:
            self._my_entries = []
        else:
            if self._shard_style == "contiguous":
                # contiguous chunk per global worker
                start = (n * gid) // gnum
                end = (n * (gid + 1)) // gnum
                self._my_entries = entries[start:end]
            else:
                # interleaved (round-robin) sharding – better balance if videos vary in segment count
                self._my_entries = entries[gid::gnum]

        # Save
        self._global_worker_id = gid
        self._global_num_workers = gnum
        self._shard_ready = True

    def __len__(self):
        # Infinite-style length, like your existing loader
        return 10_000_000 if len(self.entries) else 0

    # ---------- internals ----------

    def _pick_segment(self):
        """Pick a random (video, segment) pair and resolve its path + times."""
        vi = np.random.randint(0, len(self.entries))
        vid_meta = self.entries[vi]
        seg = random.choice(vid_meta["segments"])
        path = os.path.join(self.vid_root, vid_meta["filename"])
        return path, seg

    def _pick_segment_from_shard(self, idx: int):
        """Map a global dataloader index to this worker's video subset, then pick a random segment within that video."""
        self._setup_worker_shard()
        if not self._my_entries:
            # no data; fall back to original list to avoid crashes
            pool = self.entries
        else:
            pool = self._my_entries

        # vi = idx % len(pool)                      # stable per worker
        vi = np.random.randint(len(pool))
        vid_meta = pool[vi]
        seg = random.choice(vid_meta["segments"])  # still random within that video
        path = os.path.join(self.vid_root, vid_meta["filename"])
        return path, seg

    def _segment_frame_window(self, total_frames: int, start_s: float, end_s: float):
        """
        Map segment seconds -> inclusive frame index window within the video.
        We clamp safely to [0, total_frames-1] and require start <= end.
        """
        # Convert using assumed native_fps to stay consistent with your design.
        start_idx = int(round(start_s * self.native_fps))
        end_idx = int(round(end_s * self.native_fps)) - 1  # [start, end)
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx, min(end_idx, total_frames - 1))
        return start_idx, end_idx

    # def _fetch_one(self) -> Datapoint:
    #     vid_path, seg = self._pick_segment()
    def _fetch_one(self, idx_for_shard: int) -> Datapoint:
        vid_path, seg = self._pick_segment_from_shard(idx_for_shard)
        if not os.path.isfile(vid_path):
            raise FileNotFoundError(vid_path)

        reader = VideoReader(vid_path, num_threads=2, ctx=cpu(0), width=self.width, height=self.height)
        total = len(reader)
        if total < 1:
            raise ValueError("Empty video")

        # Segment window (inclusive indices)
        seg_start_idx, seg_end_idx = self._segment_frame_window(total, float(seg["start"]), float(seg["end"]))
        seg_len = seg_end_idx - seg_start_idx + 1
        if seg_len <= 0:
            raise ValueError("Zero-length segment after indexing")

        T = self.num_frames
        stride = self.frame_stride

        # How many frames can we take from this segment at this stride?
        # valid_len is capped by both T and segment length at stride.
        max_valid = 1 + (seg_len - 1) // stride
        valid_len = min(T, max_valid)

        ld = self.latent_downsample  # 8

        ld = self.latent_downsample
        T = self.num_frames
        stride = self.frame_stride

        # frames available in the segment at this stride
        max_valid = 1 + (seg_len - 1) // stride

        # minimum requirement in RGB frames (first + groups of ld)
        min_latents = self.num_start_frames + 2
        min_valid_needed = 1 + (min_latents - 1) * ld

        if max_valid < min_valid_needed:
            raise RuntimeError("Segment too short for min latent length")

        # choose a target ≤ both T and max_valid, then ALIGN: L = 1 + k*ld
        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed
        # Now (L - 1) % ld == 0 and L ≤ T and L ≤ max_valid

        # pick start so exactly L frames fit at this stride
        start_low = seg_start_idx
        start_high = seg_end_idx - (L - 1) * stride
        if start_high < start_low:
            start_high = start_low
        start = int(np.random.randint(start_low, start_high + 1))

        # realize indices & frames
        idx_valid = np.arange(start, start + L * stride, stride, dtype=np.int64)  # len L
        frames_valid = reader.get_batch(idx_valid).to(torch.float32) / 127.5 - 1.0  # [L,H,W,3]

        # pad to fixed T for batching
        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        # masks & lengths (ONLY from L)
        frame_mask = torch.zeros(T, dtype=torch.bool)
        frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld
        video_length = L

        # tensors
        img_tensor = frames.permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()  # [3,1,H,W]

        # hard sanity
        # assert img_tensor.shape[1] == T
        # assert frame_mask.numel() == T
        # assert frame_indices.shape[0] == T
        # assert (L - 1) % ld == 0

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": seg.get("caption", "") or "",
                "caption_idx": torch.tensor(0),
                "crop_coords": torch.zeros(8),
                "anchor_frame": anchor_tensor,
                "frame_mask": frame_mask,  # [T]
                "video_length": torch.tensor(video_length),  # L (aligned)
                "latent_length": torch.tensor(latent_length),  # 1 + (L-1)//ld
                "stride": torch.tensor(stride),
                "frame_indices": torch.from_numpy(frame_indices),
                "segment_start_s": torch.tensor(float(seg["start"])),
                "segment_end_s": torch.tensor(float(seg["end"])),
            },
        )

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one(idx)  # pass idx so we choose from this worker's shard
                latent_len = int(dp.condition["latent_length"])
                if latent_len >= self.num_start_frames + 2:
                    return dp
            except Exception:
                continue
        return self._fetch_one(idx)

# import os, csv, math, random, glob
# import numpy as np
# import tqdm, torch
# from torch.utils.data import Dataset
# from engine.data_classes import Datapoint
# from decord import VideoReader, cpu
# from decord import bridge as decord_bridge

# decord_bridge.set_bridge('torch')

# def _kinetics_filename(youtube_id: str, t0: float, t1: float) -> str:
#     # CSV shows integer seconds; filenames are zero-padded to 6 digits.
#     s0 = int(round(t0))
#     s1 = int(round(t1))
#     return f"{youtube_id}_{s0:06d}_{s1:06d}.mp4"

# def _find_video_file(videos_root: str, label: str, youtube_id: str, t0: float, t1: float) -> str | None:
#     # 1) exact expected path (what your ls shows)
#     fn = _kinetics_filename(youtube_id, t0, t1)
#     p1 = os.path.join(videos_root, fn)
#     if os.path.isfile(p1):
#         return p1

#     # 2) sometimes data is under a label subfolder
#     p2 = os.path.join(videos_root, label, fn)
#     if os.path.isfile(p2):
#         return p2

#     # 3) permissive fallback: any file in root that matches the youtube id
#     # (handles minor timestamp rounding differences)
#     hits = glob.glob(os.path.join(videos_root, f"{youtube_id}_*.mp4"))
#     if hits:
#         # Prefer an exact end-start≈10s match if multiple
#         def _score(path):
#             try:
#                 base = os.path.basename(path)
#                 a, b = base.rsplit("_", 2)[-2:]
#                 s0, s1 = int(a), int(b.split(".")[0])
#                 return -abs((s1 - s0) - int(round(t1 - t0)))
#             except Exception:
#                 return -999
#         hits.sort(key=_score, reverse=True)
#         return hits[0]

#     return None


# class KineticsDatasetFlowception(Dataset):
#     """
#     Flowception-compatible Dataset for Kinetics CSVs.
#     Assumes files like <videos_root>/<youtube_id>_<start:06d>_<end:06d>.mp4
#     """
#     def __init__(
#         self,
#         csv_path: str,
#         videos_root: str,
#         width: int,
#         height: int,
#         split: str = "train",
#         num_frames: int = 72,
#         sampling_fps: float = 24.0,
#         pick_files: int | None = None,
#         num_start_frames: int = 2,
#         latent_downsample: int = 8,
#         max_retries: int = 20,
#         allow_missing: bool = False,
#         verbose_missing: bool = True,
#     ):
#         self.width = int(width)
#         self.height = int(height)
#         self.num_start_frames = int(num_start_frames)
#         self.latent_downsample = int(latent_downsample)
#         self.max_retries = int(max_retries)
#         self.sampling_fps = float(sampling_fps)

#         # Align T to 1 + n*ld
#         T_req = int(num_frames)
#         ld = self.latent_downsample
#         if T_req <= 1:
#             raise ValueError("num_frames must be >= 2 for (first + groups-of-ld) rule")
#         rem = (T_req - 1) % ld
#         if rem != 0:
#             T_aligned = T_req + (ld - rem)
#             print(f"[info] aligning num_frames from {T_req} -> {T_aligned} (1 + n*{ld})")
#             T_req = T_aligned
#         self.num_frames = T_req

#         # Load CSV
#         rows = []
#         with open(csv_path, "r", newline="") as f:
#             rdr = csv.DictReader(f)
#             for r in rdr:
#                 if split and r.get("split", "").strip().lower() != split:
#                     continue
#                 try:
#                     rows.append((
#                         r["label"].strip(),
#                         r["youtube_id"].strip(),   # can contain '-' or '_', that's fine
#                         float(r["time_start"]),
#                         float(r["time_end"]),
#                     ))
#                 except Exception:
#                     continue

#         if pick_files is not None and pick_files < len(rows):
#             rows = random.sample(rows, pick_files)

#         # Resolve filepaths
#         self.entries = []
#         missing = 0
#         for label, ytid, t0, t1 in tqdm.tqdm(rows, desc="Indexing Kinetics"):
#             vpath = _find_video_file(videos_root, label, ytid, t0, t1)
#             if vpath and os.path.isfile(vpath):
#                 self.entries.append({
#                     "label": label,
#                     "youtube_id": ytid,
#                     "time_start": float(t0),
#                     "time_end": float(t1),
#                     "filepath": vpath,
#                 })
#             else:
#                 missing += 1
#                 if verbose_missing:
#                     print(f"[warn] missing: {ytid} [{t0},{t1}] under {videos_root}")
#                 if not allow_missing:
#                     continue

#         if not self.entries:
#             raise RuntimeError("No videos found. Check videos_root/csv layout.")

#         self.streaming_len = max(2_000_000, len(self.entries) * 50)
#         self._meta_cache = {}

#     def __len__(self):
#         return self.streaming_len

#     def _get_meta(self, vpath: str):
#         meta = self._meta_cache.get(vpath)
#         if meta is not None:
#             return meta
#         vr = VideoReader(vpath, num_threads=0, ctx=cpu(0), width=self.width, height=self.height)
#         total = len(vr)
#         try:
#             fps = float(vr.get_avg_fps())
#             if not math.isfinite(fps) or fps <= 0:
#                 fps = 24.0
#         except Exception:
#             fps = 24.0
#         self._meta_cache[vpath] = (total, fps)
#         return total, fps

#     def _fetch_one(self):
#         # random entry
#         item = self.entries[np.random.randint(0, len(self.entries))]
#         vpath = item["filepath"]

#         total, native_fps = self._get_meta(vpath)
#         if total < 1:
#             raise ValueError("Empty video")

#         # Make reader for decoding
#         vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0), width=self.width, height=self.height)
#         total = len(vr)

#         # These files are *already trimmed* to [time_start,time_end] — e.g. 10s clips.
#         # So we just sample within [0, total-1].
#         s = max(1, int(round(native_fps / max(1e-6, self.sampling_fps))))
#         ld = self.latent_downsample
#         k  = self.num_start_frames
#         T  = self.num_frames

#         max_valid = 1 + (total - 1) // s

#         # Flowception minimum: first + (k extra) + 2 latents total
#         min_latents      = k + 2
#         min_valid_needed = 1 + (min_latents - 1) * ld
#         if max_valid < min_valid_needed:
#             raise RuntimeError("Clip too short for min latent length")

#         target = min(T, max_valid)
#         L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
#         if L < min_valid_needed:
#             L = min_valid_needed

#         max_start = total - (L - 1) * s
#         start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

#         idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
#         frames_valid = vr.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
#         frames_valid = frames_valid / 127.5 - 1.0

#         if L < T:
#             last = frames_valid[-1:].clone()
#             frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)
#             frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
#         else:
#             frames = frames_valid
#             frame_indices = idx_valid

#         frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
#         latent_length = 1 + (L - 1) // ld
#         video_length  = L

#         img_tensor    = frames.permute(3, 0, 1, 2).contiguous()           # [C,T,H,W]
#         anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous() # [3,1,H,W]
#         crop_coords   = torch.zeros(8)

#         return Datapoint(
#             pixel_values=img_tensor,
#             condition={
#                 "class_id": item.get("label", ""),
#                 "caption_idx": torch.tensor(0),
#                 "crop_coords": crop_coords,
#                 "anchor_frame": anchor_tensor,
#                 "frame_mask": frame_mask,
#                 "video_length": torch.tensor(video_length),
#                 "latent_length": torch.tensor(latent_length),
#                 "stride": torch.tensor(s),
#                 "frame_indices": torch.from_numpy(frame_indices),
#             },
#         )

#     def __getitem__(self, idx):
#         for _ in range(self.max_retries):
#             try:
#                 dp = self._fetch_one()
#                 if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
#                     return dp
#             except Exception:
#                 continue
#         return self._fetch_one()


import os, csv, math, random
import numpy as np
import torch
from torch.utils.data import Dataset
from engine.data_classes import Datapoint
from decord import VideoReader, cpu
from decord import bridge as decord_bridge

decord_bridge.set_bridge("torch")


def _kinetics_filename(youtube_id: str, t0: float, t1: float) -> str:
    s0 = int(round(t0))
    s1 = int(round(t1))
    return f"{youtube_id}_{s0:06d}_{s1:06d}.mp4"


class KineticsDatasetFlowception(Dataset):
    """
    Zero-path-resolution init:
      - Assumes files are at: <videos_root>/<youtube_id>_<start:06d>_<end:06d>.mp4
      - Does NOT check existence during __init__.
    """

    def __init__(
        self,
        csv_path: str,
        videos_root: str,
        width: int,
        height: int,
        split: str = "train",
        num_frames: int = 72,
        sampling_fps: float = 24.0,
        pick_files: int | None = None,
        num_start_frames: int = 2,
        latent_downsample: int = 8,
        max_retries: int = 20,
    ):
        self.videos_root = videos_root
        self.width = int(width)
        self.height = int(height)
        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)
        self.max_retries = int(max_retries)
        self.sampling_fps = float(sampling_fps)

        # Align T to 1 + n*ld
        T_req = int(num_frames)
        ld = self.latent_downsample
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 (first + groups-of-ld)")
        rem = (T_req - 1) % ld
        if rem != 0:
            T_req += ld - rem
        self.num_frames = T_req

        # Fast CSV ingest only
        rows = []
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if split and r.get("split", "").strip().lower() != split:
                    continue
                try:
                    rows.append(
                        (
                            r["label"].strip(),
                            r["youtube_id"].strip(),
                            float(r["time_start"]),
                            float(r["time_end"]),
                        )
                    )
                except Exception:
                    continue

        if pick_files is not None and pick_files < len(rows):
            rows = random.sample(rows, pick_files)

        # Store minimal info; filepath will be formed in _fetch_one
        self.entries = rows
        self.streaming_len = max(2_000_000, len(self.entries) * 50)
        self._meta_cache = {}  # vpath -> (total_frames, fps)

    def __len__(self):
        return self.streaming_len

    def _get_meta(self, vpath: str):
        meta = self._meta_cache.get(vpath)
        if meta is not None:
            return meta
        vr = VideoReader(vpath, num_threads=0, ctx=cpu(0), width=self.width, height=self.height)
        total = len(vr)
        try:
            fps = float(vr.get_avg_fps())
            if not math.isfinite(fps) or fps <= 0:
                fps = 24.0
        except Exception:
            fps = 24.0
        self._meta_cache[vpath] = (total, fps)
        return total, fps

    def _fetch_one(self):
        label, ytid, t0, t1 = self.entries[np.random.randint(0, len(self.entries))]
        vpath = os.path.join(self.videos_root, _kinetics_filename(ytid, t0, t1))

        # These files are already trimmed to [t0, t1]
        total, native_fps = self._get_meta(vpath)
        if total < 1:
            raise RuntimeError("Empty video")

        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0), width=self.width, height=self.height)
        total = len(vr)

        s = max(1, int(round(native_fps / max(1e-6, self.sampling_fps))))
        ld = self.latent_downsample
        k = self.num_start_frames
        T = self.num_frames

        max_valid = 1 + (total - 1) // s
        min_latents = k + 2
        min_valid_needed = 1 + (min_latents - 1) * ld
        if max_valid < min_valid_needed:
            raise RuntimeError("Clip too short for min latent length")

        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed

        max_start = total - (L - 1) * s
        start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
        frames_valid = vr.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
        frames_valid = frames_valid / 127.5 - 1.0

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

        img_tensor = frames.permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()  # [3,1,H,W]
        crop_coords = torch.zeros(8)

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": label,
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
                dp = self._fetch_one()
                if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
                    return dp
            except Exception:
                continue
        return self._fetch_one()

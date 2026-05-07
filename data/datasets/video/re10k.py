# import os
# import random
# import glob
# import tqdm
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from decord import VideoReader, cpu
# from decord import bridge as decord_bridge

# decord_bridge.set_bridge("torch")  # return torch tensors from decord

# from engine.data_classes import Datapoint


# # -----------------------------
# # Helpers
# # -----------------------------

# def _rgb_len_for_latents(n_latents: int, ld: int) -> int:
#     """
#     RGB frames needed to yield `n_latents` temporal latents under:
#       - first RGB frame -> 1 latent;
#       - then every `ld` RGB frames -> +1 latent.
#     """
#     if n_latents <= 0:
#         return 0
#     return 1 + (n_latents - 1) * ld


# def build_entries_from_dir(
#     video_dir: str,
#     extensions=(".mp4", ".MP4", ".mov", ".MOV"),
#     recursive: bool = False,
# ):
#     """
#     Scan a directory and return entries like {"filepath": "/abs/path/to/clip.mp4", "description": ""}.
#     """
#     video_dir = os.path.abspath(video_dir)
#     pattern = "**/*" if recursive else "*"
#     paths = [
#         p for p in glob.glob(os.path.join(video_dir, pattern))
#         if os.path.splitext(p)[1] in extensions and os.path.isfile(p)
#     ]
#     entries = [{"filepath": os.path.abspath(p), "description": ""} for p in paths]
#     return entries


# # -----------------------------
# # Dataset
# # -----------------------------

# class Re10kMP4DatasetFlowception(Dataset):
#     """
#     Video-only (MP4) dataset for Flowception-style training that reads videos
#     directly from a directory (no .pt annotation shards needed).

#     - Interprets `num_start_latents` and `num_context_latents` in **latent space**.
#     - Guarantees each returned clip has at least `num_context_latents + num_start_latents` latents.
#     - Aligns RGB length to VAE temporal rule: T = 1 + n * latent_downsample.

#     Expected directory structure (flat or nested):
#         /path/to/videos/*.mp4

#     Returns a Datapoint with:
#       pixel_values: float32 in [-1,1], [C, T, H, W]
#       condition: {
#         "class_id": str,
#         "caption_idx": tensor(0),
#         "crop_coords": zeros(8),
#         "anchor_frame": [3,1,H,W],            # first real RGB frame (for conditioners)
#         "frame_mask": [T] bool,               # True for real frames, False for pad
#         "video_length": L,                    # real RGB frames (aligned)
#         "latent_length": 1 + (L-1)//ld,       # total latents represented in L RGB frames
#         "stride": stride,
#         "frame_indices": [T] int64,
#         "num_context_latents": K,
#         "num_start_latents": k,
#       }
#     """

#     def __init__(
#         self,
#         video_dir: str,              # directory containing .mp4 files
#         width: int,
#         height: int,
#         num_frames: int = 72,        # requested RGB length; ceil-aligned to 1 + n*ld
#         sampling_fps: float = 24.0,  # target sampling fps
#         native_fps: float = 24.0,    # nominal native fps for stride calc
#         num_start_latents: int = 2,  # k (latent-space warmup AFTER context)
#         num_context_latents: int = 0,# K (latent-space context at the very start)
#         latent_downsample: int = 8,  # ld
#         max_retries: int = 20,
#         pick_files: int | None = None,  # randomly subsample this many files from the dir
#         shuffle_filelist: bool = True,
#         recursive: bool = False,
#         seed: int | None = 17,
#     ):
#         # Gather file entries from directory
#         entries = build_entries_from_dir(video_dir, recursive=recursive)
#         if len(entries) == 0:
#             raise RuntimeError(f"No video files found in: {video_dir}")

#         if shuffle_filelist:
#             rng = random.Random(seed)
#             rng.shuffle(entries)

#         if pick_files is not None and pick_files > 0:
#             entries = entries[: int(pick_files)]

#         self.all_entries = entries
#         print(f"[Re10kMP4Dataset] items: {len(entries)} (from: {os.path.abspath(video_dir)})")

#         # ----- geometry / fps / stride -----
#         self.width  = int(width)
#         self.height = int(height)
#         self.sampling_fps = float(sampling_fps)
#         self.native_fps   = float(native_fps)

#         stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
#         self.frame_stride = max(1, stride)

#         # ----- Flowception knobs (latent-space semantics) -----
#         self.num_start_latents   = int(max(0, num_start_latents))   # k
#         self.num_context_latents = int(max(0, num_context_latents)) # K
#         self.latent_downsample   = int(latent_downsample)           # ld
#         self.max_retries         = int(max_retries)

#         # ----- Align requested T to 1 + n*ld (ceil) -----
#         T_req = int(num_frames)
#         if T_req <= 1:
#             raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
#         ld = self.latent_downsample
#         rem = (T_req - 1) % ld
#         if rem != 0:
#             T_req += (ld - rem)
#             print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
#         self.num_frames = T_req

#         # Minimum latent count required
#         # Need at least k start latents *after* the very first latent (which is always kept)
#         # => total latents must be >= K + (k + 1)
#         self.min_total_latents = max(1, self.num_context_latents + self.num_start_latents + 1)

#     def __len__(self):
#         # streaming-style: effectively infinite epochs over the directory
#         return 1_000_000

#     def _fetch_one(self):
#         it = self.all_entries[np.random.randint(0, len(self.all_entries))]
#         path = it["filepath"]
#         if not os.path.isfile(path):
#             raise FileNotFoundError(path)

#         reader = VideoReader(
#             path, num_threads=-1, ctx=cpu(0),
#             width=self.width, height=self.height
#         )
#         total = len(reader)
#         if total < 1:
#             raise ValueError(f"Empty video: {path}")

#         ld = self.latent_downsample
#         K  = self.num_context_latents
#         k  = self.num_start_latents
#         s  = self.frame_stride
#         T  = self.num_frames

#         # frames available at stride s
#         max_valid = 1 + (total - 1) // s

#         # Ensure at least (K + k) latents overall
#         min_latents_total = self.min_total_latents
#         min_valid_needed = _rgb_len_for_latents(min_latents_total, ld)
#         if max_valid < min_valid_needed:
#             raise RuntimeError(
#                 f"Clip too short: need ≥{min_latents_total} latents -> "
#                 f"≥{min_valid_needed} RGB frames at stride {s} (file: {os.path.basename(path)})"
#             )

#         # choose aligned L (≤ T and ≤ max_valid); ensure L ≥ min_valid_needed
#         target = min(T, max_valid)
#         L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
#         if L < min_valid_needed:
#             L = min_valid_needed

#         # random start that fits L at stride s
#         max_start = total - (L - 1) * s
#         start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

#         # indices and frames
#         idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
#         frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
#         frames_valid = frames_valid / 127.5 - 1.0                     # [-1,1]

#         # pad to fixed T (repeat-last to batch)
#         if L < T:
#             last = frames_valid[-1:].clone()
#             frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
#             frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
#         else:
#             frames = frames_valid
#             frame_indices = idx_valid

#         # masks & lengths
#         frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
#         latent_length = 1 + (L - 1) // ld
#         video_length  = L

#         # pack tensors
#         img_tensor    = frames.permute(3, 0, 1, 2).contiguous()           # [C,T,H,W]
#         anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous() # [3,1,H,W]

#         # sanity
#         assert img_tensor.shape[1] == T
#         assert frame_mask.numel() == T
#         assert frame_indices.shape[0] == T
#         assert (L - 1) % ld == 0

#         crop_coords = torch.zeros(8)
#         description = it.get("description", "") or ""

#         return Datapoint(
#             pixel_values=img_tensor,
#             condition={
#                 "class_id": description,
#                 "caption_idx": torch.tensor(0),
#                 "crop_coords": crop_coords,
#                 "anchor_frame": anchor_tensor,
#                 "frame_mask": frame_mask,
#                 "video_length": torch.tensor(video_length),
#                 "latent_length": torch.tensor(latent_length),
#                 "stride": torch.tensor(s),
#                 "frame_indices": torch.from_numpy(frame_indices),
#                 "num_context_latents": torch.tensor(K),
#                 "num_start_latents": torch.tensor(k),
#             },
#         )

#     def __getitem__(self, idx):
#         # keep sampling until we get a datapoint with ≥ (K + k) latents
#         need_latents = self.min_total_latents
#         for _ in range(self.max_retries):
#             try:
#                 dp = self._fetch_one()
#                 if int(dp.condition["latent_length"]) >= need_latents:
#                     return dp
#             except Exception:
#                 continue
#         # last attempt raises on failure
#         return self._fetch_one()


# re10k_mp4_dataset.py
import os
import random
import glob
import joblib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from decord import VideoReader, cpu
from decord import bridge as decord_bridge

decord_bridge.set_bridge("torch")  # return torch tensors from decord

from engine.data_classes import Datapoint


# -----------------------------
# Helpers
# -----------------------------


def _rgb_len_for_latents(n_latents: int, ld: int) -> int:
    """
    RGB frames needed to yield `n_latents` temporal latents under:
      - first RGB frame -> 1 latent;
      - then every `ld` RGB frames -> +1 latent.
    """
    if n_latents <= 0:
        return 0
    return 1 + (n_latents - 1) * ld


def build_entries_from_dir(
    video_dir: str,
    extensions=(".mp4", ".MP4", ".mov", ".MOV"),
    recursive: bool = False,
):
    """
    Scan a directory and return entries like {"filepath": "/abs/path/to/clip.mp4", "description": ""}.
    """
    video_dir = os.path.abspath(video_dir)
    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in glob.glob(os.path.join(video_dir, pattern), recursive=recursive)
        if os.path.splitext(p)[1] in extensions and os.path.isfile(p)
    ]
    entries = [{"filepath": os.path.abspath(p), "description": ""} for p in paths]
    return entries


# -----------------------------
# Dataset
# -----------------------------


class Re10kMP4DatasetFlowception(Dataset):  # supports dir scan or prebuilt .pt index
    """
    Video-only (MP4) dataset for Flowception-style training.

    Can read videos directly from a directory OR from a prebuilt joblib index (.pt).
    Also shards across DDP ranks and DataLoader workers to minimize directory contention.

    Returns a Datapoint with:
      pixel_values: float32 in [-1,1], [C, T, H, W]
      condition: {
        "class_id": str, "caption_idx": tensor(0), "crop_coords": zeros(8),
        "anchor_frame": [3,1,H,W], "frame_mask": [T] bool,
        "video_length": L, "latent_length": 1 + (L-1)//ld, "stride": stride,
        "frame_indices": [T] int64, "num_context_latents": K, "num_start_latents": k
      }
    """

    def __init__(
        self,
        video_dir: str,  # directory containing .mp4 files (ignored if index_path is used)
        width: int,
        height: int,
        num_frames: int = 72,  # requested RGB length; ceil-aligned to 1 + n*ld
        sampling_fps: float = 24.0,  # target sampling fps
        native_fps: float = 24.0,  # nominal native fps for stride calc
        num_start_latents: int = 2,  # k (latent-space warmup AFTER context)
        num_context_latents: int = 0,  # K (latent-space context at the very start)
        latent_downsample: int = 8,  # ld
        max_retries: int = 200,
        pick_files: int | None = None,  # randomly subsample this many files
        shuffle_filelist: bool = True,
        recursive: bool = False,
        seed: int | None = 17,
        # NEW: index support
        index_path: str | None = None,  # load entries from a prebuilt joblib .pt
        cache_index_to: str | None = None,  # after scanning, write entries to this .pt
    ):
        # ---- Load entries (prefer fast index if provided) ----
        if index_path is not None:
            entries = joblib.load(index_path)
            if not isinstance(entries, (list, tuple)) or len(entries) == 0:
                raise RuntimeError(f"Bad or empty index file: {index_path}")
        else:
            entries = build_entries_from_dir(video_dir, recursive=recursive)
            if len(entries) == 0:
                raise RuntimeError(f"No video files found in: {video_dir}")
            if cache_index_to is not None:
                os.makedirs(os.path.dirname(os.path.abspath(cache_index_to)), exist_ok=True)
                joblib.dump(entries, cache_index_to)

        if shuffle_filelist:
            rng = random.Random(seed)
            rng.shuffle(entries)

        if pick_files is not None and pick_files > 0:
            entries = entries[: int(pick_files)]

        # ---- optional: process-level sharding (DDP) ----
        self.rank = 0
        self.world = 1
        if dist.is_available() and dist.is_initialized():
            try:
                self.rank = dist.get_rank()
                self.world = dist.get_world_size()

            except Exception:
                pass

        # contiguous rank split
        n = len(entries)
        G = max(1, self.world)
        start = (n * self.rank) // G
        end = (n * (self.rank + 1)) // G
        print(
            f"sharding entries (contiguous) - total: {n}, rank {self.rank}/{self.world} -> [{start}:{end}) = {end - start}"
        )
        entries = entries[start:end]
        print(f"after sharding: {len(entries)}")

        self.all_entries = entries
        print(
            f"[Re10kMP4Dataset] items (post-DDP shard {self.rank}/{self.world}): "
            f"{len(entries)} (from: {os.path.abspath(video_dir)})"
            + (f" [index: {os.path.abspath(index_path)}]" if index_path else "")
        )

        # ----- geometry / fps / stride -----
        self.width = int(width)
        self.height = int(height)
        self.sampling_fps = float(sampling_fps)
        self.native_fps = float(native_fps)

        stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
        self.frame_stride = max(1, stride)

        # ----- Flowception knobs (latent-space semantics) -----
        self.num_start_latents = int(max(0, num_start_latents))  # k
        self.num_context_latents = int(max(0, num_context_latents))  # K
        self.latent_downsample = int(latent_downsample)  # ld
        self.max_retries = int(max_retries)

        # ----- Align requested T to 1 + n*ld (ceil) -----
        T_req = int(num_frames)
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
        ld = self.latent_downsample
        rem = (T_req - 1) % ld
        if rem != 0:
            T_req += ld - rem
            print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
        self.num_frames = T_req

        # Minimum latent count required
        # Need at least k start latents *after* the very first latent (which is always kept)
        # => total latents must be >= K + (k + 1)
        self.min_total_latents = max(1, self.num_context_latents + self.num_start_latents + 1)

        # for worker-local RNG & split
        self._base_seed = int(seed if seed is not None else 0)
        self._worker_entries = None
        self._worker_seeded = False

    def __len__(self):
        # streaming-style: effectively infinite epochs over the directory
        return 1_000_000

    # ---- DataLoader worker sharding ----
    def _get_worker_entries(self):
        """
        Shard this process's entry list across DataLoader workers (stride split).
        Lazily computed inside each worker process.
        """
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            return self.all_entries
        if self._worker_entries is None:
            base = self.all_entries
            shard = base[wi.id :: wi.num_workers]  # worker i: i, i+n, i+2n, ...
            if not shard:  # fallback if empty
                shard = base
            self._worker_entries = shard
        if not self._worker_seeded:
            wi = torch.utils.data.get_worker_info()
            wid = wi.id if wi else 0
            np.random.seed(self._base_seed + 1000 * self.rank + wid)
            self._worker_seeded = True
        return self._worker_entries

    def _fetch_one(self):
        entries = self._get_worker_entries()
        it = entries[np.random.randint(0, len(entries))]
        path = it["filepath"]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        reader = VideoReader(path, num_threads=-1, ctx=cpu(0), width=self.width, height=self.height)
        total = len(reader)
        if total < 1:
            raise ValueError(f"Empty video: {path}")

        ld = self.latent_downsample
        K = self.num_context_latents
        k = self.num_start_latents
        s = self.frame_stride
        T = self.num_frames

        # frames available at stride s
        max_valid = 1 + (total - 1) // s

        # Ensure at least (K + k + 1) latents overall
        min_latents_total = self.min_total_latents
        min_valid_needed = _rgb_len_for_latents(min_latents_total, ld)
        if max_valid < min_valid_needed:
            raise RuntimeError(
                f"Clip too short: need ≥{min_latents_total} latents -> "
                f"≥{min_valid_needed} RGB frames at stride {s} (file: {os.path.basename(path)})"
            )

        # choose aligned L (≤ T and ≤ max_valid); ensure L ≥ min_valid_needed
        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed

        # random start that fits L at stride s
        max_start = total - (L - 1) * s
        start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

        # indices and frames
        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
        frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
        frames_valid = frames_valid / 127.5 - 1.0  # [-1,1]

        # pad to fixed T (repeat-last to batch)
        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        # masks & lengths
        frame_mask = torch.zeros(T, dtype=torch.bool)
        frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld
        video_length = L

        # pack tensors
        img_tensor = frames.permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()  # [3,1,H,W]

        # sanity
        assert img_tensor.shape[1] == T
        assert frame_mask.numel() == T
        assert frame_indices.shape[0] == T
        assert (L - 1) % ld == 0

        crop_coords = torch.zeros(8)
        description = it.get("description", "") or ""

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": description,
                "caption_idx": torch.tensor(0),
                "crop_coords": crop_coords,
                "anchor_frame": anchor_tensor,
                "frame_mask": frame_mask,
                "video_length": torch.tensor(video_length),
                "latent_length": torch.tensor(latent_length),
                "stride": torch.tensor(s),
                "frame_indices": torch.from_numpy(frame_indices),
                "num_context_latents": torch.tensor(K),
                "num_start_latents": torch.tensor(k),
            },
        )

    def __getitem__(self, idx):
        # keep sampling until we get a datapoint with ≥ (K + k + 1) latents
        need_latents = self.min_total_latents
        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one()
                if int(dp.condition["latent_length"]) >= need_latents:
                    return dp
            except Exception:
                continue
        # last attempt raises on failure
        return self._fetch_one()

    # ---------- utilities ----------
    @staticmethod
    def write_index(video_dir: str, out_pt: str, recursive: bool = False):
        """
        Build and save a joblib index (.pt) listing video file entries.
        Each entry is {"filepath": abs_path, "description": ""}.
        """
        entries = build_entries_from_dir(video_dir, recursive=recursive)
        if len(entries) == 0:
            raise RuntimeError(f"No video files found in: {video_dir}")
        os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
        joblib.dump(entries, out_pt)
        return len(entries)

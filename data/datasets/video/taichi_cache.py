# # taichi_inmemory_dataset.py
# import os
# import joblib
# import tqdm
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from decord import VideoReader, cpu
# from decord import bridge as decord_bridge
# decord_bridge.set_bridge('torch')  # decord returns torch tensors

# from engine.data_classes import Datapoint


# class TaichiInMemoryFlowception(Dataset):
#     """
#     Fully in-memory, video-only (MP4) dataset for Flowception-style training.

#     Annotations dir (or single .pt) must contain entries like:
#         {"filepath": "/abs/path/to/clip.mp4", "description": ""}

#     Preloads every video at __init__ into uint8 tensors [L, H, W, 3] (resized).
#     During sampling we normalize to [-1, 1], align length to T = 1 + n*ld, and
#     pad by repeating the last real frame.

#     Tip for multi-worker: create the dataset in the main process, then pass it
#     to a DataLoader with persistent_workers=True. On Linux (fork), RAM is shared.
#     """

#     def __init__(
#         self,
#         annotations_dir: str,            # dir of .pt shards OR a single .pt file
#         width: int,
#         height: int,
#         num_frames: int = 72,            # requested T (ceil-align to 1 + n*ld)
#         sampling_fps: float = 24.0,      # target sampling fps
#         native_fps: float = 24.0,        # nominal native fps for stride calc
#         num_start_frames: int = 2,       # k
#         latent_downsample: int = 8,      # ld
#         pick_files: int | None = None,   # subsample shards before loading
#         max_retries: int = 20,
#         preload: bool = True,            # keep True for full in-RAM speed
#         verbose: bool = True,
#         min_motion_score: float | None = None,
#     ):
#         self.width  = int(width)
#         self.height = int(height)
#         self.sampling_fps = float(sampling_fps)
#         self.native_fps   = float(native_fps)
#         self.num_start_frames  = int(num_start_frames)   # k
#         self.latent_downsample = int(latent_downsample)  # ld
#         self.max_retries       = int(max_retries)
#         self.preload = bool(preload)

#         # -------- load annotations --------
#         if os.path.isdir(annotations_dir):
#             shard_names = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
#             if not shard_names:
#                 raise FileNotFoundError(f"No .pt files in {annotations_dir}")
#             shard_names.sort()
#             if pick_files is not None and pick_files < len(shard_names):
#                 idx = np.random.choice(len(shard_names), pick_files, replace=False)
#                 shard_names = [shard_names[i] for i in sorted(idx)]
#             shard_paths = [os.path.join(annotations_dir, s) for s in shard_names]
#         else:
#             shard_paths = [annotations_dir]

#         entries = []
#         for sp in shard_paths:
#             data = joblib.load(sp)
#             entries.extend(data)
#         if not entries:
#             raise RuntimeError("No entries loaded from annotations.")
#         self.entries = entries
#         if verbose:
#             print(f"[TaichiInMemory] annotation items: {len(self.entries)}")

#         # -------- stride / alignment --------
#         stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
#         self.frame_stride = max(1, stride)

#         T_req = int(num_frames)
#         if T_req <= 1:
#             raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
#         ld = self.latent_downsample
#         rem = (T_req - 1) % ld
#         if rem != 0:
#             T_req += (ld - rem)
#             if verbose:
#                 print(f"[TaichiInMemory] aligning num_frames -> {T_req} (1 + n*{ld})")
#         self.num_frames = T_req

#         # -------- optional full preload --------
#         self._videos_u8   = []  # list[torch.uint8] of shape [L,H,W,3]
#         self._lengths     = []  # list[int]
#         self._descriptions= []  # list[str]
#         if self.preload:
#             total_bytes = 0
#             for it in tqdm.tqdm(self.entries, desc="Preloading videos to RAM"):
#                 path = it["filepath"]
#                 desc = it.get("description", "") or ""
#                 if not os.path.isfile(path):
#                     continue
#                 # Decode and resize into uint8 (keep compact in RAM)
#                 vr = VideoReader(path, num_threads=1, ctx=cpu(0),
#                                  width=self.width, height=self.height)
#                 L = len(vr)
#                 if L < 1:
#                     continue
#                 frames_u8 = vr.get_batch(range(L)).to(torch.uint8)  # [L,H,W,3], 0..255
#                 # ensure contiguous; NEVER modify later (share via COW)
#                 frames_u8 = frames_u8.contiguous()
#                 self._videos_u8.append(frames_u8)
#                 self._lengths.append(int(L))
#                 self._descriptions.append(desc)
#                 total_bytes += frames_u8.numel()  # uint8 => bytes
#             if not self._videos_u8:
#                 raise RuntimeError("Preload failed: no valid videos decoded.")
#             if verbose:
#                 gb = total_bytes / (1024**3)
#                 print(f"[TaichiInMemory] preloaded {len(self._videos_u8)} videos "
#                       f"@ {self.width}x{self.height}; RAM ~{gb:.2f} GiB (uint8).")

#     def __len__(self):
#         # streaming-style epochs
#         return 5_000_000

#     # ---- core sampling, shared by both preload / stream paths ----
#     def _sample_from_frames(self, frames_u8: torch.Tensor, desc: str):
#         """
#         frames_u8: uint8, [L,H,W,3] (already resized)
#         """
#         L_total = int(frames_u8.shape[0])
#         ld = self.latent_downsample
#         k  = self.num_start_frames
#         s  = self.frame_stride
#         T  = self.num_frames

#         # how many frames available at stride s
#         max_valid = 1 + (L_total - 1) // s
#         min_latents = k + 2
#         min_valid_needed = 1 + (min_latents - 1) * ld
#         if max_valid < min_valid_needed:
#             raise RuntimeError("Clip too short for min latent length")

#         target = min(T, max_valid)
#         L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
#         if L < min_valid_needed:
#             L = min_valid_needed

#         max_start = L_total - (L - 1) * s
#         start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))
#         idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)

#         # gather and normalize to [-1,1] lazily
#         frames_valid = frames_u8[idx_valid].to(torch.float32) / 127.5 - 1.0  # [L,H,W,3]

#         if L < T:
#             last = frames_valid[-1:].clone()
#             frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)
#             frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
#         else:
#             frames = frames_valid
#             frame_indices = idx_valid

#         frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
#         latent_length = 1 + (L - 1) // ld

#         img_tensor    = frames.permute(3, 0, 1, 2).contiguous()           # [C,T,H,W]
#         anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous() # [3,1,H,W]
#         crop_coords   = torch.zeros(8)

#         return Datapoint(
#             pixel_values=img_tensor,
#             condition={
#                 "class_id": desc,
#                 "caption_idx": torch.tensor(0),
#                 "crop_coords": crop_coords,
#                 "anchor_frame": anchor_tensor,
#                 "frame_mask": frame_mask,
#                 "video_length": torch.tensor(L),
#                 "latent_length": torch.tensor(latent_length),
#                 "stride": torch.tensor(s),
#                 "frame_indices": torch.from_numpy(frame_indices),
#             },
#         )

#     def _fetch_one(self):
#         if self.preload:
#             # sample from preloaded RAM
#             j = np.random.randint(0, len(self._videos_u8))
#             frames_u8 = self._videos_u8[j]       # uint8 [L,H,W,3]
#             desc      = self._descriptions[j]
#             return self._sample_from_frames(frames_u8, desc)
#         else:
#             # fallback: stream from disk (rarely needed)
#             it = self.entries[np.random.randint(0, len(self.entries))]
#             path = it["filepath"]
#             if not os.path.isfile(path):
#                 raise FileNotFoundError(path)
#             vr = VideoReader(path, num_threads=1, ctx=cpu(0),
#                              width=self.width, height=self.height)
#             L = len(vr)
#             if L < 1:
#                 raise RuntimeError("Empty video")
#             frames_u8 = vr.get_batch(range(L)).to(torch.uint8)
#             desc = it.get("description", "") or ""
#             return self._sample_from_frames(frames_u8, desc)

#     def __getitem__(self, idx):
#         for _ in range(self.max_retries):
#             try:
#                 dp = self._fetch_one()
#                 # ensure enough latents
#                 if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
#                     return dp
#             except Exception:
#                 continue
#         return self._fetch_one()




# taichi_inmemory_per_worker.py
import os, joblib, numpy as np, torch, tqdm
from torch.utils.data import Dataset, get_worker_info
from collections import OrderedDict

from decord import VideoReader, cpu
from decord import bridge as decord_bridge
decord_bridge.set_bridge('torch')

from engine.data_classes import Datapoint

def _get_rank_world_size():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    r = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    w = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return r, max(w, 1)

class TaichiInMemoryFlowception(Dataset):
    """
    Each worker (and each DDP rank) preloads *only its shard* of videos into RAM.
    Frames are stored as uint8 [L,H,W,3] to keep memory reasonable.
    """

    def __init__(
        self,
        annotations_dir: str,            # dir of .pt shards OR a single .pt
        width: int,
        height: int,
        num_frames: int = 72,
        sampling_fps: float = 24.0,
        native_fps: float = 24.0,
        num_start_frames: int = 2,
        latent_downsample: int = 8,
        pick_files: int | None = None,   # subsample shards before load (optional)
        max_retries: int = 20,
        seed: int = 12345,
        verbose: bool = True,
        min_motion_score: float | None = None,
    ):
        self.width  = int(width)
        self.height = int(height)
        self.sampling_fps = float(sampling_fps)
        self.native_fps   = float(native_fps)
        self.num_start_frames  = int(num_start_frames)   # k
        self.latent_downsample = int(latent_downsample)  # ld
        self.max_retries       = int(max_retries)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # ----- load annotation entries once (main process) -----
        if os.path.isdir(annotations_dir):
            shard_names = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
            shard_names.sort()
            if not shard_names:
                raise FileNotFoundError(f"No .pt in {annotations_dir}")
            if pick_files is not None and pick_files < len(shard_names):
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(shard_names), pick_files, replace=False)
                shard_names = [shard_names[i] for i in sorted(idx)]
            shard_paths = [os.path.join(annotations_dir, s) for s in shard_names]
        else:
            shard_paths = [annotations_dir]

        entries = []
        for sp in shard_paths:
            data = joblib.load(sp)
            entries.extend(data)
        if not entries:
            raise RuntimeError("No entries loaded from annotations.")
        self._all_entries = entries

        # ----- stride / alignment -----
        stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
        self.frame_stride = max(1, stride)

        T_req = int(num_frames)
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2.")
        ld = self.latent_downsample
        rem = (T_req - 1) % ld
        if rem != 0:
            T_req += (ld - rem)
            if verbose:
                print(f"[TaichiPerWorker] aligning num_frames -> {T_req} (1 + n*{ld})")
        self.num_frames = T_req

        # ----- worker-local state (populated lazily in the worker) -----
        self._worker_ready = False
        self._entries = None            # this worker's shard of entries
        self._videos_u8 = None          # list[torch.uint8 [L,H,W,3]]
        self._descs = None              # list[str]
        self._rng = None

    def _lazy_worker_init(self):
        if self._worker_ready:
            return
        rank, world = _get_rank_world_size()
        wi = get_worker_info()
        num_workers = wi.num_workers if wi is not None else 1
        worker_id = wi.id if wi is not None else 0

        # ----- shard entries by rank, then by worker -----
        n = len(self._all_entries)
        per_rank = (n + world - 1) // world
        r0, r1 = rank * per_rank, min(n, (rank + 1) * per_rank)
        rank_entries = self._all_entries[r0:r1]

        per_worker = (len(rank_entries) + num_workers - 1) // num_workers
        w0, w1 = worker_id * per_worker, min(len(rank_entries), (worker_id + 1) * per_worker)
        entries = rank_entries[w0:w1]
        if not entries:
            # still set ready to avoid trying again
            self._entries = []
            self._videos_u8, self._descs = [], []
            self._rng = np.random.RandomState(self.seed + 17*rank + 1009*worker_id)
            self._worker_ready = True
            return

        # RNG per (rank, worker)
        self._rng = np.random.RandomState(self.seed + 17*rank + 1009*worker_id)

        # ----- preload only this shard into RAM (uint8) -----
        vids, descs = [], []
        total_bytes = 0
        for it in entries:
            path = it["filepath"]
            desc = it.get("description", "") or ""
            if not os.path.isfile(path):
                continue
            vr = VideoReader(path, num_threads=1, ctx=cpu(0),
                             width=self.width, height=self.height)
            L = len(vr)
            if L < 1:
                continue
            frames_u8 = vr.get_batch(range(L)).to(torch.uint8).contiguous()  # [L,H,W,3]
            vids.append(frames_u8)
            descs.append(desc)
            total_bytes += frames_u8.numel()  # uint8 -> bytes

        self._entries   = entries
        self._videos_u8 = vids
        self._descs     = descs
        self._worker_ready = True

        if self.verbose and get_worker_info() is not None:
            gb = total_bytes / (1024**3)
            print(f"[TaichiPerWorker][rank {rank} worker {worker_id}] "
                  f"preloaded {len(vids)} / {len(entries)} videos; ~{gb:.2f} GiB uint8")

    def __len__(self):
        return 5_000_000  # streaming-style

    # ---- common sampler on preloaded frames ----
    def _sample_from_frames(self, frames_u8: torch.Tensor, desc: str):
        L_total = int(frames_u8.shape[0])
        ld = self.latent_downsample
        k  = self.num_start_frames
        s  = self.frame_stride
        T  = self.num_frames

        max_valid = 1 + (L_total - 1) // s
        min_latents = k + 2
        min_valid_needed = 1 + (min_latents - 1) * ld
        if max_valid < min_valid_needed:
            raise RuntimeError("Clip too short for min latent length")

        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed

        max_start = L_total - (L - 1) * s
        start = 0 if max_start <= 0 else int(self._rng.randint(0, max_start))
        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)

        frames_valid = frames_u8[idx_valid].to(torch.float32) / 127.5 - 1.0  # [L,H,W,3]

        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld

        img_tensor    = frames.permute(3, 0, 1, 2).contiguous()
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous()
        crop_coords   = torch.zeros(8)

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": desc,
                "caption_idx": torch.tensor(0),
                "crop_coords": crop_coords,
                "anchor_frame": anchor_tensor,
                "frame_mask": frame_mask,
                "video_length": torch.tensor(L),
                "latent_length": torch.tensor(latent_length),
                "stride": torch.tensor(s),
                "frame_indices": torch.from_numpy(frame_indices),
            },
        )

    def _fetch_one(self):
        if not self._worker_ready:
            self._lazy_worker_init()

        if not self._videos_u8:
            raise RuntimeError("Worker has no videos after preload/sharding.")

        j = int(self._rng.randint(0, len(self._videos_u8)))
        frames_u8 = self._videos_u8[j]
        desc      = self._descs[j]
        return self._sample_from_frames(frames_u8, desc)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one()
                if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
                    return dp
            except Exception:
                continue
        return self._fetch_one()

# # # taichi_mp4_dataset.py
# # import os
# # import joblib
# # import tqdm
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset

# # from decord import VideoReader, cpu
# # from decord import bridge as decord_bridge
# # decord_bridge.set_bridge('torch')  # return torch tensors from decord

# # from engine.data_classes import Datapoint


# # class TaichiDatasetFlowception(Dataset):
# #     """
# #     Video-only (MP4) dataset for Flowception-style training.

# #     Expects an annotations directory (or single .pt file) with entries like:
# #       {"filepath": "/abs/path/to/clip.mp4", "description": ""}

# #     Returns:
# #       Datapoint(
# #         pixel_values: float32 in [-1,1], shape [C, T, H, W]
# #         condition: {
# #           "class_id": str,
# #           "caption_idx": tensor(0),
# #           "crop_coords": zeros(8),
# #           "anchor_frame": [3,1,H,W],
# #           "frame_mask": [T] bool,
# #           "video_length": L (aligned, real frames only),
# #           "latent_length": 1 + (L-1)//latent_downsample,
# #           "stride": stride,
# #           "frame_indices": [T] int64
# #         }
# #       )

# #     Alignment rule: T = 1 + n * latent_downsample (ceil-aligned from requested num_frames).
# #     """

# #     def __init__(
# #         self,
# #         annotations_dir: str,          # dir of .pt shards OR a single .pt file
# #         width: int,
# #         height: int,
# #         num_frames: int = 72,          # requested T (will be ceil aligned)
# #         sampling_fps: float = 24.0,    # target sampling fps
# #         native_fps: float = 24.0,      # nominal native fps for stride calc
# #         num_start_frames: int = 2,     # k
# #         latent_downsample: int = 8,    # ld
# #         min_motion_score: float = 3.0,
# #         pick_files: int | None = None, # randomly subsample shards; None=all
# #         max_retries: int = 20,
# #     ):
        
# #         shard_names = [annotations_dir]
# #         # Load entries (optionally filter by motion score if present + requested)
# #         entries = []
# #         for jp in tqdm.tqdm(shard_names, desc="Loading annotation shards"):
# #             data = joblib.load(os.path.join(annotations_dir, jp))
# #             entries.extend(data)

# #         self.all_entries = entries
# #         print(f"[TaichiMP4Dataset] items: {len(entries)}")

# #         # ----- geometry / fps / stride -----
# #         self.width  = int(width)
# #         self.height = int(height)
# #         self.sampling_fps = float(sampling_fps)
# #         self.native_fps   = float(native_fps)

# #         stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
# #         self.frame_stride = max(1, stride)

# #         # ----- flowception knobs -----
# #         self.num_start_frames  = int(num_start_frames)    # k
# #         self.latent_downsample = int(latent_downsample)   # ld
# #         self.max_retries       = int(max_retries)

# #         # ----- align T: 1 + n*ld -----
# #         T_req = int(num_frames)
# #         if T_req <= 1:
# #             raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
# #         ld = self.latent_downsample
# #         rem = (T_req - 1) % ld
# #         if rem != 0:
# #             T_req += (ld - rem)
# #             print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
# #         self.num_frames = T_req

# #     def __len__(self):
# #         # streaming-style
# #         return 1_000_000

# #     def _fetch_one(self):
# #         it = self.all_entries[np.random.randint(0, len(self.all_entries))]
# #         path = it["filepath"]
# #         if not os.path.isfile(path):
# #             raise FileNotFoundError(path)

# #         reader = VideoReader(
# #             path, num_threads=-1, ctx=cpu(0),
# #             width=self.width, height=self.height
# #         )
# #         total = len(reader)
# #         if total < 1:
# #             raise ValueError(f"Empty video: {path}")

# #         ld = self.latent_downsample
# #         k  = self.num_start_frames
# #         s  = self.frame_stride
# #         T  = self.num_frames

# #         # frames available at stride s
# #         max_valid = 1 + (total - 1) // s

# #         # minimum real frames needed for latents: first + (k+1) groups (>= k+2 latents total)
# #         min_latents = k + 2
# #         min_valid_needed = 1 + (min_latents - 1) * ld
# #         if max_valid < min_valid_needed:
# #             raise RuntimeError("Clip too short for min latent length")

# #         # choose aligned L (≤ T and ≤ max_valid)
# #         target = min(T, max_valid)
# #         L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
# #         if L < min_valid_needed:
# #             L = min_valid_needed

# #         # random start that fits L at stride s
# #         max_start = total - (L - 1) * s
# #         start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

# #         # indices and frames
# #         idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
# #         frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3], uint8->float
# #         frames_valid = frames_valid / 127.5 - 1.0                     # [-1,1]

# #         # pad to fixed T
# #         if L < T:
# #             last = frames_valid[-1:].clone()
# #             frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
# #             frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
# #         else:
# #             frames = frames_valid
# #             frame_indices = idx_valid

# #         # masks & lengths
# #         frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
# #         latent_length = 1 + (L - 1) // ld
# #         video_length  = L

# #         # pack tensors
# #         img_tensor    = frames.permute(3, 0, 1, 2).contiguous()           # [C,T,H,W]
# #         anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous() # [3,1,H,W]

# #         # sanity
# #         assert img_tensor.shape[1] == T
# #         assert frame_mask.numel() == T
# #         assert frame_indices.shape[0] == T
# #         assert (L - 1) % ld == 0

# #         crop_coords = torch.zeros(8)
# #         description = it.get("description", "") or ""

# #         return Datapoint(
# #             pixel_values=img_tensor,
# #             condition={
# #                 "class_id": description,
# #                 "caption_idx": torch.tensor(0),
# #                 "crop_coords": crop_coords,
# #                 "anchor_frame": anchor_tensor,
# #                 "frame_mask": frame_mask,
# #                 "video_length": torch.tensor(video_length),
# #                 "latent_length": torch.tensor(latent_length),
# #                 "stride": torch.tensor(s),
# #                 "frame_indices": torch.from_numpy(frame_indices),
# #             },
# #         )

# #     def __getitem__(self, idx):
# #         for _ in range(self.max_retries):
# #             try:
# #                 dp = self._fetch_one()
# #                 if int(dp.condition["latent_length"]) >= (self.num_start_frames + 2):
# #                     return dp
# #             except Exception:
# #                 continue
# #         # last attempt raises on failure
# #         return self._fetch_one()


# # taichi_mp4_dataset.py
# import os
# import joblib
# import tqdm
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from decord import VideoReader, cpu
# from decord import bridge as decord_bridge
# decord_bridge.set_bridge('torch')  # return torch tensors from decord

# from engine.data_classes import Datapoint


# def _rgb_len_for_latents(n_latents: int, ld: int) -> int:
#     """
#     How many RGB frames are needed to produce `n_latents` temporal latents
#     when the temporal VAE rule is: first RGB frame -> 1 latent,
#     then every `ld` RGB frames -> 1 latent.
#     """
#     if n_latents <= 0:
#         return 0
#     return 1 + (n_latents - 1) * ld


# class TaichiDatasetFlowception(Dataset):
#     """
#     Video-only (MP4) dataset for Flowception-style training.

#     Expects an annotations directory (or single .pt file) with entries like:
#       {"filepath": "/abs/path/to/clip.mp4", "description": ""}

#     Returns:
#       Datapoint(
#         pixel_values: float32 in [-1,1], shape [C, T, H, W]
#         condition: {
#           "class_id": str,
#           "caption_idx": tensor(0),
#           "crop_coords": zeros(8),
#           "anchor_frame": [3,1,H,W],
#           "frame_mask": [T] bool,
#           "video_length": L (aligned, real frames only),
#           "latent_length": 1 + (L-1)//latent_downsample,
#           "stride": stride,
#           "frame_indices": [T] int64
#         }
#       )

#     Alignment rule: T = 1 + n * latent_downsample (ceil-aligned from requested num_frames).

#     NOTE: `num_start_latents` is interpreted in **latent** space.
#     """

#     def __init__(
#         self,
#         annotations_dir: str,          # dir of .pt shards OR a single .pt file
#         width: int,
#         height: int,
#         num_frames: int = 72,          # requested T (will be ceil aligned)
#         sampling_fps: float = 24.0,    # target sampling fps
#         native_fps: float = 24.0,      # nominal native fps for stride calc
#         # num_start_latents: int = 2,    # <-- latent-space warmup
#         latent_downsample: int = 8,    # ld
#         min_motion_score: float = 3.0, # (unused here, kept for parity)
#         pick_files: int | None = None, # (unused here)
#         max_retries: int = 20,
#         # --- backwards-compat alias (deprecated) ---
#         num_start_frames: int | None = None,
#     ):
#         # --- Handle shards vs single file robustly ---
#         entries = []
#         if os.path.isdir(annotations_dir):
#             shard_names = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
#             if not shard_names:
#                 raise FileNotFoundError(f"No .pt shards in {annotations_dir}")
#             for jp in tqdm.tqdm(shard_names, desc="Loading annotation shards"):
#                 data = joblib.load(os.path.join(annotations_dir, jp))
#                 entries.extend(data)
#         else:
#             # Single .pt file
#             data = joblib.load(annotations_dir)
#             entries.extend(data)

#         self.all_entries = entries
#         print(f"[TaichiMP4Dataset] items: {len(entries)}")

#         # ----- geometry / fps / stride -----
#         self.width  = int(width)
#         self.height = int(height)
#         self.sampling_fps = float(sampling_fps)
#         self.native_fps   = float(native_fps)

#         stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
#         self.frame_stride = max(1, stride)

#         # ----- flowception knobs -----
#         # Prefer the new latent-space arg; keep old name as alias if provided.
#         # if num_start_frames is not None:
#         # Treat legacy `num_start_frames` as LATENTS (correct semantics).
#         num_start_latents = int(num_start_frames)
#         self.num_start_latents  = int(num_start_latents)   # k in latent space
#         self.latent_downsample  = int(latent_downsample)   # ld
#         self.max_retries        = int(max_retries)

#         # ----- align T: 1 + n*ld -----
#         T_req = int(num_frames)
#         if T_req <= 1:
#             raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
#         ld = self.latent_downsample
#         rem = (T_req - 1) % ld
#         if rem != 0:
#             T_req += (ld - rem)
#             print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
#         self.num_frames = T_req

#     def __len__(self):
#         # streaming-style
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
#         k  = self.num_start_latents     # <-- latent-space warmup
#         s  = self.frame_stride
#         T  = self.num_frames

#         # frames available at stride s
#         max_valid = 1 + (total - 1) // s

#         # ---- LATENT-SPACE requirement -> RGB frames needed ----
#         # keep original safety margin: need at least (k + 2) latents total
#         min_latents = k + 2
#         min_valid_needed = _rgb_len_for_latents(min_latents, ld)
#         if max_valid < min_valid_needed:
#             raise RuntimeError("Clip too short for min latent length")

#         # choose aligned L (≤ T and ≤ max_valid)
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

#         # pad to fixed T
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
#             },
#         )

#     def __getitem__(self, idx):
#         for _ in range(self.max_retries):
#             try:
#                 dp = self._fetch_one()
#                 # latent-space check (>= k + 2 latents)
#                 if int(dp.condition["latent_length"]) >= (self.num_start_latents + 2):
#                     return dp
#             except Exception:
#                 continue
#         # last attempt raises on failure
#         return self._fetch_one()






# import os
# import joblib
# import tqdm
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from decord import VideoReader, cpu
# from decord import bridge as decord_bridge
# decord_bridge.set_bridge('torch')  # return torch tensors from decord

# from engine.data_classes import Datapoint


# def _rgb_len_for_latents(n_latents: int, ld: int) -> int:
#     """
#     How many RGB frames are needed to produce `n_latents` temporal latents
#     when the temporal VAE rule is: first RGB frame -> 1 latent,
#     then every `ld` RGB frames -> 1 latent.
#     """
#     if n_latents <= 0:
#         return 0
#     # return 1 + (n_latents - 1) * ld
#     return n_latents*ld


# class TaichiDatasetFlowception(Dataset):
#     """
#     Video-only (MP4) dataset for Flowception-style training.

#     Expects an annotations directory (or single .pt file) with entries like:
#       {"filepath": "/abs/path/to/clip.mp4", "description": ""}

#     Returns:
#       Datapoint(
#         pixel_values: float32 in [-1,1], shape [C, T, H, W]
#         condition: {
#           "class_id": str,
#           "caption_idx": tensor(0),
#           "crop_coords": zeros(8),
#           "anchor_frame": [3,1,H,W],
#           "frame_mask": [T] bool,
#           "video_length": L (aligned, real frames only),
#           "latent_length": 1 + (L-1)//latent_downsample,
#           "stride": stride,
#           "frame_indices": [T] int64
#         }
#       )

#     Alignment rule: T = 1 + n * latent_downsample (ceil-aligned from requested num_frames).

#     NOTE: `num_start_latents` is interpreted in **latent** space.
#     """

#     def __init__(
#         self,
#         annotations_dir: str,          # dir of .pt shards OR a single .pt file
#         width: int,
#         height: int,
#         num_frames: int = 72,          # requested T (will be ceil aligned)
#         sampling_fps: float = 24.0,    # target sampling fps
#         native_fps: float = 24.0,      # nominal native fps for stride calc
#         # num_start_latents: int = 2,    # warmup in LATENT space
#         latent_downsample: int = 8,    # ld
#         min_motion_score: float = 3.0, # (unused here; kept for parity)
#         pick_files: int | None = None, # (unused here)
#         max_retries: int = 20,
#         # --- backwards-compat alias (deprecated) ---
#         num_start_frames: int | None = None,
#     ):
#         # --- Load entries: directory of shards OR single .pt file ---
#         entries = []
#         if os.path.isdir(annotations_dir):
#             shard_names = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
#             if not shard_names:
#                 raise FileNotFoundError(f"No .pt shards in {annotations_dir}")
#             for jp in tqdm.tqdm(shard_names, desc="Loading annotation shards"):
#                 data = joblib.load(os.path.join(annotations_dir, jp))
#                 entries.extend(data)
#         else:
#             data = joblib.load(annotations_dir)
#             entries.extend(data)

#         self.all_entries = entries
#         print(f"[TaichiMP4Dataset] items: {len(entries)}")

#         # ----- geometry / fps / stride -----
#         self.width  = int(width)
#         self.height = int(height)
#         self.sampling_fps = float(sampling_fps)
#         self.native_fps   = float(native_fps)

#         stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
#         self.frame_stride = max(1, stride)

#         # ----- Flowception knobs -----
#         # Prefer latent-space arg; if legacy provided, treat as LATENTS.
#         # if num_start_frames is not None:
#         num_start_latents = int(num_start_frames)
#         self.num_start_latents  = int(num_start_latents)   # k in latent space
#         self.latent_downsample  = int(latent_downsample)   # ld
#         self.max_retries        = int(max_retries)

#         # ----- Align T to 1 + n*ld (ceil) -----
#         T_req = int(num_frames)
#         if T_req <= 1:
#             raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
#         ld = self.latent_downsample
#         rem = (T_req - 1) % ld
#         if rem != 0:
#             T_req += (ld - rem)
#             print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
#         self.num_frames = T_req

#     def __len__(self):
#         # streaming-style
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
#         k  = self.num_start_latents     # <-- latent-space warmup
#         s  = self.frame_stride
#         T  = self.num_frames

#         # frames available at stride s
#         max_valid = 1 + (total - 1) // s

#         # ---- LATENT-SPACE requirement -> RGB frames needed ----
#         # need at least k start latents (first frame -> 1 latent; then 1 per ld)
#         min_latents = k
#         min_valid_needed = _rgb_len_for_latents(min_latents, ld)  # 1 + (k-1)*ld
#         if max_valid < min_valid_needed:
#             raise RuntimeError("Clip too short for min latent length")

#         # choose aligned L (≤ T and ≤ max_valid); L must satisfy 1 + q*ld
#         target = min(T, max_valid)
#         L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
#         if L < min_valid_needed:
#             L = min_valid_needed  # still ≤ max_valid due to the check above

#         # random start that fits L at stride s
#         max_start = total - (L - 1) * s
#         start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

#         # indices and frames
#         idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
#         frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3], uint8->float
#         frames_valid = frames_valid / 127.5 - 1.0                     # [-1,1]

#         # pad to fixed T
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
#             },
#         )

#     def __getitem__(self, idx):
#         for _ in range(self.max_retries):
#             try:
#                 dp = self._fetch_one()
#                 # latent-space check (>= k start latents)
#                 if int(dp.condition["latent_length"]) >= self.num_start_latents:
#                     return dp
#             except Exception:
#                 continue
#         # last attempt raises on failure
#         return self._fetch_one()



# taichi_mp4_dataset.py
import os
import joblib
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from decord import VideoReader, cpu
from decord import bridge as decord_bridge
decord_bridge.set_bridge('torch')  # return torch tensors from decord

from engine.data_classes import Datapoint


def _rgb_len_for_latents(n_latents: int, ld: int) -> int:
    """
    RGB frames needed to yield `n_latents` temporal latents under:
      first RGB frame -> 1 latent; then every `ld` RGB frames -> +1 latent.
    """
    if n_latents <= 0:
        return 0
    return 1 + (n_latents - 1) * ld


class TaichiDatasetFlowception(Dataset):
    """
    Video-only (MP4) dataset for Flowception-style training.

    - Interprets `num_start_latents` and `num_context_latents` in **latent space**.
    - Guarantees each returned clip has at least `num_context_latents + num_start_latents` latents.
    - Aligns RGB length to VAE temporal rule: T = 1 + n * latent_downsample.

    Expects annotations (.pt file or dir of .pt shards) with entries like:
      {"filepath": "/abs/path/to/clip.mp4", "description": ""}

    Returns a Datapoint with:
      pixel_values: float32 in [-1,1], [C, T, H, W]
      condition: {
        "class_id": str,
        "caption_idx": tensor(0),
        "crop_coords": zeros(8),
        "anchor_frame": [3,1,H,W],            # first real RGB frame (for conditioners)
        "frame_mask": [T] bool,               # True for real frames, False for pad
        "video_length": L,                    # real RGB frames (aligned)
        "latent_length": 1 + (L-1)//ld,       # total latents represented in L RGB frames
        "stride": stride,
        "frame_indices": [T] int64,
        "num_context_latents": K,             # for convenience downstream
        "num_start_latents": k,
      }
    """

    def __init__(
        self,
        annotations_dir: str,          # dir of .pt shards OR a single .pt file
        width: int,
        height: int,
        num_frames: int = 72,          # requested RGB length; ceil-aligned to 1 + n*ld
        sampling_fps: float = 24.0,    # target sampling fps
        native_fps: float = 24.0,      # nominal native fps for stride calc
        num_start_latents: int = 2,    # k (latent-space warmup AFTER context)
        num_context_latents: int = 0,  # K (latent-space context at the very start)
        latent_downsample: int = 8,    # ld
        min_motion_score: float = 3.0, # (unused here; kept for parity)
        pick_files: int | None = None, # (unused here)
        max_retries: int = 20,
        # --- backwards-compat alias (deprecated) ---
        num_start_frames: int | None = None,  # treated as LATENTS if provided
    ):
        # Load entries from a dir of shards or a single .pt file
        entries = []
        if os.path.isdir(annotations_dir):
            shard_names = [p for p in os.listdir(annotations_dir) if p.endswith(".pt")]
            if not shard_names:
                raise FileNotFoundError(f"No .pt shards in {annotations_dir}")
            for jp in tqdm.tqdm(shard_names, desc="Loading annotation shards"):
                data = joblib.load(os.path.join(annotations_dir, jp))
                entries.extend(data)
        else:
            data = joblib.load(annotations_dir)
            entries.extend(data)

        if len(entries) == 0:
            raise RuntimeError(f"No entries loaded from {annotations_dir}")

        self.all_entries = entries
        print(f"[TaichiMP4Dataset] items: {len(entries)}")

        # ----- geometry / fps / stride -----
        self.width  = int(width)
        self.height = int(height)
        self.sampling_fps = float(sampling_fps)
        self.native_fps   = float(native_fps)

        stride = int(round(self.native_fps / max(1e-8, self.sampling_fps)))
        self.frame_stride = max(1, stride)

        # ----- Flowception knobs (latent-space semantics) -----
        if num_start_frames is not None:
            num_start_latents = int(num_start_frames)  # treat legacy arg as LATENTS

        self.num_start_latents  = int(max(0, num_start_latents))    # k
        self.num_context_latents = int(max(0, num_context_latents)) # K
        self.latent_downsample  = int(latent_downsample)            # ld
        self.max_retries        = int(max_retries)

        # ----- Align requested T to 1 + n*ld (ceil) -----
        T_req = int(num_frames)
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 (first + groups-of-ld).")
        ld = self.latent_downsample
        rem = (T_req - 1) % ld
        if rem != 0:
            T_req += (ld - rem)
            print(f"[info] aligning num_frames -> {T_req} (1 + n*{ld})")
        self.num_frames = T_req

        # For convenience: the **minimum latents required** by this dataset
        self.min_total_latents = self.num_context_latents + self.num_start_latents  # K + k

    def __len__(self):
        # streaming-style
        return 1_000_000

    def _fetch_one(self):
        it = self.all_entries[np.random.randint(0, len(self.all_entries))]
        path = it["filepath"]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        reader = VideoReader(
            path, num_threads=-1, ctx=cpu(0),
            width=self.width, height=self.height
        )
        total = len(reader)
        if total < 1:
            raise ValueError(f"Empty video: {path}")

        ld = self.latent_downsample
        K  = self.num_context_latents      # context latents to reserve at the start
        k  = self.num_start_latents        # start latents to sample AFTER context
        s  = self.frame_stride
        T  = self.num_frames

        # frames available at stride s from whole video
        max_valid = 1 + (total - 1) // s

        # ---- MINIMUM RGB length to ensure at least (K + k) latents total ----
        min_latents_total = max(1, K + k)  # need at least one latent overall
        min_valid_needed = _rgb_len_for_latents(min_latents_total, ld)
        if max_valid < min_valid_needed:
            raise RuntimeError(
                f"Clip too short for required latents: need ≥{min_latents_total} "
                f"({min_valid_needed} RGB frames @ stride {s})"
            )

        # choose aligned L (≤ T and ≤ max_valid), then ensure L ≥ min_valid_needed
        target = min(T, max_valid)
        L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
        if L < min_valid_needed:
            L = min_valid_needed  # still ≤ max_valid thanks to the check above
        # L is RGB frames, aligned so that (1 + (L-1)//ld) is integer latents

        # random start that fits L frames at stride s
        max_start = total - (L - 1) * s
        start = 0 if max_start <= 0 else int(np.random.randint(0, max_start))

        # indices and frames
        idx_valid = np.arange(start, start + L * s, s, dtype=np.int64)
        frames_valid = reader.get_batch(idx_valid).to(torch.float32)  # [L,H,W,3]
        frames_valid = frames_valid / 127.5 - 1.0                     # [-1,1]

        # pad/truncate to fixed T for batching (repeat last real frame if needed)
        if L < T:
            last = frames_valid[-1:].clone()
            frames = torch.cat([frames_valid, last.repeat(T - L, 1, 1, 1)], dim=0)  # [T,H,W,3]
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frames = frames_valid
            frame_indices = idx_valid

        # masks & lengths
        frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
        latent_length = 1 + (L - 1) // ld
        video_length  = L

        # pack tensors
        img_tensor    = frames.permute(3, 0, 1, 2).contiguous()           # [C,T,H,W]
        anchor_tensor = frames_valid[:1].permute(3, 0, 1, 2).contiguous() # [3,1,H,W]

        # sanity
        assert img_tensor.shape[1] == T
        assert frame_mask.numel() == T
        assert frame_indices.shape[0] == T
        assert (L - 1) % ld == 0

        crop_coords = torch.zeros(8)
        description = it.get("description", "") or ""

        return Datapoint(
            pixel_values=img_tensor,  # [-1,1], [C,T,H,W]
            condition={
                "class_id": description,
                "caption_idx": torch.tensor(0),
                "crop_coords": crop_coords,
                "anchor_frame": anchor_tensor,                   # first real RGB frame
                "frame_mask": frame_mask,                        # [T] (True for first L)
                "video_length": torch.tensor(video_length),      # L (RGB)
                "latent_length": torch.tensor(latent_length),    # 1 + (L-1)//ld
                "stride": torch.tensor(s),
                "frame_indices": torch.from_numpy(frame_indices),
                "num_context_latents": torch.tensor(K),
                "num_start_latents": torch.tensor(k),
            },
        )

    def __getitem__(self, idx):
        # keep sampling until we get a datapoint with ≥ (K + k) latents
        K = self.num_context_latents
        k = self.num_start_latents
        need_latents = max(1, K + k)

        for _ in range(self.max_retries):
            try:
                dp = self._fetch_one()
                if int(dp.condition["latent_length"]) >= need_latents:
                    return dp
            except Exception:
                continue
        # last attempt raises on failure
        return self._fetch_one()

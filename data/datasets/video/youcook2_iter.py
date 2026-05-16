import os, json, random, math
import joblib
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from collections import OrderedDict
from decord import VideoReader, cpu, gpu
from decord import bridge as decord_bridge

decord_bridge.set_bridge("torch")

from engine.data_classes import Datapoint


class YouCook2IterFlowception(IterableDataset):
    """
    Iterates efficiently: for each (sharded) video, yield ALL valid segments of that
    video (one sample per segment) before moving to the next video.

    An entry looks like:
      {"filename": "K6Uk5vNi1_Q.mp4",
       "caption": [{"start": 0.0, "end": 2.87, "caption": "..."},
                   {"start": 2.87, "end": 5.14, "caption": "..."},
                   ...]}

    Features
    --------
    - Per-worker sharding across DDP ranks × DataLoader workers.
    - LRU cache of VideoReader to avoid reopen cost.
    - Optional GPU decode (NVDEC) -> returns CUDA tensors directly.
    - Repeat-last padding, mask, anchor inside-segment, latent_length.
    - Deterministic or per-epoch shuffling via set_epoch(seed).

    One pass over this dataset yields exactly one sample per valid segment
    for this worker's shard. Wrap it with `cycle()` if you want infinite epochs.
    """

    def __init__(
        self,
        annotations,  # list[dict] or path to .pt/.pkl/.joblib/.json
        vid_root: str,
        width: int,
        height: int,
        num_frames: int = 72,
        sampling_fps: float = 24.0,
        native_fps: float = 24.0,
        num_start_frames: int = 2,
        latent_downsample: int = 8,
        # segment filtering (ensure enough frames for Flowception)
        drop_too_short: bool = True,
        # sharding & ordering
        shard_style: str = "interleaved",  # or "contiguous"
        shuffle_videos: bool = False,
        shuffle_segments: bool = False,
        base_seed: int = 123,
        # decoding performance
        reader_cache_size: int = 12,
        vr_num_threads: int = -1,
        use_gpu_decode: bool = False,
        gpu_device_id: int | None = None,
    ):
        # Load annotations
        if isinstance(annotations, str):
            if annotations.endswith((".pt", ".pkl", ".joblib")):
                annotations = joblib.load(annotations)
            elif annotations.endswith(".json"):
                with open(annotations, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            else:
                raise ValueError(f"Unsupported annotations file: {annotations}")

        self.vid_root = vid_root
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.sampling_fps = float(sampling_fps)
        self.native_fps = float(native_fps)
        self.num_start_frames = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)

        stride = int(round(self.native_fps / self.sampling_fps))
        self.frame_stride = max(1, stride)

        # Pre-filter segments if desired (saves retries during iteration)
        if drop_too_short:
            ld, k, s, nfps = (
                int(self.latent_downsample),
                int(self.num_start_frames),
                int(self.frame_stride),
                float(self.native_fps),
            )
            # need at least ld*k + 1 latent samples -> raw frames >= s*(ld*k)+1
            min_seg_frames = s * (ld * k) + 1
            min_seconds = min_seg_frames / nfps
        else:
            min_seconds = 0.0

        cleaned = []
        kept = dropped = 0
        for it in annotations:
            segs_all = it.get("caption", []) or []
            segs = []
            for sgm in segs_all:
                st = sgm.get("start", None)
                en = sgm.get("end", None)
                if isinstance(st, (int, float)) and isinstance(en, (int, float)) and en > st:
                    if (en - st) + 1e-6 >= min_seconds:
                        segs.append(sgm)
                        kept += 1
                    else:
                        dropped += 1
                else:
                    dropped += 1
            if segs:
                cleaned.append({"filename": it["filename"], "segments": segs})
        self.entries = cleaned
        print(
            f"[YouCook2Iter] videos with ≥1 valid segment: {len(self.entries)} | "
            f"segments kept: {kept}, dropped: {dropped}"
        )

        # Sharding & ordering
        self._shard_style = shard_style
        self.shuffle_videos = bool(shuffle_videos)
        self.shuffle_segments = bool(shuffle_segments)
        self.base_seed = int(base_seed)
        self._epoch = 0

        # Decode knobs
        self._reader_cache_size = int(reader_cache_size)
        self._vr_num_threads = -1  # int(vr_num_threads)
        self._use_gpu_decode = bool(use_gpu_decode)
        self._gpu_device_id = gpu_device_id

        # Late-initialized per-worker state
        self._vr_cache = None  # OrderedDict[str, VideoReader]
        self._my_videos = None  # list of (filename, segments)
        self._rng = None  # random.Random per worker/epoch

    # ---------- public controls ----------

    def set_epoch(self, epoch: int):
        """Call this each epoch to reshuffle (if shuffle_* is True)."""
        self._epoch = int(epoch)

    # ---------- internal helpers ----------

    def _ensure_worker_state(self):
        if self._vr_cache is not None:
            return
        self._vr_cache = OrderedDict()

        wi = get_worker_info()
        wid, wnum = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()

        gnum = world * wnum
        gid = rank * wnum + wid

        # shard videos
        vids = self.entries
        if self._shard_style == "contiguous":
            start = (len(vids) * gid) // gnum
            end = (len(vids) * (gid + 1)) // gnum
            my_videos = vids[start:end]
        else:
            my_videos = vids[gid::gnum]

        # RNG per worker/epoch for deterministic shuffle if requested
        seed = (self.base_seed + self._epoch) * 1000003 + gid
        self._rng = random.Random(seed)

        if self.shuffle_videos:
            my_videos = my_videos.copy()
            self._rng.shuffle(my_videos)

        self._my_videos = my_videos

    def _ctx_and_device(self):
        if not self._use_gpu_decode:
            return cpu(0), None
        dev = self._gpu_device_id
        if dev is None:
            dev = 0
            if torch.cuda.is_available():
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    dev = torch.distributed.get_rank() % torch.cuda.device_count()
                else:
                    dev = 0
        return gpu(dev), dev

    def _get_reader(self, path: str) -> VideoReader:
        cache = self._vr_cache
        vr = cache.get(path) if cache is not None else None
        if vr is not None:
            cache.move_to_end(path)
            return vr
        ctx, _ = self._ctx_and_device()
        vr = VideoReader(
            path,
            num_threads=-1,  # self._vr_num_threads,
            ctx=cpu(0),  # ctx,
            width=self.width,
            height=self.height,
        )
        cache[path] = vr
        if len(cache) > self._reader_cache_size:
            cache.popitem(last=False)
        return vr

    def _segment_frame_window(self, total_frames: int, start_s: float, end_s: float):
        # [start, end) in seconds -> inclusive frame indices
        start_idx = int(round(start_s * self.native_fps))
        end_idx = int(round(end_s * self.native_fps)) - 1
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx, min(end_idx, total_frames - 1))
        return start_idx, end_idx

    # ---------- core iterator ----------

    def __iter__(self):
        self._ensure_worker_state()
        device_ctx, dev_id = self._ctx_and_device()  # not used directly but establishes device choice

        # Walk videos in this worker’s shard
        for v in self._my_videos:
            filename = v["filename"]
            segs = v["segments"]
            seg_indices = list(range(len(segs)))
            if self.shuffle_segments:
                self._rng.shuffle(seg_indices)

            # Open/cached reader once per video, reuse for all segments
            path = os.path.join(self.vid_root, filename)
            if not os.path.isfile(path):
                continue
            try:
                reader = self._get_reader(path)
                total = len(reader)
                if total <= 0:
                    continue
            except Exception:
                continue

            # Iterate all segments for this video
            for si in seg_indices:
                seg = segs[si]
                try:
                    # ---- sample clip inside this segment ----
                    seg_start_idx, seg_end_idx = self._segment_frame_window(
                        total, float(seg["start"]), float(seg["end"])
                    )
                    seg_len = seg_end_idx - seg_start_idx + 1
                    if seg_len <= 0:
                        continue

                    T = self.num_frames
                    s = self.frame_stride

                    max_valid = 1 + (seg_len - 1) // s
                    valid_len = min(T, max_valid)

                    # choose a start so that start + (valid_len-1)*s <= seg_end_idx
                    start_low = seg_start_idx
                    start_high = seg_end_idx - (valid_len - 1) * s
                    if start_high < start_low:
                        start_high = start_low
                    start = int(self._rng.randint(start_low, start_high))

                    idx_valid = np.arange(start, start + valid_len * s, s, dtype=np.int64)

                    frames_valid = reader.get_batch(idx_valid)  # [L,H,W,3] torch uint8
                    anchor = frames_valid[:1]  # [1,H,W,3]

                    # pad by repeating last
                    if valid_len < T:
                        pad = frames_valid[-1:].expand(T - valid_len, -1, -1, -1)
                        frames_u8 = torch.cat([frames_valid, pad], dim=0)  # [T,H,W,3]
                    else:
                        frames_u8 = frames_valid

                    # To [-1,1], [C,T,H,W]
                    img_tensor = frames_u8.permute(3, 0, 1, 2).to(torch.float32).div_(127.5).sub_(1.0)
                    anchor_tensor = anchor.permute(3, 0, 1, 2).to(torch.float32).div_(127.5).sub_(1.0)

                    frame_mask = torch.zeros(T, dtype=torch.bool, device=img_tensor.device)
                    frame_mask[:valid_len] = True
                    if valid_len < T:
                        pad_idx = np.repeat(idx_valid[-1:], T - valid_len)
                        frame_indices = np.concatenate([idx_valid, pad_idx]).astype(np.int64)
                    else:
                        frame_indices = idx_valid

                    latent_len = (valid_len + self.latent_downsample - 1) // self.latent_downsample
                    # Gate for Flowception (enough latent steps to pick k starts when skipping index 0)
                    if latent_len < self.num_start_frames + 2:
                        continue

                    crop_coords = torch.zeros(8, device=img_tensor.device)

                    yield Datapoint(
                        pixel_values=img_tensor,
                        condition={
                            "class_id": seg.get("caption", "") or "",
                            "caption_idx": torch.tensor(0, device=img_tensor.device),
                            "crop_coords": crop_coords,
                            "anchor_frame": anchor_tensor,
                            "frame_mask": frame_mask,
                            "video_length": torch.tensor(valid_len, device=img_tensor.device),
                            "latent_length": torch.tensor(latent_len, device=img_tensor.device),
                            "stride": torch.tensor(self.frame_stride, device=img_tensor.device),
                            "frame_indices": torch.from_numpy(frame_indices),
                            # "segment_start_s": torch.tensor(float(seg["start"]), device=img_tensor.device),
                            # "segment_end_s": torch.tensor(float(seg["end"]), device=img_tensor.device),
                            # "video_filename": filename,
                            # "segment_index": torch.tensor(int(si), device=img_tensor.device),
                        },
                    )

                except Exception:
                    # skip bad segments, continue through the rest
                    continue

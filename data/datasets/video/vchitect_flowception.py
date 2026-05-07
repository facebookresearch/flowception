# vchitect_tar_flowception_fps.py
import io, os, json, sqlite3, numpy as np, torch, tqdm
from pathlib import Path
from torch.utils.data import Dataset
import av  # pip install av
from torchvision.transforms.functional import resize
from engine.data_classes import Datapoint  # your class

# add near your imports
import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')  # returns torch tensors instead of NDArray

def _decode_with_decord_from_bytes(
    data: bytes,
    width: int,
    height: int,
    T: int,                   # requested aligned length
    ld: int,                  # latent downsample
    k: int,                   # num_start_frames
    sampling_fps: float,
    default_native_fps: float,
    prefer_gpu: bool = False,
):
    bio = io.BytesIO(data)  # keep a ref alive while vr exists
    ctx = gpu(0) if (prefer_gpu and decord.__dict__.get('_cuda_enabled_', False)) else cpu(0)

    # decode at target size directly
    vr = VideoReader(bio, ctx=ctx, width=width, height=height, num_threads=0)

    # native FPS (best-effort)
    try:
        native_fps = float(vr.get_avg_fps())
        if not (native_fps > 0):
            native_fps = float(default_native_fps)
    except Exception:
        native_fps = float(default_native_fps)

    s = max(1, int(round(native_fps / float(sampling_fps))))  # stride
    total = len(vr)

    # how many valid frames exist at this stride?
    max_valid = 1 + (total - 1) // s

    # minimum valid frames = first + groups for (k + 2) latents
    min_latents = k + 2
    min_valid_needed = 1 + (min_latents - 1) * ld
    if max_valid < min_valid_needed:
        raise RuntimeError("Video too short for min latent length")

    # choose L aligned to 1 + n*ld, clipped by T and max_valid
    target = min(T, max_valid)
    L = 1 if target <= 1 else 1 + ((target - 1) // ld) * ld
    if L < min_valid_needed:
        L = min_valid_needed

    # pick a start that fits L at stride s
    max_start = max(1, total - (L - 1) * s)
    start = int(np.random.randint(0, max_start)) if max_start > 1 else 0

    # exact frame indices and safe clamp
    idx = np.arange(start, start + L * s, s, dtype=np.int64)
    idx[idx >= total] = total - 1

    # efficient random access in one go, returns [L, H, W, 3] uint8 (torch tensor because of the bridge)
    frames = vr.get_batch(idx.tolist())
    # to [C, T, H, W]
    frames = frames.permute(3, 0, 1, 2).contiguous()  # [3, L, H, W], uint8

    return frames, s, L, idx


# add at top
from collections import OrderedDict

class _LocLRU(OrderedDict):
    def __init__(self, capacity=200_000):
        super().__init__(); self.capacity = capacity
    def get(self, k, default=None):
        v = super().get(k, default)
        if v is not None: self.move_to_end(k)
        return v
    def put(self, k, v):
        self[k] = v; self.move_to_end(k)
        if len(self) > self.capacity: self.popitem(last=False)


# ------------------ utilities ------------------

def _chunks(lst, n=900):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class _FDCache:
    """Small LRU cache of open tar FDs (per worker)."""
    def __init__(self, capacity=32):
        self.capacity = capacity
        self._order, self._fds = [], {}
    def open(self, path: str):
        if path in self._fds:
            self._order.remove(path); self._order.append(path)
            return self._fds[path]
        fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        self._order.append(path)
        if len(self._order) > self.capacity:
            old = self._order.pop(0)
            os.close(self._fds.pop(old, None))
        return fd
    def close_all(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear(); self._order.clear()

def _norm_neg1_pos1_uint8(x: torch.Tensor) -> torch.Tensor:
    # x is uint8 or float in [0,255] -> [-1,1]
    return x.to(torch.float32).div_(127.5).sub_(1.0)

def _fps_from_stream(vstream, default_fps: float = 24.0) -> float:
    """Best-effort FPS from metadata; falls back to default_fps."""
    try:
        r = vstream.average_rate
        if r:
            f = float(r)
            if f > 0:
                return f
    except Exception:
        pass
    try:
        frames = int(vstream.frames or 0)
        if frames > 0 and vstream.duration and vstream.time_base:
            seconds = float(vstream.duration * vstream.time_base)
            if seconds > 0:
                f = frames / seconds
                if f > 0:
                    return f
    except Exception:
        pass
    return float(default_fps)

def _estimate_total_frames(vstream, fps_fallback: float) -> int:
    """Len-like estimate using direct metadata or duration*fallback_fps."""
    try:
        if vstream.frames and int(vstream.frames) > 0:
            return int(vstream.frames)
    except Exception:
        pass
    try:
        if vstream.duration and vstream.time_base:
            secs = float(vstream.duration * vstream.time_base)
            if secs > 0:
                return max(1, int(round(secs * fps_fallback)))
    except Exception:
        pass
    # last resort: ~10 seconds of video at fallback fps
    return max(1, int(round(10 * fps_fallback)))

def _seek_to_frame(container, vstream, start_idx: int, fps: float):
    """Approximate seek to a given frame index using timestamps."""
    try:
        if fps and vstream.time_base:
            target_sec = start_idx / fps
            pts = int(target_sec / float(vstream.time_base))
            container.seek(pts, any_frame=False, backward=True, stream=vstream)
    except Exception:
        pass

# ------------------ main dataset ------------------

class VChitectTarFlowception(Dataset):
    def __init__(
        self,
        annotations_json: str,
        index_db: str,
        width: int,
        height: int,
        num_frames: int = 72,             # will ceil to 1 + n*ld
        min_motion_score: float | None = None,
        sampling_fps: float = 24.0,        # desired sampling rate
        default_native_fps: float = 24.0,  # fallback when metadata missing
        num_start_frames: int = 2,         # k
        latent_downsample: int = 8,        # ld
        max_retries: int = 20,
        drop_missing: bool = False,
        preload_locations: bool = False,
    ):
        # load & optional motion filter
        ann = json.loads(Path(annotations_json).read_text())

        self.ann = ann

        self.width  = int(width)
        self.height = int(height)
        self.sampling_fps      = float(sampling_fps)
        self.default_native_fps = float(default_native_fps)
        self.num_start_frames  = int(num_start_frames)
        self.latent_downsample = int(latent_downsample)
        self.max_retries = int(max_retries)
        self.db_path = str(index_db)

        # Align T to rule: T = 1 + n*ld (ceil)
        T_req = int(num_frames)
        ld = self.latent_downsample
        if T_req <= 1:
            raise ValueError("num_frames must be >= 2 for (first + groups-of-ld) rule")
        rem = (T_req - 1) % ld
        if rem != 0:
            T_req += (ld - rem)
        self.num_frames = T_req

        # SQLite: keep only present basenames and (optionally) preload locations
        names = [os.path.basename(it["video"]) for it in self.ann]
        self._loc = {}  # basename -> (tar, offset, size)
        print("0")

        self._fdcache = None

        # NEW: per-worker DB state (lazy-initialized inside the worker)
        self._conn = None
        self._cur = None
        self._sql = "SELECT tar, offset, size FROM vids WHERE basename=? LIMIT 1"

        # keep preload map if you use it; add a small on-demand LRU too
        self._preloaded = {}     # filled only if preload_locations=True
        self._lru = _LocLRU(200_000)

        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as c:
            c.execute("PRAGMA query_only=ON")
            if drop_missing:
                found = set()
                for chunk in _chunks(names):
                    q = ",".join("?" * len(chunk))
                    for (nm,) in c.execute(f"SELECT basename FROM vids WHERE basename IN ({q})", chunk):
                        found.add(nm)
                self.ann = [it for it in self.ann if os.path.basename(it["video"]) in found]
                names = [os.path.basename(it["video"]) for it in self.ann]

            if preload_locations and names:
                for chunk in _chunks(names):
                    q = ",".join("?" * len(chunk))
                    for basename, tar, offset, size in c.execute(
                        f"SELECT basename, tar, offset, size FROM vids WHERE basename IN ({q})", chunk
                    ):
                        self._preloaded[basename] = (tar, offset, size)


    def __len__(self):
        return 1_000_000  # streaming-style

    # --- tar I/O ---
    def _get_fd(self, tar_path: str):
        if self._fdcache is None:
            self._fdcache = _FDCache(capacity=32)
        return self._fdcache.open(tar_path)

    def _lookup(self, basename: str):
        loc = self._loc.get(basename)
        if loc is not None:
            return loc
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as c:
            row = c.execute("SELECT tar, offset, size FROM vids WHERE basename=? LIMIT 1", (basename,)).fetchone()
        if not row:
            raise FileNotFoundError(basename)
        return row

    def _read_member_bytes(self, tar_path: str, offset: int, size: int) -> bytes:
        fd = self._get_fd(tar_path)
        try:
            return os.pread(fd, size, offset)  # POSIX fast-path
        except AttributeError:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, size)
        
    # -- NEW: open one connection per worker on first use --
    def _conn_open(self):
        if self._conn is None:
            # Read-only, immutable = fewer fs calls; allow reuse in worker threads if any
            uri = f"file:{self.db_path}?mode=ro&immutable=1"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
            self._cur = self._conn.cursor()
            # helpful PRAGMAs for read perf (tune to your RAM)
            self._cur.execute("PRAGMA query_only=ON")
            self._cur.execute("PRAGMA temp_store=MEMORY")
            self._cur.execute("PRAGMA mmap_size=268435456")    # 256 MB
            self._cur.execute("PRAGMA cache_size=-200000")     # ~200 MB page cache
        return self._cur

    def _lookup(self, basename: str):
        # 1) preload hit?
        loc = self._preloaded.get(basename)
        if loc is not None:
            return loc
        # 2) LRU hit?
        loc = self._lru.get(basename)
        if loc is not None:
            return loc
        # 3) single prepared query on persistent connection
        cur = self._conn_open()
        row = cur.execute(self._sql, (basename,)).fetchone()
        if not row:
            raise FileNotFoundError(basename)
        self._lru.put(basename, row)
        return row

    def __del__(self):
        try:
            if self._fdcache: self._fdcache.close_all()
            if self._cur: self._cur.close()
            if self._conn: self._conn.close()
        except Exception:
            pass

    # --- sample fetch ---
    def _fetch_one(self):
        j = np.random.randint(0, len(self.ann))
        item = self.ann[j]
        basename = os.path.basename(item["video"])
        tar_path, offset, size = self._lookup(basename)
        data = self._read_member_bytes(tar_path, offset, size)

        ld, k, T = self.latent_downsample, self.num_start_frames, self.num_frames

        try:
            frames_u8, s, L, idx_valid = _decode_with_decord_from_bytes(
                data,
                width=self.width, height=self.height,
                T=T, ld=ld, k=k,
                sampling_fps=self.sampling_fps,
                default_native_fps=self.default_native_fps,
                prefer_gpu=False,  # set True if you built decord with CUDA
            )
        except Exception:
            # optional fallback to your PyAV path if decord fails on a weird file
            raise

        # normalize to [-1, 1], pad to T, and build metadata (unchanged from your version)
        pixel_values = _norm_neg1_pos1_uint8(frames_u8)  # [3,L,H,W] -> float [-1,1]

        if L < T:
            pad = pixel_values[:, -1:].expand(3, T - L, self.height, self.width).contiguous()
            pixel_values = torch.cat([pixel_values, pad], dim=1)
            frame_indices = np.concatenate([idx_valid, np.repeat(idx_valid[-1], T - L)])
        else:
            frame_indices = idx_valid

        frame_mask    = torch.zeros(T, dtype=torch.bool); frame_mask[:L] = True
        latent_length = torch.tensor(1 + (L - 1) // ld, dtype=torch.long)
        video_length  = torch.tensor(L, dtype=torch.long)
        stride_tensor = torch.tensor(s, dtype=torch.long)
        anchor_frame  = pixel_values[:, :1]  # [3,1,H,W]

        cls = item.get("description") or item.get("text") or ""
        return Datapoint(
            pixel_values=pixel_values,  # [C,T,H,W]
            condition={
                "class_id": cls,
                "caption_idx": torch.tensor(0, dtype=torch.long),
                "crop_coords": torch.zeros(8, dtype=torch.float32),
                "anchor_frame": anchor_frame,
                "frame_mask": frame_mask,
                "video_length": video_length,
                "latent_length": latent_length,
                "stride": stride_tensor,
                "frame_indices": torch.from_numpy(frame_indices.astype(np.int64)),
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

    def __del__(self):
        try:
            if self._fdcache: self._fdcache.close_all()
        except Exception:
            pass

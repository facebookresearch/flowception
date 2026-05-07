import torch


def sample_start_frames(M1: torch.Tensor, k: int, skip_first: int = 1, gen: torch.Generator | None = None):
    """
    Vectorized start-frame sampler.
    Always includes:
      - frame 0
      - last valid frame (vid_len-1)
    Then samples remaining frames uniformly from valid candidates.

    Args:
        M1: [B,L] bool/0-1 valid-frame mask (assumes valid frames are prefix length vid_len)
        k:  number of start frames per sample
        skip_first: exclude indices < skip_first from RANDOM picks (forced 0 still included)
        gen: optional torch.Generator for reproducibility
    Returns:
        start: [B,L] bool, exactly k True (if possible; otherwise as many as exist)
    """
    device = M1.device
    B, L = M1.shape
    M1b = M1.bool()

    # infer lengths and last index
    vid_len = M1b.long().sum(dim=1).clamp_min(1)  # [B]
    last = (vid_len - 1).clamp(0, L - 1)  # [B]

    start = torch.zeros((B, L), dtype=torch.bool, device=device)

    # force first and last only when skip_first == 0 (caller will add them otherwise)
    if skip_first == 0:
        if k >= 1:
            start[:, 0] = True
        if k >= 2:
            start[torch.arange(B, device=device), last] = True

    forced = start.sum(dim=1)  # [B]
    # candidates: valid, not already chosen, and not in first skip_first indices (for random picks)
    idx = torch.arange(L, device=device)[None, :]  # [1,L]
    valid = idx < vid_len[:, None]  # [B,L]
    cand = valid & (~start)
    if skip_first > 0:
        cand &= idx >= skip_first

    cand_count = cand.sum(dim=1)  # [B]
    rem = (k - forced).clamp_min(0)
    rem = torch.minimum(rem, cand_count)  # can't pick more than available
    max_rem = int(rem.max().item())
    if max_rem == 0:
        return start

    # random scores; invalidate non-candidates with -inf so topk ignores them
    scores = torch.rand((B, L), device=device, generator=gen)
    scores = scores.masked_fill(~cand, float("-inf"))

    # take top max_rem indices per row
    top_idx = scores.topk(k=max_rem, dim=1).indices  # [B,max_rem]

    # keep only first rem[b] of those top indices
    keep = torch.arange(max_rem, device=device)[None, :] < rem[:, None]  # [B,max_rem]

    # scatter selected indices into a mask and OR into start
    picked = torch.zeros((B, L), dtype=torch.bool, device=device)
    picked.scatter_(1, top_idx, keep)
    start |= picked
    return start


def sample_start_frames_interp(
    mask: torch.Tensor,
    k: int,
    skip_first: int = 0,
    *,
    generator: torch.Generator | None = None,
):
    """
    mask: (B, L) bool, each row == [True...True, False...False]
    k: number to select per row (can be 0)
    skip_first: number of leading True positions to skip when sampling
    """
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be (B, L) boolean")
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    B, L = mask.shape
    lengths = mask.sum(dim=1)  # number of True at row start

    # Early-out for k == 0: valid and common, return all False
    if k == 0:
        return torch.zeros_like(mask)

    # Eligible positions per row = {skip_first, ..., lengths[b]-1}
    ar = torch.arange(L, device=mask.device).expand(B, L)
    eligible = (ar >= skip_first) & (ar < lengths.unsqueeze(1))
    avail = eligible.sum(dim=1)

    # If any row has fewer than k eligible positions, fail clearly
    if (avail < k).any():
        bad = (avail < k).nonzero(as_tuple=False).squeeze(1).tolist()
        details = [(int(lengths[b].item()), int(avail[b].item())) for b in bad]
        raise ValueError(
            f"Rows {bad} don't have enough eligible positions for k={k} "
            f"(row lengths & avail after skip_first={skip_first}: {details})."
        )

    # Vectorized sampling where we can, plus a deterministic fast-path
    selected = torch.zeros_like(mask)

    # Rows where avail == k → take all eligible deterministically
    rows_all = (avail == k).nonzero(as_tuple=False).squeeze(1)
    if rows_all.numel():
        selected[rows_all] = eligible[rows_all]

    # Rows where avail > k → sample k distinct indices
    rows_sample = (avail > k).nonzero(as_tuple=False).squeeze(1)
    if rows_sample.numel():
        weights = eligible[rows_sample].float()  # 1.0 for eligible, 0.0 else
        idx = torch.multinomial(weights, num_samples=k, replacement=False, generator=generator)
        # Use index_put_ to avoid the gotcha with scatter_ on sliced tensors
        row_exp = rows_sample[:, None].expand_as(idx)
        selected[row_exp, idx] = True

    return selected


def _block_mask_from_frame_mask(frame_mask: torch.Tensor, D_full: int, group: int) -> torch.Tensor:
    """
    frame_mask: [B, T] (bool/0-1), True for real frames
    D_full:     latent depth before skipping
    group:      VAE block size (first block uses frame 0; next blocks use 1+k*group)

    Returns:
      M_blocks: [B, D_full] bool
    """
    B, T = frame_mask.shape
    device = frame_mask.device

    # indices = [0] + [1 + k*group] for k in range(D_full-1), clamped to T-1
    if D_full <= 0:
        raise ValueError(f"D_full must be > 0, got {D_full}")
    idx0 = torch.tensor([0], device=device, dtype=torch.long)
    if D_full > 1:
        tail = 1 + torch.arange(D_full - 1, device=device, dtype=torch.long) * group
        tail = torch.clamp(tail, max=T - 1)
        idx = torch.cat([idx0, tail], dim=0)  # [D_full]
    else:
        idx = idx0  # [1]

    # gather frame_mask at those indices to make per-block mask
    M_blocks = frame_mask.gather(1, idx.unsqueeze(0).expand(B, -1)).to(torch.bool)  # [B, D_full]
    return M_blocks


def pick_latents_after_skip(
    latents: torch.Tensor,  # [B, C, D, H, W]
    frame_mask: torch.Tensor,  # [B, T] (bool or 0/1)
    *,
    group: int = 8,
    num_start_frames: int = 2,
    skip="random",  # ← 'none', 'first', 'last', 'random', True (='random'), False (='none')
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      lat_sel: [B, C, D', H, W]     (after skipping first or last, or keeping all)
      M_sel:   [B, D'] bool         (block mask aligned with lat_sel)

    Args:
        skip: Controls which frames to keep:
            - 'none' or False: Keep all frames
            - 'first': Drop first frame (use frames 1:)
            - 'last': Drop last frame (use frames :-1)
            - 'random' or True: Randomly choose between 'first' and 'last'
    """
    assert latents.ndim == 5, f"latents must be [B,C,D,H,W], got {latents.shape}"
    B, C, D_full, H, W = latents.shape

    fm = frame_mask.to(torch.bool)
    M_blocks = _block_mask_from_frame_mask(fm, D_full, group)  # [B, D_full]

    # Normalize skip parameter
    if skip is False:
        skip = "none"
    elif skip is True:
        skip = "random"

    # Handle 'none' case - keep all frames
    if skip == "none":
        lat_sel, M_sel = latents, M_blocks
    else:
        # Prepare both options
        lat_after_first, M_after_first = latents[:, :, 1:], M_blocks[:, 1:]
        lat_before_last, M_before_last = latents[:, :, :-1], M_blocks[:, :-1]

        # Select based on skip mode
        if skip == "first":
            lat_sel, M_sel = lat_after_first, M_after_first
        elif skip == "last":
            lat_sel, M_sel = lat_before_last, M_before_last
        elif skip == "random":
            # 50/50 random choice
            if torch.rand(1, device=latents.device).item() > 0.5:
                lat_sel, M_sel = lat_after_first, M_after_first
            else:
                lat_sel, M_sel = lat_before_last, M_before_last
        else:
            raise ValueError(f"Invalid skip mode: {skip}. Must be 'none', 'first', 'last', or 'random'")

        # If any row would have < num_start_frames after skipping, force [:, :, 1:]
        need = int(num_start_frames)
        if (M_sel.sum(dim=1) < need).any():
            lat_sel, M_sel = lat_after_first, M_after_first

    # Minimal padding if depth still too small (rare but safe)
    # Only needed when we've skipped frames
    if skip != "none":
        need = int(num_start_frames)
        D_sel = lat_sel.shape[2]
        if D_sel < need:
            rep = lat_sel[:, :, -1:].repeat(1, 1, need - D_sel, 1, 1)
            lat_sel = torch.cat([lat_sel, rep], dim=2)
            pad = torch.ones(B, need - D_sel, dtype=torch.bool, device=latents.device)
            M_sel = torch.cat([M_sel, pad], dim=1)

    return lat_sel, M_sel


# helpers/masks.py
import torch


def build_interpolation_mask(T_lat, min_keys=2, max_keys=6, include_first=True, include_last=True):
    """
    Returns a bool tensor [T_lat] with 1's at chosen keyframes (known frames), 0 elsewhere.
    """
    assert T_lat >= 2
    K = int(torch.randint(min_keys, max_keys + 1, (1,)).item())
    K = min(K, T_lat)

    all_idx = torch.arange(T_lat)
    fixed = []
    if include_first:
        fixed.append(0)
    if include_last:
        fixed.append(T_lat - 1)

    remaining = [i for i in all_idx.tolist() if i not in fixed]
    if len(remaining) > 0 and (K - len(fixed)) > 0:
        picks = torch.tensor(remaining)[torch.randperm(len(remaining))[: K - len(fixed)]].tolist()
    else:
        picks = []

    idx = sorted(set(fixed + picks))
    mask = torch.zeros(T_lat, dtype=torch.bool)
    mask[idx] = True
    return mask, torch.tensor(idx, dtype=torch.long)

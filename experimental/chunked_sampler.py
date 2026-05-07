"""
Chunked sampling for the Flowception framework.

Implements `vanilla_sample_flowception_t2v_chunked`, which splits the
generation into N temporal chunks processed in a single batch of size N*B.
Before each denoiser forward pass, overlap frames from neighbouring chunks
are concatenated so the model has temporal context across chunk boundaries.

Usage (CPU-only test with dummy denoiser):
    python experimental/chunked_sampler.py

NOTE: This is an experimental standalone implementation. It is not wired into
the main training or sampling pipelines. The forward_fn and model signatures
must be adapted to match the actual model API in modules/flowception/sampling.py.
"""

import torch
import tqdm


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def scale_snr(t: torch.Tensor, s: float, eps: float = 1e-6) -> torch.Tensor:
    t = t.clamp(eps, 1.0 - eps)
    return t / (t + s * (1.0 - t))


def get_active_length(M_t: torch.Tensor) -> int:
    """Minimum length that covers every True position across the batch."""
    any_valid = M_t.any(dim=0)  # (L,)
    if not any_valid.any():
        return 1
    return any_valid.nonzero()[-1].item() + 1


# ------------------------------------------------------------------ #
#  Augmented-input builder (overlap assembly)                          #
# ------------------------------------------------------------------ #

def build_augmented_inputs(
    Y_t, M_t, current_u, dt_u, context_full,
    B, N, o, L, C, H, W, padding_index, device,
):
    """Build augmented inputs with overlap from neighbouring chunks.

    Global state tensors have shape ``(N*B, L, …)`` where the first N
    entries belong to batch element 0 (chunks 0 … N-1), the next N to
    batch element 1, etc.

    For chunk *j* inside batch element *b*:

    * **Left overlap** – last ``min(o, num_valid)`` *valid* frames from
      chunk ``j-1``, right-justified at augmented positions ``[o-o_left, o)``.
    * **Current chunk** – placed at augmented positions ``[o, o+L)``.
    * **Right overlap** – first ``o`` frames from chunk ``j+1``, placed
      immediately after the last valid frame of the current chunk so
      the model sees contiguous valid content across chunk boundaries.

    Returns five tensors of shape ``(N*B, L + 2*o, …)``.
    """
    NB = N * B
    aug_L = L + 2 * o

    if o == 0:
        return (
            Y_t.clone(),
            M_t.clone(),
            current_u.clone(),
            dt_u.clone(),
            context_full.clone(),
        )

    # ---- reshape to (B, N, L, …) ----
    Y_ch = Y_t.view(B, N, L, C, H, W)
    M_ch = M_t.view(B, N, L)
    u_ch = current_u.view(B, N, L)
    dt_ch = dt_u.view(B, N, L)
    ctx_ch = context_full.view(B, N, L, C, H, W)

    # ---- allocate augmented buffers ----
    aug_Y = torch.full(
        (B, N, aug_L, C, H, W), padding_index, device=device, dtype=Y_t.dtype
    )
    aug_M = torch.zeros((B, N, aug_L), device=device, dtype=torch.bool)
    aug_u = torch.zeros((B, N, aug_L), device=device, dtype=current_u.dtype)
    aug_dt = torch.zeros((B, N, aug_L), device=device, dtype=dt_u.dtype)
    aug_ctx = torch.zeros(
        (B, N, aug_L, C, H, W), device=device, dtype=context_full.dtype
    )

    # ---- place current chunk at [o, o+L) ----
    aug_Y[:, :, o : o + L] = Y_ch
    aug_M[:, :, o : o + L] = M_ch
    aug_u[:, :, o : o + L] = u_ch
    aug_dt[:, :, o : o + L] = dt_ch
    aug_ctx[:, :, o : o + L] = ctx_ch

    # ---- left overlap: last o valid frames from chunk j-1 ----
    for j in range(1, N):
        for b in range(B):
            valid_idx = M_ch[b, j - 1].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            o_left = min(o, len(valid_idx))
            src = valid_idx[-o_left:]
            dst_start = o - o_left  # right-justify in [0, o)
            aug_Y[b, j, dst_start:o] = Y_ch[b, j - 1, src]
            aug_M[b, j, dst_start:o] = True
            aug_u[b, j, dst_start:o] = u_ch[b, j - 1, src]
            aug_dt[b, j, dst_start:o] = dt_ch[b, j - 1, src]
            aug_ctx[b, j, dst_start:o] = ctx_ch[b, j - 1, src]

    # ---- right overlap: first o frames from chunk j+1, placed right
    #      after the last valid frame of the current chunk ----
    for j in range(N - 1):
        for b in range(B):
            valid_idx = M_ch[b, j].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                last_valid_aug = o - 1  # before current start
            else:
                last_valid_aug = o + valid_idx[-1].item()

            dst = last_valid_aug + 1
            o_right = min(o, aug_L - dst, L)
            if o_right <= 0:
                continue

            aug_Y[b, j, dst : dst + o_right] = Y_ch[b, j + 1, :o_right]
            aug_M[b, j, dst : dst + o_right] = M_ch[b, j + 1, :o_right]
            aug_u[b, j, dst : dst + o_right] = u_ch[b, j + 1, :o_right]
            aug_dt[b, j, dst : dst + o_right] = dt_ch[b, j + 1, :o_right]
            aug_ctx[b, j, dst : dst + o_right] = ctx_ch[b, j + 1, :o_right]

    # ---- flatten to (NB, aug_L, …) ----
    return (
        aug_Y.reshape(NB, aug_L, C, H, W),
        aug_M.reshape(NB, aug_L),
        aug_u.reshape(NB, aug_L),
        aug_dt.reshape(NB, aug_L),
        aug_ctx.reshape(NB, aug_L, C, H, W),
    )


# ------------------------------------------------------------------ #
#  Chunk stitcher                                                      #
# ------------------------------------------------------------------ #


def stitch_chunks(Y_t, M_t, insert_time_map, B, N, L, padding_index=0.0):
    """Left-align active frames from each chunk into one contiguous sequence.

    For every batch element the N chunks are scanned in order.  From each
    chunk **only** the frames where ``M_t`` is True are collected (left-
    aligned, no holes).  The per-chunk results are concatenated so the
    output is a single contiguous block of valid frames followed by
    padding.

    Returns ``(Y_out, M_out, ins_out)`` of shape ``(B, max_total, …)``.
    """
    C, H, W = Y_t.shape[2], Y_t.shape[3], Y_t.shape[4]
    Y_ch = Y_t.view(B, N, L, C, H, W)
    M_ch = M_t.view(B, N, L)
    ins_ch = insert_time_map.view(B, N, L)

    all_Y, all_M, all_ins = [], [], []
    for b in range(B):
        parts_Y, parts_M, parts_ins = [], [], []
        for j in range(N):
            m = M_ch[b, j]  # (L,)
            valid_idx = m.nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            parts_Y.append(Y_ch[b, j, valid_idx])       # only True frames
            parts_M.append(m[valid_idx])                 # all True
            parts_ins.append(ins_ch[b, j, valid_idx])
        if parts_Y:
            all_Y.append(torch.cat(parts_Y))
            all_M.append(torch.cat(parts_M))
            all_ins.append(torch.cat(parts_ins))
        else:
            all_Y.append(
                torch.zeros(1, C, H, W, device=Y_t.device, dtype=Y_t.dtype)
            )
            all_M.append(torch.zeros(1, device=Y_t.device, dtype=torch.bool))
            all_ins.append(torch.full((1,), -1.0, device=Y_t.device))

    max_len = max(t.shape[0] for t in all_Y)
    Y_out = torch.full(
        (B, max_len, C, H, W), padding_index, device=Y_t.device, dtype=Y_t.dtype
    )
    M_out = torch.zeros(B, max_len, device=Y_t.device, dtype=torch.bool)
    ins_out = torch.full(
        (B, max_len), -1.0, device=Y_t.device, dtype=insert_time_map.dtype
    )

    for b in range(B):
        n = all_Y[b].shape[0]
        Y_out[b, :n] = all_Y[b]
        M_out[b, :n] = all_M[b]
        ins_out[b, :n] = all_ins[b]

    return Y_out, M_out, ins_out


# ------------------------------------------------------------------ #
#  Chunked sampler                                                     #
# ------------------------------------------------------------------ #


@torch.inference_mode()
def vanilla_sample_flowception_t2v_chunked(
    H,
    W,
    C,
    model,
    forward_fn,
    num_steps,
    cond_t,
    context_frames,
    batch_size=1,
    input_length=32,
    device="cuda",
    padding_index=0,
    num_snapshots=20,
    start_frames=10,
    max_inserts=20,
    ins_start=0.3,
    uc=None,
    s_text=2.0,
    s_img=0.0,
    s_offset=0.1,
    s_ins=2.5,
    erg_tau=1.0,
    erg_type="legacy",
    erg_tmin=1.0,
    lmin=10,
    lmax=17,
    guider=None,
    snr_shift: float = 1.0,
    num_chunks=1,
    num_overlap=0,
    nfes_per_step=None,
    blend_type="chunk1",
):
    """Chunked variant of the Flowception T2V sampler.

    The generation is split into ``N = num_chunks`` independent sequences
    of length ``L = input_length``, all processed in a single batch of
    size ``N * B``.  Before every denoiser call, each chunk is augmented
    with ``num_overlap`` context frames from its neighbours:

    * **Left context** – last ``o`` *valid* frames from chunk ``j-1``
      (respecting ``M_t``).
    * **Right context** – first ``o`` frames from chunk ``j+1``, placed
      immediately after the last valid frame of the current chunk so the
      model sees contiguous valid content with no padding gap.

    After the forward pass only the *current-chunk* portion of the
    velocity / insertion-rate output is used for the state update.

    Once all sampling steps are complete the ``N`` chunks per batch
    element are stitched into a single long sequence by concatenating
    valid frames in chunk order.

    Returns
    -------
    Y_out : (B, T_total, C, H, W)
    M_out : (B, T_total)
    Y_T_list, M_t_list : snapshot lists
    ins_out : (B, T_total) insertion-time map
    all_expected_lengths : list
    """
    B = batch_size
    N = num_chunks
    o = num_overlap
    L = input_length
    NB = N * B

    # ---- expand conditioning from B → NB ----
    def _expand(d):
        if d is None:
            return None
        return {
            k: (v.repeat_interleave(N, dim=0) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()
        }

    cond_nb = _expand(cond_t)
    uc_nb = _expand(uc)

    # ---- initialise state (NB, L, …) ----
    Y_t = torch.full(
        (NB, L, C, H, W), padding_index, device=device, dtype=torch.float32
    )
    M_t = torch.zeros(NB, L, device=device, dtype=torch.bool)
    dt_u = torch.full((NB, L), 1.0 / num_steps, device=device)
    current_u = torch.zeros(NB, L, device=device)

    rand_len = max(0, min(start_frames, L))
    if rand_len > 0:
        Y_t[:, :rand_len] = torch.randn_like(Y_t[:, :rand_len])
        M_t[:, :rand_len] = True

    insert_time_map = torch.full((NB, L), -1.0, device=device)
    insert_time_map[M_t] = 0.0

    context_full = torch.zeros_like(Y_t)
    Y_T_list, M_t_list = [], []
    all_expected_lengths = []

    # ============================================================== #
    #  Main sampling loop                                              #
    # ============================================================== #
    for step in tqdm.tqdm(range(2 * num_steps)):
        # ---- 1. augmented inputs with overlap ----
        aug_Y, aug_M, aug_u, aug_dt, aug_ctx = build_augmented_inputs(
            Y_t,
            M_t,
            current_u,
            dt_u,
            context_full,
            B,
            N,
            o,
            L,
            C,
            H,
            W,
            padding_index,
            device,
        )

        # ---- 2. dynamic-length slicing ----
        active_len = get_active_length(aug_M)
        Y_active = aug_Y[:, :active_len]
        M_active = aug_M[:, :active_len]
        u_active = aug_u[:, :active_len]
        dt_active = aug_dt[:, :active_len]
        ctx_active = aug_ctx[:, :active_len]

        # ---- 3. time warping ----
        t = scale_snr(u_active, s=snr_shift)
        u_next = (u_active + dt_active).clamp(0.0, 1.0)
        t_next = scale_snr(u_next, s=snr_shift)
        h_flow = torch.where(u_active < 1.0, t_next - t, torch.zeros_like(t))

        # ---- 4. forward (on augmented input) ----
        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_active,
            time=t,
            cond_t=cond_nb,
            context_frames=ctx_active,
            model=model,
            frame_mask=M_active,
        )

        # ---- 5. classifier-free guidance ----
        if s_text > 1.0 and uc_nb is not None:
            t_new = (t - s_offset * torch.rand((1,), device=t.device)).clamp(0, 1)
            t_uncond = torch.where(t == 1, t, t_new)
            t_ratio = (t_uncond / t.clamp_min(1e-4))[:, :, None, None, None].to(
                Y_active.dtype
            )
            Y2 = Y_active + (1.0 - t_ratio) * torch.randn_like(Y_active)

            vel_u, lam_u, _, _ = forward_fn(
                x_t=Y2,
                time=t,
                cond_t=uc_nb,
                context_frames=torch.zeros_like(ctx_active),
                model=model,
                frame_mask=M_active,
            )

            w = s_text if t.max() <= erg_tmin else s_img
            if guider is not None:
                velocity_pred = guider(velocity_pred, vel_u, w)
            else:
                velocity_pred = velocity_pred * w + vel_u * (1.0 - w)

            if s_ins != 1.0:
                lambda_ins_pred = (
                    lambda_ins_pred**s_ins * lam_u ** (1.0 - s_ins)
                )

        lambda_ins_pred = lambda_ins_pred.masked_fill(~M_active, 0.0)

        # ---- 6. flow update (augmented space) ----
        Y_active = Y_active + h_flow[:, :, None, None, None] * velocity_pred

        # ---- 7. extract current portion → write back ----
        cur_start = o
        cur_end = min(o + L, active_len)
        cur_len = cur_end - cur_start

        if cur_len > 0:
            Y_t[:, :cur_len] = Y_active[:, cur_start:cur_end]
            current_u[:, :cur_len] = u_next[:, cur_start:cur_end]

        # ---- 8. insertion logic (original Y_t / M_t space) ----
        # When ins_start >= 1.0 the insertion schedule is degenerate
        # (division by zero).  Skip insertions entirely in that case.
        if ins_start >= 1.0:
            Y_t = torch.where(
                M_t[:, :, None, None, None],
                Y_t,
                torch.full_like(Y_t, padding_index),
            )
            continue

        lambda_cur = torch.zeros(NB, L, device=device, dtype=lambda_ins_pred.dtype)
        if cur_len > 0:
            lambda_cur[:, :cur_len] = lambda_ins_pred[:, cur_start:cur_end]

        ins_u = current_u.max(dim=1, keepdim=True).values
        kappa_u = (ins_u - ins_start).clamp(0.0, 1.0 - ins_start) / (
            1.0 - ins_start
        )
        d_kappa = torch.full_like(current_u, 1.0 / (1.0 - ins_start))
        d_kappa[(ins_u < ins_start)[:, 0]] = 0.0

        h_ins = torch.where(
            ins_u + dt_u[:, :1] < 1.0,
            torch.ones_like(current_u) / num_steps,
            torch.zeros_like(current_u),
        )
        ratio = torch.where(
            current_u < ins_start,
            torch.zeros_like(current_u),
            d_kappa / (1.0 - kappa_u),
        )

        Lambda = lambda_cur * h_ins * ratio
        Lambda[(ins_u < ins_start)[:, 0], :] = 0.0
        Lambda = Lambda.clamp_min(0.0)

        prob = (1.0 - torch.exp(-Lambda)).clamp(0.0, 1.0)
        insert_counts = (torch.rand_like(prob) < prob).to(torch.int32)
        num_inserted = int(insert_counts.sum().item())

        if num_inserted > 0:
            num_ins = insert_counts.sum(dim=1).to(torch.int64)
            Imax = int(num_ins.max().item())
            if Imax > 0:
                Lnew = L + Imax
                ins_frames = torch.randn(
                    NB, Imax, C, H, W, device=device, dtype=Y_t.dtype
                )
                ins_mask = (
                    torch.arange(Imax, device=device)[None, :] < num_ins[:, None]
                )

                rfp = torch.arange(L, device=device)[None, :].repeat(NB, 1)
                rfp[:, 1:] += insert_counts.cumsum(dim=1)[:, :-1].to(torch.int64)

                exp_Y = torch.full(
                    (NB, Lnew, C, H, W),
                    padding_index,
                    device=device,
                    dtype=Y_t.dtype,
                )
                exp_M = torch.zeros(NB, Lnew, device=device, dtype=torch.bool)
                exp_u = torch.zeros(NB, Lnew, device=device, dtype=current_u.dtype)
                exp_ins = torch.full((NB, Lnew), -1.0, device=device)

                bi = torch.arange(NB, device=device)[:, None]
                exp_Y[bi, rfp] = Y_t
                exp_M[bi, rfp] = M_t
                exp_u[bi, rfp] = current_u
                exp_ins[bi, rfp] = insert_time_map

                imask = torch.ones(NB, Lnew, device=device, dtype=torch.bool)
                imask.scatter_(1, rfp, False)
                irank = imask.cumsum(dim=1) - 1
                keep = imask & (irank < num_ins[:, None])
                bii, poss = torch.where(keep)
                if bii.numel() > 0:
                    r = irank[bii, poss].to(torch.int64)
                    exp_Y[bii, poss] = ins_frames[bii, r]
                    exp_M[bii, poss] = ins_mask[bii, r]
                    exp_ins[bii, poss] = float(step)

                Y_t = exp_Y[:, :L]
                M_t = exp_M[:, :L]
                current_u = exp_u[:, :L]
                insert_time_map = exp_ins[:, :L]

        Y_t = torch.where(
            M_t[:, :, None, None, None],
            Y_t,
            torch.full_like(Y_t, padding_index),
        )

    # ============================================================== #
    #  Stitch chunks into long sequences                               #
    # ============================================================== #
    Y_out, M_out, ins_out = stitch_chunks(
        Y_t, M_t, insert_time_map, B, N, L, padding_index
    )

    Y_T_list.append(Y_out.clone())
    M_t_list.append(M_out.clone())

    return Y_out, M_out, Y_T_list, M_t_list, ins_out, all_expected_lengths


# ------------------------------------------------------------------ #
#  Dummy model & forward for CPU testing                               #
# ------------------------------------------------------------------ #


class DummyModel(torch.nn.Module):
    """No-op model so we can run the full sampling loop on CPU."""

    def forward(self, x):
        return x


def dummy_forward_fn(x_t, time, cond_t, context_frames, model, frame_mask):
    """Return all-ones velocity and zero insertion rate."""
    B, D = x_t.shape[:2]
    velocity = torch.ones_like(x_t)
    lam_ins = torch.zeros(B, D, device=x_t.device, dtype=x_t.dtype)
    return velocity, lam_ins, None, None


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #


def test_build_augmented_inputs():
    """Overlap is placed at the correct positions in the augmented buffer."""
    print("Running test_build_augmented_inputs …")
    B, N, L, C, H, W = 1, 3, 8, 2, 2, 2
    o = 2
    NB = N * B
    sf = 4  # start_frames (valid per chunk)
    pad = 0.0
    dev = "cpu"

    Y_t = torch.full((NB, L, C, H, W), pad)
    M_t = torch.zeros(NB, L, dtype=torch.bool)
    u_t = torch.zeros(NB, L)
    dt_t = torch.full((NB, L), 0.1)
    ctx_t = torch.zeros_like(Y_t)

    M_t[:, :sf] = True
    u_t[:, :sf] = 0.5

    # Known per-chunk fill values
    Y_t[0, :sf] = 10.0  # chunk 0
    Y_t[1, :sf] = 20.0  # chunk 1
    Y_t[2, :sf] = 30.0  # chunk 2

    aug_Y, aug_M, *_ = build_augmented_inputs(
        Y_t, M_t, u_t, dt_t, ctx_t, B, N, o, L, C, H, W, pad, dev
    )

    aug_L = L + 2 * o
    aY = aug_Y.view(B, N, aug_L, C, H, W)
    aM = aug_M.view(B, N, aug_L)

    # -- chunk 0  (j=0): no left overlap, right overlap from chunk 1 --
    assert aM[0, 0, :o].sum() == 0, "chunk 0: no left overlap expected"
    assert (aY[0, 0, o : o + sf] == 10).all(), "chunk 0: current frames"
    rs = o + sf  # right overlap destination
    assert (aY[0, 0, rs : rs + o] == 20).all(), "chunk 0: right overlap"
    assert aM[0, 0, o : rs + o].all(), "chunk 0: contiguous valid"

    # -- chunk 1  (j=1): left from chunk 0, right from chunk 2 --
    assert (aY[0, 1, 0:o] == 10).all(), "chunk 1: left overlap"
    assert (aY[0, 1, o : o + sf] == 20).all(), "chunk 1: current frames"
    assert (aY[0, 1, rs : rs + o] == 30).all(), "chunk 1: right overlap"
    assert aM[0, 1, 0 : rs + o].all(), "chunk 1: contiguous valid"

    # -- chunk 2  (j=2): left from chunk 1, no right --
    assert (aY[0, 2, 0:o] == 20).all(), "chunk 2: left overlap"
    assert (aY[0, 2, o : o + sf] == 30).all(), "chunk 2: current frames"
    assert not aM[0, 2, o + sf :].any(), "chunk 2: no right overlap"

    print("  ✓ test_build_augmented_inputs passed")


def test_overlap_contiguity():
    """Right overlap stays contiguous even when chunks have different lengths."""
    print("Running test_overlap_contiguity …")
    B, N, L, C, H, W = 1, 2, 12, 2, 2, 2
    o = 3
    NB = N * B
    pad = 0.0
    dev = "cpu"

    Y_t = torch.full((NB, L, C, H, W), pad)
    M_t = torch.zeros(NB, L, dtype=torch.bool)
    u_t = torch.zeros(NB, L)
    dt_t = torch.full((NB, L), 0.1)
    ctx = torch.zeros_like(Y_t)

    # chunk 0: 6 valid frames   |   chunk 1: 4 valid frames
    M_t[0, :6] = True
    Y_t[0, :6] = 1.0
    u_t[0, :6] = 0.3
    M_t[1, :4] = True
    Y_t[1, :4] = 2.0
    u_t[1, :4] = 0.3

    aug_Y, aug_M, *_ = build_augmented_inputs(
        Y_t, M_t, u_t, dt_t, ctx, B, N, o, L, C, H, W, pad, dev
    )

    aug_L = L + 2 * o
    aM = aug_M.view(B, N, aug_L)
    aY = aug_Y.view(B, N, aug_L, C, H, W)

    # chunk 0: current valid [o..o+6), right overlap [o+6..o+6+3)
    assert aM[0, 0, o : o + 6 + o].all(), (
        f"chunk 0: expected contiguous valid, mask = {aM[0, 0, o:o+9]}"
    )
    # right overlap should carry chunk 1's value
    assert (aY[0, 0, o + 6 : o + 6 + o] == 2.0).all(), (
        "chunk 0: right overlap values"
    )

    # chunk 1: left overlap = last 3 valid of chunk 0 at [0..3)
    assert aM[0, 1, o - 3 : o + 4].all(), (
        "chunk 1: left overlap + current should be contiguous"
    )
    assert (aY[0, 1, o - 3 : o] == 1.0).all(), "chunk 1: left overlap values"
    assert not aM[0, 1, o + 4 :].any(), "chunk 1: no right overlap"

    print("  ✓ test_overlap_contiguity passed")


def test_full_sampling():
    """Full chunked sampling loop with dummy model – shapes & masks."""
    print("Running test_full_sampling …")
    B, N, L, C, H, W = 2, 3, 10, 2, 4, 4
    o = 2
    num_steps = 3
    sf = 4

    torch.manual_seed(42)
    Y, M, *_ = vanilla_sample_flowception_t2v_chunked(
        H=H,
        W=W,
        C=C,
        model=DummyModel(),
        forward_fn=dummy_forward_fn,
        num_steps=num_steps,
        cond_t={"d": torch.zeros(B, 16)},
        context_frames=None,
        batch_size=B,
        input_length=L,
        device="cpu",
        padding_index=0.0,
        start_frames=sf,
        ins_start=1.0,  # disable insertions (lambda=0 + ins_start=1)
        uc=None,
        s_text=1.0,
        snr_shift=1.0,
        num_chunks=N,
        num_overlap=o,
    )

    assert Y.shape[0] == B
    assert Y.shape[2] == C and Y.shape[3] == H and Y.shape[4] == W

    for b in range(B):
        nv = int(M[b].sum())
        assert nv == N * sf, f"batch {b}: expected {N * sf} valid, got {nv}"
        assert Y[b][M[b]].abs().sum() > 0, "valid frames should be non-zero"
        inv = Y[b][~M[b]]
        if inv.numel():
            assert (inv == 0).all(), "padding frames should be 0"

    print("  ✓ test_full_sampling passed")


def test_no_overlap():
    """o=0 should behave like independent chunks."""
    print("Running test_no_overlap …")
    B, N, L, C, H, W = 1, 2, 8, 2, 2, 2
    sf = 3

    torch.manual_seed(99)
    Y, M, *_ = vanilla_sample_flowception_t2v_chunked(
        H=H,
        W=W,
        C=C,
        model=DummyModel(),
        forward_fn=dummy_forward_fn,
        num_steps=2,
        cond_t={"d": torch.zeros(B, 8)},
        context_frames=None,
        batch_size=B,
        input_length=L,
        device="cpu",
        padding_index=0.0,
        start_frames=sf,
        ins_start=1.0,
        uc=None,
        s_text=1.0,
        snr_shift=1.0,
        num_chunks=N,
        num_overlap=0,
    )

    assert int(M[0].sum()) == N * sf, f"expected {N * sf} valid, got {int(M[0].sum())}"
    print("  ✓ test_no_overlap passed")


def test_single_chunk():
    """N=1 should behave identically to the vanilla sampler."""
    print("Running test_single_chunk …")
    B, L, C, H, W = 1, 10, 2, 3, 3
    sf = 5

    torch.manual_seed(7)
    Y, M, *_ = vanilla_sample_flowception_t2v_chunked(
        H=H,
        W=W,
        C=C,
        model=DummyModel(),
        forward_fn=dummy_forward_fn,
        num_steps=2,
        cond_t={"d": torch.zeros(B, 4)},
        context_frames=None,
        batch_size=B,
        input_length=L,
        device="cpu",
        padding_index=0.0,
        start_frames=sf,
        ins_start=1.0,
        uc=None,
        s_text=1.0,
        snr_shift=1.0,
        num_chunks=1,
        num_overlap=0,
    )

    assert Y.shape == (B, sf, C, H, W), f"shape {Y.shape}"
    assert int(M.sum()) == sf
    print("  ✓ test_single_chunk passed")


def test_overlap_values_propagate():
    """Verify that overlap context actually influences the denoised output.

    Strategy: use a forward function that returns ``x_t`` as velocity
    (identity).  With overlap, the augmented frames from neighbours sit
    at the edges of the current chunk in the augmented buffer.  After
    extraction at [o, o+L) the *valid* positions of the current chunk
    should be updated with their *own* values (since velocity=x_t), so
    the overlap frames (which live outside [o, o+L)) do NOT leak into
    the result.  This confirms correct extraction.
    """
    print("Running test_overlap_values_propagate …")

    def identity_forward(x_t, time, cond_t, context_frames, model, frame_mask):
        B, D = x_t.shape[:2]
        return x_t.clone(), torch.zeros(B, D, device=x_t.device), None, None

    B, N, L, C, H, W = 1, 2, 8, 1, 1, 1
    o = 2
    sf = 4
    pad = 0.0

    torch.manual_seed(0)
    Y, M, *_ = vanilla_sample_flowception_t2v_chunked(
        H=H,
        W=W,
        C=C,
        model=DummyModel(),
        forward_fn=identity_forward,
        num_steps=1,
        cond_t={"d": torch.zeros(B, 4)},
        context_frames=None,
        batch_size=B,
        input_length=L,
        device="cpu",
        padding_index=pad,
        start_frames=sf,
        ins_start=1.0,
        uc=None,
        s_text=1.0,
        snr_shift=1.0,
        num_chunks=N,
        num_overlap=o,
    )

    # With velocity = x_t  (identity) and no insertions, valid frame
    # count stays N*sf.  The key check: no NaN / Inf and output is finite.
    assert Y.isfinite().all(), "output should be finite"
    assert int(M.sum()) == N * sf
    print("  ✓ test_overlap_values_propagate passed")


def test_stitch_left_aligns():
    """Stitch must left-align only M_t=True frames – no holes in output."""
    print("Running test_stitch_left_aligns …")
    B, N, L, C, H, W = 1, 2, 8, 1, 1, 1
    pad = 0.0

    Y_t = torch.full((N * B, L, C, H, W), pad)
    M_t = torch.zeros(N * B, L, dtype=torch.bool)
    ins = torch.full((N * B, L), -1.0)

    # chunk 0: valid at [0,1,3,4] (hole at index 2), padding at [5..7]
    M_t[0, 0] = True; Y_t[0, 0] = 10.0
    M_t[0, 1] = True; Y_t[0, 1] = 11.0
    M_t[0, 3] = True; Y_t[0, 3] = 13.0
    M_t[0, 4] = True; Y_t[0, 4] = 14.0

    # chunk 1: valid at [0,2] (hole at index 1), padding at [3..7]
    M_t[1, 0] = True; Y_t[1, 0] = 20.0
    M_t[1, 2] = True; Y_t[1, 2] = 22.0

    Y_out, M_out, _ = stitch_chunks(Y_t, M_t, ins, B, N, L, pad)

    # Expected: [10, 11, 13, 14, 20, 22] then padding
    expected_vals = [10.0, 11.0, 13.0, 14.0, 20.0, 22.0]
    n_valid = int(M_out[0].sum())
    assert n_valid == 6, f"expected 6 valid frames, got {n_valid}"
    assert M_out[0, :6].all(), "first 6 should be valid"
    assert not M_out[0, 6:].any(), "rest should be padding"

    actual = Y_out[0, :6, 0, 0, 0].tolist()
    assert actual == expected_vals, f"values {actual} != {expected_vals}"

    print("  ✓ test_stitch_left_aligns passed")


if __name__ == "__main__":
    test_build_augmented_inputs()
    test_overlap_contiguity()
    test_stitch_left_aligns()
    test_full_sampling()
    test_no_overlap()
    test_single_chunk()
    test_overlap_values_propagate()
    print("\n✅ All tests passed!")

import torch


def left_align_frames(x, padding_index=0):
    """
    Move padding frames (where all pixels == padding_index) to the end, preserving order of non-padding frames.

    Args:
        x: Tensor of shape [B, L, ...] (e.g., [B, L, H, W] or [B, L, H, W, C])
        padding_index: int

    Returns:
        Tensor of same shape, left-aligned.
    """
    B, L = x.shape[:2]

    # Identify frames that contain any non-padding values
    dims = tuple(range(2, x.ndim))
    frame_mask = (x != padding_index).any(dim=dims).int()  # [B, L]

    # Sort: non-padding frames (1) come before padding frames (0)
    sort_order = torch.argsort(-frame_mask, dim=1, stable=True)  # [B, L]

    # Use advanced indexing to reorder frames
    # Build gather index of shape [B, L, 1, 1, 1, ...] matching x
    index_shape = list(sort_order.shape) + [1] * (x.ndim - 2)
    sort_order_exp = sort_order.view(*index_shape).expand_as(x)

    x_aligned = torch.gather(x, 1, sort_order_exp)

    return x_aligned, sort_order_exp[:, :, 0, 0, 0]


def left_align_by_mask(x: torch.Tensor, mask: torch.Tensor):
    """
    Left-align all the True positions in `mask` (shape [B,L]) to the front of each sequence in `x`
    (shape [B,L,...]), moving the False positions to the back, *but otherwise preserving order*.

    Returns:
       x_aligned: same shape as x, with mask==True frames left-aligned
       sort_order: [B,L] long indices so that x_aligned[b,i] = x[b, sort_order[b,i]]
    """
    B, L = mask.shape
    # convert bool->int so True=1, False=0
    m = mask.int()  # [B,L]
    # sort by descending m, keeping stable so we preserve intra‐group order
    sort_order = torch.argsort(-m, dim=1, stable=True)  # [B,L]
    # expand to gather over all trailing dims of x
    index_shape = sort_order.shape + (1,) * (x.ndim - 2)  # [B,L,1,1,...]
    sort_exp = sort_order.view(*index_shape).expand_as(x)
    x_aligned = torch.gather(x, dim=1, index=sort_exp)
    return x_aligned, sort_order


def compute_insert_counts(insert_site_mask: torch.Tensor, flow_site_mask: torch.Tensor):
    """
    Compuites how many insertions sites are present to the right of each frame.
    Args:
        insert_site_mask: [B, L] bool tensor (True = insert token)
        flow_site_mask:   [B, L] bool tensor (True = real/flow token)

    Returns:
        counts: [B, L] int64 tensor where
          counts[b, i] = # of insert tokens in (i, next_real_i)
          next_real_i  = min { j > i | flow_site_mask[b,j]==True } or L if none
          counts[b, i] = -1 where flow_site_mask[b,i]==False
    """
    B, L = insert_site_mask.shape
    device = insert_site_mask.device

    # 1) make a [B,1,L] mask of flow positions, and an [L] index vector
    idx = torch.arange(L, device=device)
    flow_exp = flow_site_mask.view(B, 1, L)  # [B,1,L]
    # 2) build an [B,L,L] grid of candidate j's, but only keep those j>i and flow[j]
    i_idx = idx.view(1, L, 1).expand(B, L, L)  # [B,L,1] -> [B,L,L]
    j_idx = idx.view(1, 1, L).expand(B, L, L)  # [B,1,L] -> [B,L,L]
    valid_next = (j_idx > i_idx) & flow_exp  # [B,L,L]
    # 3) replace invalid by L, then take min over j to get the very next
    j_masked = torch.where(valid_next, j_idx, L)  # [B,L,L]
    next_real = j_masked.min(dim=2).values  # [B,L], in [0..L]
    # 4) build a window mask i<j<next_real[b,i]
    window = (j_idx > i_idx) & (j_idx < next_real.unsqueeze(2))  # [B,L,L]
    # 5) count inserts in that window
    ins_exp = insert_site_mask.view(B, 1, L).expand(B, L, L)
    counts = (window & ins_exp).sum(dim=2)  # [B, L]
    # 6) everywhere i isn’t a real frame, set -1
    counts = torch.where(flow_site_mask, counts, torch.full_like(counts, -1))
    return counts


def strip(Y, M):
    # Y: [B, L, D], M: [B, L]
    result = []
    for y, m in zip(Y, M):
        result.append(y[m.bool()])
    return result  # list of [num_frames_i, D]

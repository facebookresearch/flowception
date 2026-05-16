import torch, tqdm
import torch.nn.functional as F


@torch.inference_mode()
def vanilla_sample_flowception(
    first_frames,
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
):
    H, W, C = first_frames.shape[-3:]
    B = batch_size

    # support for more than one context frame.
    Y_t = torch.full((B, input_length, H, W, C), padding_index, device=device, dtype=torch.float32)
    M_t = torch.zeros(B, input_length, device=device, dtype=torch.bool)

    K = 0
    if first_frames is not None:
        # first_frames: [B, K, H, W, C]  (your caller passes BCHWT elsewhere; keep consistent with this function)
        K = first_frames.shape[1]
        K = min(K, input_length)
        Y_t[:, :K] = first_frames[:, :K]
        M_t[:, :K] = True

    rand_len = max(0, min(start_frames, input_length - K))
    if rand_len > 0:
        Y_t[:, K : K + rand_len] = torch.randn_like(Y_t[:, K : K + rand_len])
        M_t[:, K : K + rand_len] = True

    insert_time_map = torch.full((B, input_length), -1.0, dtype=torch.float32, device=device)
    insert_time_map[M_t] = 0

    dt = torch.full((B, input_length), 1 / num_steps, device=device)

    current_t = torch.zeros(B, input_length, dtype=torch.float32, device=device)

    Y_T_list = []
    M_t_list = []

    step = 0
    num_frames = input_length
    total_inserts = torch.zeros(B, device=device)
    all_expected_lengths = []

    context_full = torch.zeros_like(Y_t)
    context_full[:, :K] = first_frames[:, :K]

    for step in tqdm.tqdm(range(2 * num_steps)):
        # keep h_flow: it already zeroes updates where t >= 1

        t = current_t
        h = dt
        h_flow = torch.where(t + h < 1.0, h, torch.zeros_like(h))

        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            # context_frames=context_frames,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,
        )

        if s_text > 1.0:
            t_new = (t - s_offset * torch.rand((1,), device=t.device)).clip(0, 1)
            t_uncond = torch.where(t == 1, t, t_new)
            t_ratio = (t_uncond / t.clip(1e-4, None))[:, :, None, None, None].to(Y_t.dtype)
            # Y2 = t_ratio * Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
            Y2 = Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
            Y2[:, 0] = 0 * Y_t[:, 0]
            context_uncond = torch.zeros_like(context_full)

            velocity_pred_uncond, lambda_ins_pred_uncond, _, _ = forward_fn(
                # x_t=Y_t,
                x_t=Y2,
                time=t,
                cond_t=uc,
                # context_frames=context_frames,
                # context_frames=context_full,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )
            velocity_pred = velocity_pred * s_text + velocity_pred_uncond * (1.0 - s_text)
            if s_ins != 1.0:
                lambda_ins_pred = lambda_ins_pred**s_ins * lambda_ins_pred_uncond ** (1 - s_ins)
        lambda_ins_pred[(Y_t == padding_index)[..., 0, 0, 0]] = 0.0

        Y_t[:, 1:] = Y_t[:, 1:] + h_flow[:, 1:, None, None, None] * velocity_pred[:, 1:]

        # Compute poisson insert counts (lambda * h per position)
        ins_t = torch.max(current_t, dim=1, keepdim=True).values
        kappa_t = (ins_t - ins_start).clip(0, 1 - ins_start) / (
            1 - ins_start
        )  # 0 from 1 to ins_start, then linear up to 1.

        d_kappa = torch.full_like(h_flow, 1 / (1 - ins_start))
        d_kappa[(ins_t < ins_start)[:, 0]] = 0

        h_ins = torch.where((ins_t + h < 1.0), torch.ones_like(h) / num_steps, torch.zeros_like(h))

        ratio = torch.where(current_t < ins_start, torch.zeros_like(h), d_kappa / (1 - kappa_t))
        # ratio = torch.where(current_t < ins_start, torch.zeros_like(h), torch.ones_like(h))
        h_lamb = lambda_ins_pred * h_ins * ratio
        h_lamb[(ins_t < ins_start)[:, 0], :] = 0.0

        prob_insertion = h_lamb
        # prob_insertion = 1 - torch.exp(-h_lamb)

        # Compute expected insertions per video before sampling
        expected_inserts = lambda_ins_pred.sum(dim=1)  # shape [B]
        all_expected_lengths.append((expected_inserts + M_t.sum(1)).tolist())

        # Either one insertion occurs or no insertion occurs
        insert_counts = (torch.rand_like(prob_insertion) < prob_insertion).int()

        total_inserts = total_inserts + insert_counts.sum(dim=1)
        num_inserted = insert_counts.sum().item()

        # advance time
        current_t = (current_t + h_flow).clip(0, 1)

        if num_inserted > 0:
            insertions = insert_counts
            num_insertions = insert_counts.sum(1)

            # generate frames to insert
            inserting_frames = torch.randn((B, num_insertions.max(), H, W, C), device=device)
            inserting_mask = (
                torch.arange(num_insertions.max(), device=device)[None].repeat(B, 1) < num_insertions[:, None]
            )

            real_frames_pos = torch.arange(num_frames, device=device)[None].repeat(B, 1)
            real_frames_pos[:, 1:] = real_frames_pos[:, 1:] + insertions.cumsum(1)[:, :-1]

            # Now find the positions of the inserted frames.
            all_frames_indices = torch.arange(num_frames + num_insertions.max(), device=device)
            insert_indices_oneh = 1 - (
                real_frames_pos[:, :, None] == all_frames_indices[None, None, :].repeat(B, 1, 1)
            ).sum(1)

            expanded_frames = torch.full(
                (B, num_frames + num_insertions.max(), H, W, C), -1, dtype=torch.float32, device=device
            )
            expanded_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )
            expanded_mask = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.bool, device=device
            )
            expanded_ins_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )

            batch_idx = torch.arange(B, device=device)[:, None]
            expanded_frames[batch_idx, real_frames_pos, :, :, :] = Y_t
            expanded_mask[batch_idx, real_frames_pos] = M_t
            expanded_times[batch_idx, real_frames_pos] = current_t
            expanded_ins_times[batch_idx, real_frames_pos] = insert_time_map

            I = insert_indices_oneh.sum(dim=1)[0]  # number of insertions per batch (same for all)
            insert_frames_pos = (
                insert_indices_oneh.int().topk(I, dim=1).indices.sort(dim=1).values
            )  # shape [B, I]

            expanded_frames[batch_idx, insert_frames_pos, :, :, :] = inserting_frames
            expanded_mask[batch_idx, insert_frames_pos] = inserting_mask
            expanded_ins_times[batch_idx, insert_frames_pos] = step

            # everything is in interleaved as needed, now we can clip back to keep the same length.
            Y_t = expanded_frames[:, :num_frames]
            M_t = expanded_mask[:, :num_frames]
            insert_time_map = expanded_ins_times[:, :num_frames]

            current_t = expanded_times[:, :num_frames]

        Y_t = torch.where(
            M_t[:, :, None, None, None],
            Y_t,
            torch.full_like(Y_t, padding_index),
        )

        step += 1

    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())

    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


@torch.inference_mode()
def vanilla_sample_noisy_flowception(
    first_frames,
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
):
    H, W, C = first_frames.shape[-3:]
    B = batch_size

    # support for more than one context frame.
    Y_t = torch.full((B, input_length, H, W, C), padding_index, device=device, dtype=torch.float32)
    M_t = torch.zeros(B, input_length, device=device, dtype=torch.bool)

    K = 0
    if first_frames is not None:
        # first_frames: [B, K, H, W, C]  (your caller passes BCHWT elsewhere; keep consistent with this function)
        K = first_frames.shape[1]
        K = min(K, input_length)
        # Y_t[:, :K] = first_frames[:, :K]
        M_t[:, :K] = True

    rand_len = max(0, min(start_frames, input_length - K))
    if rand_len > 0:
        Y_t[:, K : K + rand_len] = torch.randn_like(Y_t[:, K : K + rand_len])
        M_t[:, K : K + rand_len] = True

    insert_time_map = torch.full((B, input_length), -1.0, dtype=torch.float32, device=device)
    insert_time_map[M_t] = 0

    dt = torch.full((B, input_length), 1 / num_steps, device=device)

    current_t = torch.zeros(B, input_length, dtype=torch.float32, device=device)

    Y_T_list = []
    M_t_list = []

    step = 0
    num_frames = input_length
    total_inserts = torch.zeros(B, device=device)
    all_expected_lengths = []

    context_full = torch.zeros_like(Y_t)
    # shifted schedule with less noise.
    context_full[:, :K] = first_frames[:, :K] * s_offset / 2 + (1 - s_offset / 2) * torch.randn_like(
        first_frames[:, :K]
    )

    Y_noise = torch.randn_like(first_frames[:, :K])

    for step in tqdm.tqdm(range(2 * num_steps)):
        # keep h_flow: it already zeroes updates where t >= 1

        t = current_t
        h = dt
        h_flow = torch.where(t + h < 1.0, h, torch.zeros_like(h))
        ct = current_t.max(dim=1, keepdim=True).values[:, :, None, None, None]
        ct2 = (s_offset + ct).clip(0, 1)
        ct3 = (s_offset / 2 + ct).clip(0, 1)

        Y_t[:, :K] = first_frames[:, :K] * ct + (1 - ct) * Y_noise

        context_full = torch.zeros_like(Y_t)
        # shifted schedule with less noise.
        context_full[:, :K] = first_frames[:, :K] * ct3 + (1 - ct3) * torch.randn_like(first_frames[:, :K])

        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            # context_frames=context_frames,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,
        )

        if s_text > 1.0:
            context_uncond = torch.zeros_like(Y_t)
            context_uncond[:, :K] = first_frames[:, :K] * ct2 + (1 - ct2) * torch.randn_like(
                first_frames[:, :K]
            )

            velocity_pred_uncond, lambda_ins_pred_uncond, _, _ = forward_fn(
                x_t=Y_t,
                time=t,
                cond_t=uc,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )
            velocity_pred = velocity_pred * s_text + velocity_pred_uncond * (1.0 - s_text)
            if s_ins != 1.0:
                lambda_ins_pred = lambda_ins_pred**s_ins * lambda_ins_pred_uncond ** (1 - s_ins)
        lambda_ins_pred[(Y_t == padding_index)[..., 0, 0, 0]] = 0.0

        # Y_t[:, 1:] = Y_t[:, 1:] + h_flow[:, 1:, None, None, None] * velocity_pred[:, 1:]
        Y_t = Y_t + h_flow[:, :, None, None, None] * velocity_pred

        # Compute poisson insert counts (lambda * h per position)
        ins_t = torch.max(current_t, dim=1, keepdim=True).values
        kappa_t = (ins_t - ins_start).clip(0, 1 - ins_start) / (
            1 - ins_start
        )  # 0 from 1 to ins_start, then linear up to 1.

        d_kappa = torch.full_like(h_flow, 1 / (1 - ins_start))
        d_kappa[(ins_t < ins_start)[:, 0]] = 0

        h_ins = torch.where((ins_t + h < 1.0), torch.ones_like(h) / num_steps, torch.zeros_like(h))

        ratio = torch.where(current_t < ins_start, torch.zeros_like(h), d_kappa / (1 - kappa_t))
        # ratio = torch.where(current_t < ins_start, torch.zeros_like(h), torch.ones_like(h))
        h_lamb = lambda_ins_pred * h_ins * ratio
        h_lamb[(ins_t < ins_start)[:, 0], :] = 0.0

        prob_insertion = h_lamb
        # prob_insertion = 1 - torch.exp(-h_lamb)

        # Compute expected insertions per video before sampling
        expected_inserts = lambda_ins_pred.sum(dim=1)  # shape [B]
        all_expected_lengths.append((expected_inserts + M_t.sum(1)).tolist())

        # Either one insertion occurs or no insertion occurs
        insert_counts = (torch.rand_like(prob_insertion) < prob_insertion).int()

        total_inserts = total_inserts + insert_counts.sum(dim=1)
        num_inserted = insert_counts.sum().item()

        # advance time
        current_t = (current_t + h_flow).clip(0, 1)

        if num_inserted > 0:
            insertions = insert_counts
            num_insertions = insert_counts.sum(1)

            # generate frames to insert
            inserting_frames = torch.randn((B, num_insertions.max(), H, W, C), device=device)
            inserting_mask = (
                torch.arange(num_insertions.max(), device=device)[None].repeat(B, 1) < num_insertions[:, None]
            )

            real_frames_pos = torch.arange(num_frames, device=device)[None].repeat(B, 1)
            real_frames_pos[:, 1:] = real_frames_pos[:, 1:] + insertions.cumsum(1)[:, :-1]

            # Now find the positions of the inserted frames.
            all_frames_indices = torch.arange(num_frames + num_insertions.max(), device=device)
            insert_indices_oneh = 1 - (
                real_frames_pos[:, :, None] == all_frames_indices[None, None, :].repeat(B, 1, 1)
            ).sum(1)

            expanded_frames = torch.full(
                (B, num_frames + num_insertions.max(), H, W, C), -1, dtype=torch.float32, device=device
            )
            expanded_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )
            expanded_mask = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.bool, device=device
            )
            expanded_ins_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )

            batch_idx = torch.arange(B, device=device)[:, None]
            expanded_frames[batch_idx, real_frames_pos, :, :, :] = Y_t
            expanded_mask[batch_idx, real_frames_pos] = M_t
            expanded_times[batch_idx, real_frames_pos] = current_t
            expanded_ins_times[batch_idx, real_frames_pos] = insert_time_map

            I = insert_indices_oneh.sum(dim=1)[0]  # number of insertions per batch (same for all)
            insert_frames_pos = (
                insert_indices_oneh.int().topk(I, dim=1).indices.sort(dim=1).values
            )  # shape [B, I]

            expanded_frames[batch_idx, insert_frames_pos, :, :, :] = inserting_frames
            expanded_mask[batch_idx, insert_frames_pos] = inserting_mask
            expanded_ins_times[batch_idx, insert_frames_pos] = step

            # everything is in interleaved as needed, now we can clip back to keep the same length.
            Y_t = expanded_frames[:, :num_frames]
            M_t = expanded_mask[:, :num_frames]
            insert_time_map = expanded_ins_times[:, :num_frames]

            current_t = expanded_times[:, :num_frames]

        Y_t = torch.where(
            M_t[:, :, None, None, None],
            Y_t,
            torch.full_like(Y_t, padding_index),
        )

        step += 1

    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())

    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


def scale_snr(t: torch.Tensor, s: float, eps: float = 1e-6) -> torch.Tensor:
    t = t.clamp(eps, 1.0 - eps)
    # s>1 => spend more time near 0 (noisy)
    return t / (t + s * (1.0 - t))


@torch.inference_mode()
def vanilla_sample_flowception_t2v(
    first_frames,
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
    snr_shift: float = 1.0,
):
    """
    Fixes:
      1) DO NOT zero lambda by (Y_t == padding_index) pixel values -> use mask M_t
      2) Correct per-sample insertion placement (no I = ...[0] bug)
      3) Safer Bernoulli thinning: p = 1 - exp(-Lambda)  (Lambda = mean inserts over the step)
      4) Enforce max_inserts budget per sample without silently dropping other batch elements
    """

    # --- shapes ---
    H, W, C = first_frames.shape[-3:]
    B = batch_size
    num_frames = input_length

    # --- init sequence (NHWC latents layout as in your code) ---
    Y_t = torch.full((B, num_frames, H, W, C), padding_index, device=device, dtype=torch.float32)
    M_t = torch.zeros((B, num_frames), device=device, dtype=torch.bool)
    dt_u = torch.full((B, num_frames), 1.0 / num_steps, device=device, dtype=torch.float32)

    current_u = torch.zeros((B, num_frames), device=device, dtype=torch.float32)

    K = 0

    rand_len = max(0, min(start_frames, num_frames - K))
    if rand_len > 0:
        Y_t[:, K : K + rand_len] = torch.randn_like(Y_t[:, K : K + rand_len])
        M_t[:, K : K + rand_len] = True

    insert_time_map = torch.full((B, num_frames), -1.0, dtype=torch.float32, device=device)
    insert_time_map[M_t] = 0.0

    Y_T_list, M_t_list = [], []
    total_inserts = torch.zeros(B, device=device, dtype=torch.float32)
    all_expected_lengths = []

    # context always empty in your current usage
    context_full = torch.zeros_like(Y_t)

    for step in tqdm.tqdm(range(2 * num_steps)):
        t = scale_snr(current_u, s=snr_shift)  # [B,L]
        u_next = (current_u + dt_u).clamp(0.0, 1.0)  # [B,L]
        t_next = scale_snr(u_next, s=snr_shift)  # [B,L]

        # flow step size is in denoising time
        h_flow = torch.where(current_u < 1.0, (t_next - t), torch.zeros_like(t))

        # --- forward ---
        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,
        )

        # --- text CFG (your existing logic preserved) ---
        if s_text > 1.0 and uc is not None:
            t_new = (t - s_offset * torch.rand((1,), device=t.device)).clamp(0, 1)
            t_uncond = torch.where(t == 1, t, t_new)
            t_ratio = (t_uncond / t.clamp_min(1e-4))[:, :, None, None, None].to(Y_t.dtype)

            # your noise injection variant
            Y2 = Y_t + (1.0 - t_ratio) * torch.randn_like(Y_t)
            context_uncond = torch.zeros_like(context_full)

            vel_u, lam_u, _, _ = forward_fn(
                x_t=Y2,
                time=t,
                cond_t=uc,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )

            velocity_pred = velocity_pred * s_text + vel_u * (1.0 - s_text)

            if s_ins != 1.0:
                # geometric mixing in rate-space (your choice kept)
                lambda_ins_pred = lambda_ins_pred**s_ins * lam_u ** (1.0 - s_ins)

        # ✅ FIX #1: mask lambda using M_t (NOT pixel equality vs padding_index)
        lambda_ins_pred = lambda_ins_pred.masked_fill(~M_t, 0.0)

        # --- flow update ---
        Y_t = Y_t + h_flow[:, :, None, None, None] * velocity_pred

        ins_u = torch.max(current_u, dim=1, keepdim=True).values  # [B,1]
        kappa_u = (ins_u - ins_start).clamp(0.0, 1.0 - ins_start) / (1.0 - ins_start)

        d_kappa = torch.full_like(current_u, 1.0 / (1.0 - ins_start))
        d_kappa[(ins_u < ins_start)[:, 0]] = 0.0

        # inserts happen with uniform du (not warped dt)
        h_ins = torch.where(
            (ins_u + dt_u[:, :1] < 1.0), torch.ones_like(current_u) / num_steps, torch.zeros_like(current_u)
        )

        ratio = torch.where(current_u < ins_start, torch.zeros_like(current_u), d_kappa / (1.0 - kappa_u))
        Lambda = lambda_ins_pred * h_ins * ratio
        Lambda[(ins_u < ins_start)[:, 0], :] = 0.0
        Lambda = Lambda.clamp_min(0.0)

        prob = (1.0 - torch.exp(-Lambda)).clamp(0.0, 1.0)
        insert_counts = (torch.rand_like(prob) < prob).to(torch.int32)

        # advance insertion clock for existing slots (uniform)
        current_u = torch.where(current_u < 1.0, u_next, current_u)

        # update totals
        # total_inserts = total_inserts + insert_counts.sum(dim=1).to(total_inserts.dtype)
        num_inserted = int(insert_counts.sum().item())

        # --- perform insertion (interleave) ---
        if num_inserted > 0:
            num_insertions = insert_counts.sum(dim=1).to(torch.int64)  # [B]
            Imax = int(num_insertions.max().item())
            if Imax > 0:
                Lnew = num_frames + Imax

                # frames to insert (per-sample padded to Imax)
                inserting_frames = torch.randn((B, Imax, H, W, C), device=device, dtype=Y_t.dtype)
                inserting_mask = (
                    torch.arange(Imax, device=device)[None, :] < num_insertions[:, None]
                )  # [B,Imax]

                # compute where real frames land after inserting
                real_frames_pos = torch.arange(num_frames, device=device)[None, :].repeat(
                    B, 1
                )  # [B,num_frames]
                # shift positions after each slot by cumulative inserts before it
                real_frames_pos[:, 1:] = real_frames_pos[:, 1:] + insert_counts.cumsum(dim=1)[:, :-1].to(
                    torch.int64
                )

                # allocate expanded buffers
                expanded_frames = torch.full(
                    (B, Lnew, H, W, C), padding_index, device=device, dtype=Y_t.dtype
                )
                expanded_mask = torch.zeros((B, Lnew), device=device, dtype=torch.bool)
                expanded_times = torch.zeros((B, Lnew), device=device, dtype=current_u.dtype)
                expanded_ins_times = torch.full((B, Lnew), -1.0, device=device, dtype=insert_time_map.dtype)

                bidx = torch.arange(B, device=device)[:, None]

                # place existing real frames
                expanded_frames[bidx, real_frames_pos] = Y_t
                expanded_mask[bidx, real_frames_pos] = M_t
                expanded_times[bidx, real_frames_pos] = current_u
                expanded_ins_times[bidx, real_frames_pos] = insert_time_map

                # ✅ FIX #2: compute insertion positions per sample correctly
                insert_mask = torch.ones((B, Lnew), device=device, dtype=torch.bool)
                insert_mask.scatter_(1, real_frames_pos, False)  # True where an inserted slot can go

                # rank of each insertion slot (0..I_slots-1) per row
                insert_rank = (
                    insert_mask.cumsum(dim=1) - 1
                )  # [-1 for non-insert early, but we'll gate with insert_mask]

                # keep only first num_insertions[b] slots for each sample
                keep_insert = insert_mask & (insert_rank < num_insertions[:, None])

                # scatter the inserted frames by (b, pos) -> inserting_frames[b, rank]
                bi, pos = torch.where(keep_insert)
                if bi.numel() > 0:
                    r = insert_rank[bi, pos].to(torch.int64)  # [N]
                    expanded_frames[bi, pos] = inserting_frames[bi, r]
                    expanded_mask[bi, pos] = True
                    # start inserted frames at time 0 (consistent with your previous default)
                    # expanded_times[bi, pos] stays 0
                    expanded_ins_times[bi, pos] = float(step)

                # clip back to fixed length
                Y_t = expanded_frames[:, :num_frames]
                M_t = expanded_mask[:, :num_frames]
                current_u = expanded_times[:, :num_frames]
                insert_time_map = expanded_ins_times[:, :num_frames]

        # apply padding for masked slots
        Y_t = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y_t, padding_index))

    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())
    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


@torch.inference_mode()
def vanilla_sample_flowception_prescribed(
    first_frames,
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
    insertion_rule="learned",  # "learned" | "random" | "middle" | "right"
):
    H, W, C = first_frames.shape[-3:]
    B = batch_size

    # support for more than one context frame.
    Y_t = torch.full((B, input_length, H, W, C), padding_index, device=device, dtype=torch.float32)
    M_t = torch.zeros(B, input_length, device=device, dtype=torch.bool)

    K = 0
    if first_frames is not None:
        # first_frames: [B, K, H, W, C]  (your caller passes BCHWT elsewhere; keep consistent with this function)
        K = first_frames.shape[1]
        K = min(K, input_length)
        Y_t[:, :K] = first_frames[:, :K]
        M_t[:, :K] = True

    rand_len = max(0, min(start_frames, input_length - K))
    if rand_len > 0:
        Y_t[:, K : K + rand_len] = torch.randn_like(Y_t[:, K : K + rand_len])
        M_t[:, K : K + rand_len] = True

    insert_time_map = torch.full((B, input_length), -1.0, dtype=torch.float32, device=device)
    insert_time_map[M_t] = 0

    dt = torch.full((B, input_length), 1 / num_steps, device=device)

    current_t = torch.zeros(B, input_length, dtype=torch.float32, device=device)

    Y_T_list = []
    M_t_list = []

    step = 0
    num_frames = input_length
    total_inserts = torch.zeros(B, device=device)
    all_expected_lengths = []

    context_full = torch.zeros_like(Y_t)
    context_full[:, :K] = first_frames[:, :K]

    for step in tqdm.tqdm(range(2 * num_steps)):
        # keep h_flow: it already zeroes updates where t >= 1

        t = current_t
        h = dt
        h_flow = torch.where(t + h < 1.0, h, torch.zeros_like(h))

        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            # context_frames=context_frames,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,
        )

        if s_text > 1.0:
            t_new = (t - s_offset * torch.rand((1,), device=t.device)).clip(0, 1)
            t_uncond = torch.where(t == 1, t, t_new)
            t_ratio = (t_uncond / t.clip(1e-4, None))[:, :, None, None, None].to(Y_t.dtype)
            # Y2 = t_ratio * Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
            Y2 = Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
            Y2[:, 0] = 0 * Y_t[:, 0]
            context_uncond = torch.zeros_like(context_full)

            velocity_pred_uncond, lambda_ins_pred_uncond, _, _ = forward_fn(
                # x_t=Y_t,
                x_t=Y2,
                time=t,
                cond_t=uc,
                # context_frames=context_frames,
                # context_frames=context_full,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )
            velocity_pred = velocity_pred * s_text + velocity_pred_uncond * (1.0 - s_text)
            if s_ins != 1.0:
                lambda_ins_pred = lambda_ins_pred**s_ins * lambda_ins_pred_uncond ** (1 - s_ins)
        lambda_ins_pred[(Y_t == padding_index)[..., 0, 0, 0]] = 0.0

        Y_t[:, 1:] = Y_t[:, 1:] + h_flow[:, 1:, None, None, None] * velocity_pred[:, 1:]

        # d_kappa = torch.full_like(h_flow, 1 / (1 - ins_start))
        # d_kappa[(ins_t < ins_start)[:, 0]] = 0

        # h_ins = torch.where((ins_t + h < 1.0), torch.ones_like(h) / num_steps, torch.zeros_like(h))

        # prob_insertion = h_lamb
        # # prob_insertion = 1 - torch.exp(-h_lamb)

        # # Either one insertion occurs or no insertion occurs
        # insert_counts = (torch.rand_like(prob_insertion) < prob_insertion).int()

        if insertion_rule == "learned":
            # --- your original Poisson-ish logic ---
            # Compute poisson insert counts (lambda * h per position)
            ins_t = torch.max(current_t, dim=1, keepdim=True).values
            kappa_t = (ins_t - ins_start).clip(0, 1 - ins_start) / (
                1 - ins_start
            )  # 0 from 1 to ins_start, then linear up to 1.

            d_kappa = torch.full_like(h_flow, 1 / (1 - ins_start))
            d_kappa[(ins_t < ins_start)[:, 0]] = 0

            h_ins = torch.where(
                (ins_t + h < 1.0),
                torch.ones_like(h) / num_steps,
                torch.zeros_like(h),
            )

            ratio = torch.where(current_t < ins_start, torch.zeros_like(h), d_kappa / (1 - kappa_t))
            h_lamb = lambda_ins_pred * h_ins * ratio
            h_lamb[(ins_t < ins_start)[:, 0], :] = 0.0

            prob_insertion = h_lamb
            # prob_insertion = 1 - torch.exp(-h_lamb)

            # --- Expected inserts (for logging) ---
            expected_inserts = lambda_ins_pred.sum(dim=1)  # shape [B]
            all_expected_lengths.append((expected_inserts + M_t.sum(1)).tolist())

            # Either one insertion occurs or no insertion occurs
            insert_counts = (torch.rand_like(prob_insertion) < prob_insertion).int()

        else:
            B, num_frames = M_t.shape
            device = M_t.device

            ins_t = torch.max(current_t, dim=1, keepdim=True).values  # [B,1]
            can_insert_mask = ins_t + h < 1.0  # [B,L] bool

            # How many real frames (non-padding) we currently have?
            current_real = M_t.sum(dim=1).float()  # [B]
            # Target: full sequence
            target_real = torch.full_like(current_real, float(input_length))
            remaining_needed = (target_real - current_real).clamp(min=0.0)  # [B]

            # Number of steps left in the loop (we run 2*num_steps total)
            total_steps = 2 * num_steps
            remaining_steps = total_steps - (step + 1)
            remaining_steps = max(remaining_steps, 1)

            expected_inserts_this_step = remaining_needed / float(remaining_steps)  # [B]

            insert_counts = torch.zeros((B, num_frames), device=device, dtype=torch.int32)

            if insertion_rule == "random":
                p = torch.zeros_like(current_real)
                valid = current_real > 0
                p[valid] = expected_inserts_this_step[valid] / current_real[valid].clamp(min=1.0)
                p = p.clamp(0.0, 1.0)  # sanity

                # Broadcast p over positions and mask by M_t (only real frames).
                p_frame = (p[:, None]).expand(B, num_frames)

                rand = torch.rand((B, num_frames), device=device)
                insert_mask = (rand < p_frame) & M_t  # only after real frames
                insert_counts = insert_mask.int()
                insert_counts = insert_counts * can_insert_mask.int()

            elif insertion_rule == "middle":
                for b in range(B):
                    if remaining_needed[b] <= 0:
                        continue

                    # probability of inserting *one* frame at this step
                    p_b = expected_inserts_this_step[b].clamp(0.0, 1.0)

                    if torch.rand(1, device=device) < p_b:
                        real_idx = torch.where(M_t[b])[0]
                        if real_idx.numel() == 0:
                            continue
                        mid_pos = real_idx[real_idx.numel() // 2]
                        insert_counts[b, mid_pos] = 1

                insert_counts = insert_counts * can_insert_mask.int()

            elif insertion_rule == "right":
                for b in range(B):
                    if remaining_needed[b] <= 0:
                        continue
                    p_b = expected_inserts_this_step[b].clamp(0.0, 1.0)
                    if torch.rand(1, device=device) < p_b:
                        real_idx = torch.where(M_t[b])[0]
                        if real_idx.numel() == 0:
                            continue
                        last_pos = real_idx[-1]  # append to the right
                        insert_counts[b, last_pos] = 1
                insert_counts = insert_counts * can_insert_mask.int()

            else:
                raise ValueError(f"Unknown insertion_rule: {insertion_rule}")

            # logging to keep all_expected_lengths comparable (using the
            # *target* number of real frames).
            all_expected_lengths.append((remaining_needed + current_real).tolist())

        total_inserts = total_inserts + insert_counts.sum(dim=1)
        num_inserted = insert_counts.sum().item()

        # advance time
        current_t = (current_t + h_flow).clip(0, 1)

        if num_inserted > 0:
            insertions = insert_counts
            num_insertions = insert_counts.sum(1)

            # generate frames to insert
            inserting_frames = torch.randn((B, num_insertions.max(), H, W, C), device=device)
            inserting_mask = (
                torch.arange(num_insertions.max(), device=device)[None].repeat(B, 1) < num_insertions[:, None]
            )

            real_frames_pos = torch.arange(num_frames, device=device)[None].repeat(B, 1)
            real_frames_pos[:, 1:] = real_frames_pos[:, 1:] + insertions.cumsum(1)[:, :-1]

            # Now find the positions of the inserted frames.
            all_frames_indices = torch.arange(num_frames + num_insertions.max(), device=device)
            insert_indices_oneh = 1 - (
                real_frames_pos[:, :, None] == all_frames_indices[None, None, :].repeat(B, 1, 1)
            ).sum(1)

            expanded_frames = torch.full(
                (B, num_frames + num_insertions.max(), H, W, C), -1, dtype=torch.float32, device=device
            )
            expanded_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )
            expanded_mask = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.bool, device=device
            )
            expanded_ins_times = torch.full(
                (B, num_frames + num_insertions.max()), 0.0, dtype=torch.float32, device=device
            )

            batch_idx = torch.arange(B, device=device)[:, None]
            expanded_frames[batch_idx, real_frames_pos, :, :, :] = Y_t
            expanded_mask[batch_idx, real_frames_pos] = M_t
            expanded_times[batch_idx, real_frames_pos] = current_t
            expanded_ins_times[batch_idx, real_frames_pos] = insert_time_map

            I = insert_indices_oneh.sum(dim=1)[0]  # number of insertions per batch (same for all)
            insert_frames_pos = (
                insert_indices_oneh.int().topk(I, dim=1).indices.sort(dim=1).values
            )  # shape [B, I]

            expanded_frames[batch_idx, insert_frames_pos, :, :, :] = inserting_frames
            expanded_mask[batch_idx, insert_frames_pos] = inserting_mask
            expanded_ins_times[batch_idx, insert_frames_pos] = step

            # everything is in interleaved as needed, now we can clip back to keep the same length.
            Y_t = expanded_frames[:, :num_frames]
            M_t = expanded_mask[:, :num_frames]
            insert_time_map = expanded_ins_times[:, :num_frames]

            current_t = expanded_times[:, :num_frames]

        Y_t = torch.where(
            M_t[:, :, None, None, None],
            Y_t,
            torch.full_like(Y_t, padding_index),
        )

        step += 1

    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())

    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


@torch.inference_mode()
def vanilla_sample_interp_flowception(
    first_frames,
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
    start_frames=0,  # no random start for interpolation
    max_inserts=20,
    ins_start=0.3,
    uc=None,
    s_text=1.0,
    s_img=0.0,
    s_offset=0.05,
    s_ins=1.0,
    num_uncond=1,
):
    H, W, C = first_frames.shape[-3:]
    B = batch_size
    L = input_length

    # ==== I2V buffer setup (unchanged structure) ====
    Y_t = torch.full((B, L, H, W, C), padding_index, device=device, dtype=torch.float32)
    M_t = torch.zeros(B, L, device=device, dtype=torch.bool)

    # --- Context initialization: use all provided context frames, no random start
    K = 0
    if first_frames is not None:
        K = min(first_frames.shape[1], L)
        Y_t[:, :K] = first_frames[:, :K]
        M_t[:, :K] = True

    # persistent mask of ORIGINAL context slots; anchors for flow (frozen)
    fixed_ctx_mask = torch.zeros_like(M_t)
    if K > 0:
        fixed_ctx_mask[:, :K] = True

    # defined aligned context stream.
    context_aligned = torch.zeros((B, L, H, W, C), device=device, dtype=torch.float32)
    context_aligned[fixed_ctx_mask] = Y_t[fixed_ctx_mask]

    insert_time_map = torch.full((B, L), -1.0, dtype=torch.float32, device=device)
    insert_time_map[M_t] = 0.0

    dt = torch.full((B, L), 1.0 / num_steps, device=device)
    current_t = torch.zeros(B, L, dtype=torch.float32, device=device)

    # make context frames live at t=1 (and they stay frozen because h_flow_eff=0 on anchors)
    # current_t[fixed_ctx_mask] = 0.0
    current_t[fixed_ctx_mask] = 0.0
    insert_time_map[fixed_ctx_mask] = 0.0  # optional: clearer bookkeeping

    Y_T_list, M_t_list = [], []
    num_frames = L
    total_inserts = torch.zeros(B, device=device)
    all_expected_lengths = []

    def cap_by_max_inserts(prob, total_inserts_vec):
        if max_inserts is None:
            return prob
        can_insert = total_inserts_vec[:, None] < max_inserts
        return prob * can_insert

    keep_emb = True
    if keep_emb and isinstance(keep_emb, list):
        uc["class_labels"][1] = cond_t["class_labels"][1].clone()

    for step in tqdm.tqdm(range(2 * num_steps)):
        t = current_t
        h = dt
        # same as I2V, but we'll freeze anchors via h_flow_eff
        h_flow = torch.where(t + h < 1.0, h, torch.zeros_like(h))
        h_flow_eff = torch.where(fixed_ctx_mask, torch.zeros_like(h_flow), h_flow)

        velocity_pred, lambda_ins_pred, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            # context_frames=None, #context_frames,
            context_frames=context_aligned,
            model=model,
            frame_mask=M_t,
        )
        if s_text > 1.0:
            # slots where we are allowed to alter x_t for guidance
            guide_mask = M_t & ~fixed_ctx_mask  # active & not context anchors
            # optional: also require still flowing this step
            guide_mask = guide_mask & (h_flow > 0)  # keeps guidance off when t is saturated

            # sample per-batch, per-slot offset
            eps = torch.rand_like(t)  # (B, L)
            t_shift = (t - s_offset * eps).clamp(0.0, 1.0)
            t_uncond = torch.where(guide_mask, t_shift, t)  # anchors & non-active keep their t

            # build broadcast masks
            guide5 = guide_mask[:, :, None, None, None]
            t_ratio = (t_uncond / t.clamp_min(1e-4))[:, :, None, None, None]

            # Y2 = Y_t + s_offset * (4 * t * (1-t))[:, :, None, None, None].pow(0.3) * noise
            # Y2 = torch.where(guide5, Y2, Y_t)

            velocity_pred_uncond, lambda_ins_pred_uncond = (
                torch.zeros_like(velocity_pred),
                torch.zeros_like(lambda_ins_pred),
            )
            for nu in range(num_uncond):
                t_new = (t - s_offset * torch.rand((1,), device=t.device)).clip(0, 1)
                t_uncond = torch.where(t == 1, t, t_new)
                t_ratio = (t_uncond / t.clip(1e-4, None))[:, :, None, None, None].to(Y_t.dtype)
                # Y2 = t_ratio * Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
                Y2 = Y_t + (1 - t_ratio) * torch.randn_like(Y_t)

                Y2 = Y2 * M_t[:, :, None, None, None] + Y_t * (~M_t[:, :, None, None, None])
                # make sure context frames are left untouched.
                Y2 = torch.where(guide5, Y2, Y_t)
                Y2[:, 0] = Y_t[:, 0]

                # t_new = (t - s_offset * torch.rand_like(t)).clip(0, 1)
                # t_uncond = torch.where(t == 1, t, t_new)

                velocity_pred_uncond0, lambda_ins_pred_uncond0, _, _ = forward_fn(
                    # x_t=Y_t,
                    # x_t=Y_t,
                    x_t=Y2,
                    time=t,  # _new,
                    # cond_t=uc,
                    cond_t=cond_t,
                    context_frames=context_aligned,
                    model=model,
                    frame_mask=M_t,
                )
                velocity_pred_uncond += velocity_pred_uncond0
                lambda_ins_pred_uncond += lambda_ins_pred_uncond0
            velocity_pred_uncond = velocity_pred_uncond / num_uncond
            lambda_ins_pred_uncond = lambda_ins_pred_uncond / num_uncond

            velocity_pred = velocity_pred * s_text + velocity_pred_uncond * (1.0 - s_text)
        # if s_ins != 1.0:
        #     lambda_ins_pred = lambda_ins_pred**s_ins * lambda_ins_pred_uncond**(1-s_ins)
        if s_ins != 1.0:
            L = M_t.shape[1]
            ar = torch.arange(L, device=M_t.device)
            last_idx = (M_t * ar).amax(dim=1)  # (B,)
            is_last_active = ar.unsqueeze(0).eq(last_idx.unsqueeze(1))  # (B,L)
            # active positions excluding the rightmost active one
            guide_mask_rates = M_t & ~is_last_active  # (B,L)

            # stable mix (log-space), then apply only where allowed
            eps = 1e-8
            lam_mix = torch.exp(
                s_ins * torch.log(lambda_ins_pred.clamp_min(eps))
                + (1.0 - s_ins) * torch.log(lambda_ins_pred_uncond.clamp_min(eps))
            )
            lambda_ins_pred = torch.where(guide_mask_rates, lam_mix, lambda_ins_pred)

        # ===== flow update (freeze only anchors) =====
        Y_t[:, 1:] = Y_t[:, 1:] + h_flow_eff[:, 1:, None, None, None] * velocity_pred[:, 1:]
        # ===== insertion schedule (I2V) =====
        ins_t = current_t.max(dim=1, keepdim=True).values
        kappa_t = (ins_t - ins_start).clamp(0, 1 - ins_start) / (1 - ins_start)
        d_kappa = torch.full_like(h_flow, 1.0 / (1 - ins_start))
        d_kappa[(ins_t < ins_start)[:, 0]] = 0.0

        h_ins = torch.where(ins_t + h < 1.0, torch.ones_like(h) / num_steps, torch.zeros_like(h))

        # === minimal change: rates only on active slots ===
        # (I2V used padding-index check; this is equivalent and simpler here)
        lambda_ins_pred = lambda_ins_pred.masked_fill(~M_t, 0.0)

        ratio = torch.where(current_t < ins_start, torch.zeros_like(h), d_kappa / (1 - kappa_t))
        h_lamb = lambda_ins_pred * h_ins * ratio
        h_lamb[(ins_t < ins_start)[:, 0], :] = 0.0

        # === minimal change: allow insert on ANY active except the rightmost active one ===
        ar = torch.arange(num_frames, device=M_t.device)
        last_idx = (M_t * ar).amax(dim=1)  # [B]
        allowed_insert = M_t & (ar.unsqueeze(0) < last_idx.unsqueeze(1))
        allowed_insert[:, -1] = False

        prob_insertion = h_lamb * allowed_insert.float()
        prob_insertion = cap_by_max_inserts(prob_insertion, total_inserts)

        expected_inserts = (h_lamb * allowed_insert.float()).sum(dim=1)
        all_expected_lengths.append((expected_inserts + M_t.sum(1)).tolist())

        # === Bernoulli per-site (I2V) ===
        insert_counts = (torch.rand_like(prob_insertion) < prob_insertion).int()
        ins_mask = insert_counts.bool() & allowed_insert

        total_inserts = total_inserts + insert_counts.sum(dim=1)
        num_inserted = int(insert_counts.sum().item())

        # advance time (freeze anchors)
        # current_t = (current_t + h_flow_eff).clamp(0, 1)

        # we should actually advance time for context frames.
        current_t = (current_t + h_flow).clamp(0, 1)

        # ===== interleave to the RIGHT (identical to I2V, plus fixed_ctx propagation) =====
        if num_inserted > 0:
            num_insertions = ins_mask.sum(1)  # [B]
            max_I = int(num_insertions.max().item())
            inserting_frames = torch.randn((B, max_I, H, W, C), device=device)
            inserting_mask = torch.arange(max_I, device=device)[None, :] < num_insertions[:, None]

            real_pos = torch.arange(num_frames, device=device)[None, :].repeat(B, 1)
            real_pos[:, 1:] = real_pos[:, 1:] + ins_mask.cumsum(1)[:, :-1]  # shift to the right (I2V)

            all_indices = torch.arange(num_frames + max_I, device=device)
            insert_indices_oneh = 1 - (real_pos[:, :, None] == all_indices[None, None, :]).sum(1)

            expanded_frames = torch.full(
                (B, num_frames + max_I, H, W, C), padding_index, dtype=torch.float32, device=device
            )
            expanded_times = torch.full((B, num_frames + max_I), 0.0, dtype=torch.float32, device=device)
            expanded_mask = torch.zeros((B, num_frames + max_I), dtype=torch.bool, device=device)
            expanded_ins_times = torch.full((B, num_frames + max_I), -1.0, dtype=torch.float32, device=device)
            expanded_fixed_ctx = torch.zeros((B, num_frames + max_I), dtype=torch.bool, device=device)

            bidx = torch.arange(B, device=device)[:, None]
            expanded_frames[bidx, real_pos] = Y_t
            expanded_mask[bidx, real_pos] = M_t
            expanded_times[bidx, real_pos] = current_t
            expanded_ins_times[bidx, real_pos] = insert_time_map
            expanded_fixed_ctx[bidx, real_pos] = fixed_ctx_mask  # propagate anchors

            I = insert_indices_oneh.sum(dim=1)[0]
            insert_frames_pos = insert_indices_oneh.int().topk(I, dim=1).indices.sort(dim=1).values

            expanded_frames[bidx, insert_frames_pos] = inserting_frames
            expanded_mask[bidx, insert_frames_pos] = inserting_mask
            expanded_ins_times[bidx, insert_frames_pos] = float(step)
            # anchors remain False at newly inserted positions

            # clip back to logical length (I2V)
            Y_t = expanded_frames[:, :num_frames]
            M_t = expanded_mask[:, :num_frames]
            insert_time_map = expanded_ins_times[:, :num_frames]
            current_t = expanded_times[:, :num_frames]
            fixed_ctx_mask = expanded_fixed_ctx[:, :num_frames]

            # ensure original context always stays real
            M_t = M_t | fixed_ctx_mask

            context_aligned = torch.zeros(B, L, H, W, C, device=device, dtype=Y_t.dtype)
            context_aligned[fixed_ctx_mask] = Y_t[fixed_ctx_mask]  # or your pre-encoded clean anchors

        # keep padding slots equal to padding_index
        Y_t = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y_t, padding_index))

    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())
    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


@torch.inference_mode()
def vanilla_sample_flow_only_fullseq(
    first_frames,  # [B, K, H, W, C] (same convention as your flowception sampler)
    model,
    forward_fn,
    num_steps,
    cond_t,
    context_frames,  # unused here; kept for signature parity
    batch_size=1,
    input_length=32,
    device="cuda",
    padding_index=0,
    num_snapshots=20,  # kept for signature parity
    start_frames=None,  # unused; fixed-length so no staged growth
    max_inserts=None,  # unused
    ins_start=None,  # unused
    uc=None,
    s_text=2.0,
    s_img=0.0,  # unused in flow-only
    s_offset=0.1,  # same as your guided path
    s_ins=2.5,  #
    **kwargs,
):
    # ---- shapes / init (match your original conventions) ----
    if first_frames is None:
        raise ValueError("flow-only fullseq sampler needs at least one context frame.")

    B = batch_size
    K = min(first_frames.shape[1], input_length)
    H, W, C = first_frames.shape[-3:]  # follow your H,W,C naming (compatible with your code)

    # Video buffer & full mask (ALL TRUE — full sequence generation)
    Y_t = torch.full((B, input_length, H, W, C), padding_index, device=device, dtype=first_frames.dtype)
    M_t = torch.ones(B, input_length, device=device, dtype=torch.bool)

    # place context frames
    Y_t[:, :K] = first_frames[:, :K]

    # init remaining frames with noise (fixed length, no growth)
    if K < input_length:
        Y_t[:, K:] = torch.randn_like(Y_t[:, K:])

    # per-position time and step
    dt = torch.full((B, input_length), 1.0 / float(num_steps), device=device, dtype=Y_t.dtype)
    t = torch.zeros(B, input_length, device=device, dtype=Y_t.dtype)

    # static context fed to the model (freeze first K frames)
    context_full = torch.zeros_like(Y_t)
    context_full[:, :K] = first_frames[:, :K]

    # outputs to mirror original function
    Y_T_list = []
    M_t_list = []
    insert_time_map = torch.zeros(B, input_length, device=device, dtype=torch.float32)  # all zeros
    all_expected_lengths = []  # list of per-step lists (length B), always == input_length

    # ---- flow-only denoising loop (no insert logic) ----
    for _ in tqdm.tqdm(range(num_steps)):
        # forward
        vel, _, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,  # full True
        )

        # cfg on velocity (your same trick with t_offset)
        if s_text > 1.0 and uc is not None:
            t_new = (t - s_offset * torch.rand((1,), device=t.device)).clip(0, 1)
            t_uncond = torch.where(t == 1, t, t_new)
            t_ratio = (t_uncond / t.clip(1e-4, None))[:, :, None, None, None].to(Y_t.dtype)
            # Y2 = Y_t * t_ratio + (1 - t_ratio) * torch.randn_like(Y_t)
            Y2 = Y_t + (1 - t_ratio) * torch.randn_like(Y_t)
            # Y2 = t_ratio * Y_t + (t_ratio <0.9) * torch.randn_like(Y_t)
            # Y2[:, 0] = Y_t[:, 0]
            Y2[:, 0] = 0 * Y_t[:, 0]
            context_uncond = torch.zeros_like(context_full)

            vel_u, _, _, _ = forward_fn(
                x_t=Y2,
                # time=t,
                time=t,
                cond_t=uc,  # cond_t, #uc,
                # context_frames=context_full,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )
            vel = vel * s_text + vel_u * (1.0 - s_text)

        # update only non-context frames (keep first K frozen)
        if K < input_length:
            Y_t[:, K:] = Y_t[:, K:] + dt[:, K:, None, None, None] * vel[:, K:]

        t = (t + dt).clamp(max=1.0)

        # keep padding convention (mask is all True so this is a no-op, but safe)
        Y_t = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y_t, padding_index))

        # expected length per video is always the fixed sequence length
        all_expected_lengths.append([int(input_length)] * B)

    # final snapshots (your original code appends final states)
    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())

    # exactly the same return signature as vanilla_sample_flowception:
    # Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths
    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths


@torch.inference_mode()
def vanilla_sample_flow_only_fullseq_t2v(
    first_frames,  # [B, K, H, W, C] (same convention as your flowception sampler)
    model,
    forward_fn,
    num_steps,
    cond_t,
    context_frames,  # unused here; kept for signature parity
    batch_size=1,
    input_length=32,
    device="cuda",
    padding_index=0,
    num_snapshots=20,  # kept for signature parity
    start_frames=None,  # unused; fixed-length so no staged growth
    max_inserts=None,  # unused
    ins_start=None,  # unused
    uc=None,
    s_text=2.0,
    s_img=0.0,  # unused in flow-only
    s_offset=0.0,  # same as your guided path
    s_ins=1.0,  #
    **kwargs,
):
    # ---- shapes / init (match your original conventions) ----
    if first_frames is None:
        raise ValueError("flow-only fullseq sampler needs at least one context frame.")

    B = batch_size
    # K = min(first_frames.shape[1], input_length)
    K = 0
    H, W, C = first_frames.shape[-3:]  # follow your H,W,C naming (compatible with your code)

    # Video buffer & full mask (ALL TRUE — full sequence generation)
    Y_t = torch.full((B, input_length, H, W, C), padding_index, device=device, dtype=first_frames.dtype)
    M_t = torch.ones(B, input_length, device=device, dtype=torch.bool)

    # init remaining frames with noise (fixed length, no growth)

    Y_t = torch.randn_like(Y_t)

    # per-position time and step
    dt = torch.full((B, input_length), 1.0 / float(num_steps), device=device, dtype=Y_t.dtype)
    t = torch.zeros(B, input_length, device=device, dtype=Y_t.dtype)

    # static context fed to the model (freeze first K frames)
    context_full = torch.zeros_like(Y_t)

    # outputs to mirror original function
    Y_T_list = []
    M_t_list = []
    insert_time_map = torch.zeros(B, input_length, device=device, dtype=torch.float32)  # all zeros
    all_expected_lengths = []  # list of per-step lists (length B), always == input_length

    # ---- flow-only denoising loop (no insert logic) ----
    for _ in tqdm.tqdm(range(num_steps)):
        # forward
        vel, _, _, _ = forward_fn(
            x_t=Y_t,
            time=t,
            cond_t=cond_t,
            context_frames=context_full,
            model=model,
            frame_mask=M_t,  # full True
        )

        # cfg on velocity (your same trick with t_offset)
        if s_text > 1.0 and uc is not None:
            context_uncond = torch.zeros_like(context_full)

            vel_u, _, _, _ = forward_fn(
                x_t=Y_t,
                # time=t,
                time=t,
                cond_t=uc,  # cond_t, #uc,
                # context_frames=context_full,
                context_frames=context_uncond,
                model=model,
                frame_mask=M_t,
            )
            vel = vel * s_text + vel_u * (1.0 - s_text)

        # update only non-context frames (keep first K frozen)
        if K < input_length:
            Y_t[:, K:] = Y_t[:, K:] + dt[:, K:, None, None, None] * vel[:, K:]

        t = (t + dt).clamp(max=1.0)

        # keep padding convention (mask is all True so this is a no-op, but safe)
        Y_t = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y_t, padding_index))

        # expected length per video is always the fixed sequence length
        all_expected_lengths.append([int(input_length)] * B)

    # final snapshots (your original code appends final states)
    Y_T_list.append(Y_t.clone())
    M_t_list.append(M_t.clone())

    return Y_t, M_t, Y_T_list, M_t_list, insert_time_map, all_expected_lengths

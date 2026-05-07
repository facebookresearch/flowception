import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from accelerate.utils import save_fsdp_model
from helpers.checkpoint import get_fsdp_ckpt_kwargs
from einops import rearrange

from engine.data_classes import TrainTuple
from engine.trainer import Trainer
from engine.utils import (
    append_dims,
    vae_accepts_latent_mask,
    randomly_select_slice,
    get_ema_loss_heatmap_to_wandb,
    update_ema_loss_bins,
    sample_tau_g_edge_equalized,
    logit_normal_weight,
    scale_snr,
    downsample_video,
    GPUBatchVideoAug,
)
from torch.distributions import Beta
from helpers.video_utils.visualization import frames_to_gif, write_frames_ffmpeg
from modules.diffusion.denoiser import build_denoiser_wrapper
from modules.diffusion.denoiser_scaling import CondOTScaling
from modules.diffusion.sigma_sampling import CondOTUniformDiscretization
from modules.flowception.align import compute_insert_counts, left_align_by_mask, left_align_frames
from modules.flowception.helpers import sample_start_frames_interp, pick_latents_after_skip
from modules.flowception.losses import poisson_loss
from modules.flowception.sampling import vanilla_sample_interp_flowception
from modules.flowception.schedulers import get_kappa_scheduler
from modules.sampling.apg import APGGuider


def select_random_frames(x: torch.Tensor, K: int) -> torch.Tensor:
    """
    x: [B, L, C, H, W] video tensor
    K: number of random *middle* frames to add (without replacement)

    returns: [B, K+2, C, H, W] (first + K random middle + last), in order
    """
    B, L, C, H, W = x.shape
    if L == 0:
        raise ValueError("L must be >= 1")
    if L == 1:
        idx = torch.zeros(B, 2, dtype=torch.long, device=x.device)
        return x[torch.arange(B)[:, None], idx, ...]

    M = max(L - 2, 0)
    K_eff = min(K, M)

    if K_eff > 0:
        rand_scores = torch.rand(B, M, device=x.device)
        mid_idx = torch.topk(rand_scores, K_eff, dim=1).indices + 1
    else:
        mid_idx = torch.empty(B, 0, dtype=torch.long, device=x.device)

    first = torch.zeros(B, 1, dtype=torch.long, device=x.device)
    last = torch.full((B, 1), L - 1, dtype=torch.long, device=x.device)

    idx = torch.cat([first, mid_idx, last], dim=1)
    idx, _ = torch.sort(idx, dim=1)

    out = x[torch.arange(B, device=x.device)[:, None], idx, ...]
    return out


class FlowceptionInterpolate(Trainer):
    """Flowception trainer for video interpolation generation.

    Given anchor frames at both ends (and optionally random middle frames),
    iteratively inserts new frames between existing ones using a
    continuous normalizing flow. Training uses a combined velocity prediction
    loss and a Poisson process loss for frame insertion timing.

    Key differences from Flowception (i2v):
      - Context frames are sampled randomly (first, last, + random middle)
      - Context frame dropout is per-frame (P(all K dropped) = cfg_p)
      - Uses sample_start_frames_interp instead of sample_start_frames
    """

    def __init__(
        self,
        cfg,
        accelerator,
        model,
        ema,
        conditioner,
        dataloader,
        val_dataloader,
        optimizer,
        device,
        vae,
        scheduler=None,
        output_dir=None,
        start_epoch=0,
        global_step=0,
        local_step=0,
        logger=None,
        extra_dataloader=None,
    ):
        super().__init__(
            cfg,
            accelerator,
            model,
            ema,
            conditioner,
            dataloader,
            val_dataloader,
            optimizer,
            device,
            vae,
            scheduler,
            output_dir,
            start_epoch,
            global_step,
            local_step,
            logger,
            extra_dataloader=extra_dataloader,
        )

        self.tau_g_sampling = cfg.FLOWCEPTION.TAU_GLOBAL_DIST

        self.discretizer = CondOTUniformDiscretization(sigma_scale=cfg.FRAMEWORK.SIGMA_SCALE)

        denoiser_scaler = CondOTScaling()

        self.denoiser = build_denoiser_wrapper(
            cfg.FRAMEWORK.DENOISER.lower(),
            scaling=denoiser_scaler,
            num_idx=self.timesteps,
            discretization=self.discretizer,
            uncond_gen=cfg.SAMPLER.UNCOND_GEN,
            image_cfg=cfg.SAMPLER.IMAGE_GUIDANCE_SCALE,
            flowception_setup=True,
        ).to(self.accelerator.device, non_blocking=True)

        self.guider = None
        if cfg.SAMPLER.APG.ENABLE:
            self.guider = APGGuider(
                eta=cfg.SAMPLER.APG.ETA,
                momentum=cfg.SAMPLER.APG.MOMENTUM,
                norm_threshold=cfg.SAMPLER.APG.NORM_THRESHOLD,
            )

        self.denoiser.guider = self.guider

        self.temporal_factor = cfg.MODEL.VAE.TEMPORAL_FACTOR

        self.latent_depth = 1 + cfg.DATA.MAX_FRAMES // self.temporal_factor
        self.num_context_frames = cfg.DATA.NUM_CONTEXT_FRAMES

        self.image_guidance_scale = cfg.SAMPLER.IMAGE_GUIDANCE_SCALE
        self.mask_first_frame = cfg.MODEL.VIDEO.RD_MASK_FIRST_FRAME
        self.ema_loss_bins = {}

        self.grad_ema = 0.02
        self.grad_ema_var = 0.0002
        self.grad_ema_beta = 0.99

        self.kappa_scheduler = get_kappa_scheduler(cfg.FLOWCEPTION.KAPPA_SCHEDULER)
        self.num_start_frames = cfg.FLOWCEPTION.NUM_START_FRAMES
        self.padding_index = cfg.DATA.PADDING_INDEX

        self.sampling_fps = cfg.DATA.VIDEO.SAMPLING_FPS
        self.train_image_only = cfg.SOLVER.IMAGE_ONLY

        self.guidance_offset = cfg.FLOWCEPTION.SAMPLING.GUIDANCE_OFFSET
        self.guidance_insertion_scale = cfg.FLOWCEPTION.SAMPLING.GUIDANCE_INS

        self.insertion_rule = cfg.FLOWCEPTION.SAMPLING.INSERTION_RULE
        self.image_loss_weight = cfg.FLOWCEPTION.LOSS.IMG_WEIGHT

        self.poisson_loss_weight = cfg.FLOWCEPTION.LOSS.POISSON_WEIGHT
        self.gpu_aug = GPUBatchVideoAug()
        self.aug_gen = torch.Generator(device=self.accelerator.device)
        if isinstance(cfg.SOLVER.IM_SIZE, (list, tuple)):
            self.target_h, self.target_w = cfg.SOLVER.IM_SIZE
        else:
            self.target_h = self.target_w = cfg.SOLVER.IM_SIZE

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def get_loss_per_bin(self, timesteps, losses):
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        loss_binned = {}
        for start, end in ranges:
            mask = (timesteps >= start) & (timesteps < end)
            avg_loss = losses[mask].mean()
            loss_binned[f"train/loss_t_{start}:{end}"] = avg_loss.detach()
        return loss_binned

    def get_loss_per_frame(self, input, target):
        framewise_mse = (input - target).pow(2).mean((0, 1, 3, 4))
        loss_binned = {}
        for i in range(len(framewise_mse)):
            loss_binned[f"train/per_layer/loss_{i:02d}"] = framewise_mse[i].detach()
        return loss_binned

    def image_loss(self, model, batch):
        """
        Single-image EDM loss for the velocity prediction denoiser.
        """
        denoiser = self.denoiser

        img = batch.pixel_values.to(self.accelerator.device, non_blocking=True)
        if img.ndim == 4:
            img = img[:, :, None]

        c = batch.condition["class_id"]
        cond2 = batch.condition["crop_coords"].to(self.accelerator.device, non_blocking=True)
        anchor_img = batch.condition.get("anchor_frame", img[:, :, :1]).to(
            self.accelerator.device, non_blocking=True
        )
        cond = (
            c.to(self.accelerator.device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c
        )

        with self.accelerator.autocast():
            with torch.no_grad():
                latents = (self.vae.encode(img).latent_dist.sample() - self.vae_shift_factor).mul_(
                    self.vae_scale_factor
                )
                cond_t = self.conditioner(ids=cond, image=anchor_img, cond=cond2, drop=True)

            if isinstance(cond_t["class_labels"], list):
                cond_t["class_labels"][1] = torch.zeros_like(cond_t["class_labels"][1])

            u = torch.randn((latents.shape[0],), device=self.accelerator.device)
            timesteps = torch.sigmoid(u)

            t_unsq = append_dims(timesteps, latents.ndim)

            noise = torch.randn_like(latents)

            x_noisy = t_unsq * latents + (1 - t_unsq) * noise
            x_noisy = x_noisy.permute(0, 2, 1, 3, 4).contiguous()

            context_frames = torch.zeros_like(x_noisy)

            outs = denoiser(
                network=model,
                input=x_noisy,
                sigma=timesteps[:, None],
                cond=cond_t,
                context_frames=context_frames,
                return_means=False,
                return_velocity=True,
            )
            pred = outs[0]

            target = latents - noise
            loss = F.mse_loss(pred, target)

        return loss

    def _sample_tau_like(self, start_frames: torch.Tensor) -> torch.Tensor:
        B, L = start_frames.shape
        device = start_frames.device
        tau0 = torch.rand(B, 1, device=device) * 2.0
        taui = tau0 - torch.rand(B, L, device=device)
        tau = torch.where(start_frames, tau0.expand_as(taui), taui)
        return tau

    def sample_tau_tau_max(
        self,
        start_frames: torch.Tensor,
        M1: torch.Tensor,
        ctx_mask: torch.Tensor,
        vid_lengths: torch.Tensor,
        device=None,
        global_dist: str = "uniform",
        max_tries: int = 8,
    ):
        if device is None:
            device = start_frames.device
        B, L = start_frames.shape

        frame_idx = torch.arange(L, device=device)[None, :]
        valid_len = frame_idx < vid_lengths[:, None]
        cand = M1.bool() & (~ctx_mask.bool()) & valid_len

        has_any_cand = cand.any(dim=1)

        t_ins_all = torch.zeros(B, L, device=device)
        tau_all = torch.zeros(B, L, device=device)
        tau_global_all = torch.zeros(B, 1, device=device)

        need = has_any_cand.clone()
        tries = 0

        eps = 1e-3

        beta_alpha = 1.42
        beta_beta = 2.64
        beta_tail_weight = 0.05

        while need.any() and tries < max_tries:
            idx = torch.where(need)[0]
            Bbad = idx.numel()
            if Bbad == 0:
                break

            t_ins_bad = torch.rand(Bbad, L, device=device)
            sf_bad = start_frames[idx]
            t_ins_bad = torch.where(sf_bad, torch.zeros_like(t_ins_bad), t_ins_bad)

            cand_bad = cand[idx]
            minus_inf = torch.full_like(t_ins_bad, -1e9)
            t_ins_cand = torch.where(cand_bad, t_ins_bad, minus_inf)
            t_max = t_ins_cand.max(dim=1).values
            has_any_here = t_max > -1e8
            t_max = torch.where(has_any_here, t_max, torch.ones_like(t_max))
            tau_max = t_max + 1.0 - eps

            if global_dist == "lognorm":
                sigma = 1.0
                mu = 0.0
                e0 = torch.randn(Bbad, 1, device=device) * sigma + mu
                u = torch.sigmoid(e0).clamp(eps, 1.0 - eps)

            elif global_dist == "beta":
                beta_dist = Beta(beta_alpha, beta_beta)
                t_beta = beta_dist.sample((Bbad, 1)).to(device)
                tail_samples = torch.rand(Bbad, 1, device=device)
                mask = torch.rand(Bbad, 1, device=device) < beta_tail_weight
                u = torch.where(mask, tail_samples, t_beta)
                u = u.clamp(None, 1.0 - eps)

            else:
                u = torch.rand(Bbad, 1, device=device).clamp(eps, 1.0 - eps)

            tau_global_bad = u * tau_max.unsqueeze(1)

            tau_bad = tau_global_bad - t_ins_bad

            t_ins_all[idx] = t_ins_bad
            tau_all[idx] = tau_bad
            tau_global_all[idx] = tau_global_bad

            signal_mask_bad = cand_bad & (tau_bad >= 0.0) & (tau_bad < 1.0)
            ok_bad = signal_mask_bad.any(dim=1)

            need[idx[ok_bad]] = False

            tries += 1

        still_bad = has_any_cand & need
        if still_bad.any():
            idx = torch.where(still_bad)[0]
            Brepair = idx.numel()
            cand_bad = cand[idx]
            t_ins_bad = t_ins_all[idx]

            j = torch.multinomial(cand_bad.float() + 1e-8, num_samples=1).squeeze(1)
            tau_star = torch.rand(Brepair, device=device)

            tau_all[idx, j] = tau_star
            tau_global_all[idx, 0] = tau_star + t_ins_bad[torch.arange(Brepair, device=device), j]

        return tau_all, tau_global_all.squeeze(1)

    def _resample_tau_until_valid(
        self,
        tau: torch.Tensor,
        M1: torch.Tensor,
        ctx_mask: torch.Tensor,
        vid_lengths: torch.Tensor,
        start_frames: torch.Tensor,
        max_tries: int = 8,
        global_dist: str = "uniform",
        eps: float = 1e-3,
    ):
        device = tau.device
        B, L = tau.shape
        vid_lengths = vid_lengths.to(device)

        valid_len = torch.arange(L, device=device)[None, :] < vid_lengths[:, None]
        candidates = M1.bool() & (~ctx_mask.bool()) & valid_len

        def has_signal(t: torch.Tensor) -> torch.Tensor:
            return (candidates & (t >= 0) & (t < 1)).any(dim=1)

        ok = has_signal(tau)
        tries = 0

        while (~ok).any() and tries < max_tries:
            need = ~ok
            if need.any():
                tau_new, _ = self.sample_tau_tau_max(
                    start_frames=start_frames[need],
                    M1=M1[need],
                    ctx_mask=ctx_mask[need],
                    vid_lengths=vid_lengths[need],
                    device=device,
                    global_dist=global_dist,
                )
                tau = tau.clone()
                tau[need] = tau_new
                ok = has_signal(tau)
            tries += 1

        if (~ok).any():
            need = ~ok
            cand = candidates[need]

            no_cand = ~cand.any(dim=1)
            if no_cand.any():
                safe_pos = ~ctx_mask[need] & valid_len[need]
                none_safe = ~safe_pos.any(dim=1)
                safe_pos[none_safe] = valid_len[need][none_safe]
                j_nocand = safe_pos.float().argmax(dim=1)
                u = torch.rand(j_nocand.shape[0], device=device).clamp(eps, 1.0 - eps)
                tau[need.nonzero().flatten()[no_cand], j_nocand] = u

            has_cand_rows = cand.any(dim=1)
            if has_cand_rows.any():
                cand_rows = cand[has_cand_rows]
                j = cand_rows.float().argmax(dim=1)
                u = torch.rand(j.shape[0], device=device).clamp(eps, 1.0 - eps)
                row_idx = need.nonzero().flatten()[has_cand_rows]
                tau[row_idx, j] = u

        return tau

    def get_flowception_loss(self, Y1, M1, vid_lengths, cond_t, context_frames):
        t_flowception_start = time.time()

        batch_size = Y1.shape[0]
        num_frames = Y1.shape[1]
        device = self.accelerator.device

        Y0 = torch.randn_like(Y1)
        Y0[:, 0] = Y1[:, 0]
        M0 = torch.zeros_like(M1)

        start_frames = sample_start_frames_interp(M1, k=self.num_start_frames, skip_first=1)
        assert (start_frames.sum(1) == self.num_start_frames).all() and ~start_frames[:, 0].any()

        M0[start_frames] = 1
        M0[:, 0] = 1
        M0[torch.arange(M0.shape[0], device=device), vid_lengths - 1] = 1

        # Sample context frames randomly (interpolation mode)
        ctx_mask = torch.rand(M1.shape, device=self.accelerator.device) < (
            self.num_context_frames / self.latent_depth
        )
        ctx_mask[:, 0] = True
        ctx_mask[torch.arange(M1.shape[0], device=device), vid_lengths - 1] = True

        # Per-frame dropout so that P(all K context frames dropped) = cfg_p
        p = getattr(self.conditioner, "cfg_p", 0.0)
        B, L = ctx_mask.shape

        if p > 0.0:
            K = ctx_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            log_p = torch.log(torch.tensor(p, device=device))
            q = torch.exp(log_p / K)

            u = torch.rand(B, L, device=device)
            drop_mask = (u < q) & ctx_mask
        else:
            drop_mask = torch.zeros_like(ctx_mask, dtype=torch.bool)

        ctx_keep = ctx_mask & ~drop_mask

        context_aligned = torch.zeros_like(Y1)
        context_aligned[ctx_keep] = Y1[ctx_keep]

        M0 = torch.where(ctx_mask, torch.ones_like(M1, dtype=M1.dtype), M0)

        tau, _ = self.sample_tau_tau_max(
            start_frames=start_frames,
            M1=M1,
            ctx_mask=ctx_mask,
            vid_lengths=vid_lengths,
            device=self.accelerator.device,
            global_dist=self.tau_g_sampling,
        )
        tau = self._resample_tau_until_valid(
            tau=tau,
            M1=M1,
            ctx_mask=ctx_mask,
            vid_lengths=vid_lengths,
            start_frames=start_frames,
            max_tries=20,
            global_dist=self.tau_g_sampling,
        )

        masking_cond = tau >= 0
        t_raw = torch.clip(tau, min=0, max=1)
        t_den = scale_snr(t_raw, sigma_scale=self.sigma_scale)
        t_unsq = t_den[:, :, None, None, None]

        Y_t = (1 - t_unsq) * Y0 + t_unsq * Y1
        M_t = torch.where(masking_cond, M1, M0)

        M_t[:, 0] = 1
        Y_t[:, 0] = Y1[:, 0]

        # Inject kept context frames from Y1 into Y_t
        Y_t = torch.where(ctx_keep[:, :, None, None, None], Y1, Y_t)

        # Zero out DROPPED context frames in Y_t
        if drop_mask.any():
            Y_t = torch.where(
                drop_mask[:, :, None, None, None],
                torch.zeros_like(Y_t),
                Y_t,
            )

        M_t = torch.where(ctx_mask, torch.ones_like(M_t, dtype=M_t.dtype), M_t)

        Y_t_masked = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y0, self.padding_index))

        insert_site_mask = (M_t == 0) & (M1 == 1)
        valid_frame_mask = torch.arange(M_t.shape[1], device=M_t.device)[None, :] < vid_lengths[:, None]

        insert_site_mask = insert_site_mask & valid_frame_mask
        flow_site_mask = (M_t == 1) & valid_frame_mask

        insert_counts = compute_insert_counts(insert_site_mask, flow_site_mask)

        # ======== STRIP SEQUENCE ========
        X_t, left_aligned_indices = left_align_by_mask(Y_t, M_t)
        X_t_m, left_aligned_indices = left_align_by_mask(Y_t_masked, M_t)
        t_left_aligned_den, _ = left_align_by_mask(t_den, M_t)
        t_left_aligned_raw, _ = left_align_by_mask(t_raw, M_t)
        tau_left_aligned, _ = left_align_by_mask(tau, M_t)

        B, L, C, H, W = Y1.shape

        aug_indices = left_aligned_indices.view(B, L, 1, 1, 1)
        aug_indices = aug_indices.expand(-1, -1, C, H, W)

        context_left_aligned = torch.gather(context_aligned, dim=1, index=aug_indices)

        aligned_flow_site_mask = torch.gather(flow_site_mask, dim=1, index=left_aligned_indices)
        aligned_insert_counts = torch.gather(insert_counts, dim=1, index=left_aligned_indices)

        t_flowception_prepare = time.time() - t_flowception_start

        t_forward_start = time.time()
        with self.accelerator.autocast():
            velocity_pred, lambda_ins_pred, means, means_y = self.compute_model(
                x_t=X_t_m,
                time=t_left_aligned_den,
                cond_t=cond_t,
                context_frames=context_left_aligned,
                model=self.model,
                frame_mask=aligned_flow_site_mask,
            )
        t_forward = time.time() - t_forward_start

        t_loss_start = time.time()

        target_velocity = torch.gather(Y1 - Y0, dim=1, index=aug_indices)

        # Build valid mask: flow sites ∧ not context ∧ tau<1
        ctx_mask_aligned = torch.gather(ctx_mask, dim=1, index=left_aligned_indices)
        valid_flow_mask = aligned_flow_site_mask & (~ctx_mask_aligned) & (tau_left_aligned < 1.0)

        # ---- velocity loss (masked) ----
        mask5 = valid_flow_mask[:, :, None, None, None].float()
        diff2 = (velocity_pred - target_velocity).pow(2)
        denom = (
            mask5.sum() * velocity_pred.shape[2] * velocity_pred.shape[3] * velocity_pred.shape[4]
        ).clamp_min(1e-8)

        vel_loss = (diff2 * mask5).sum() / denom

        valid_rate_mask = aligned_flow_site_mask & (tau_left_aligned < 1.0)
        insert_ll = poisson_loss(lambda_ins_pred, aligned_insert_counts, valid_rate_mask, tau_left_aligned)

        rate_loss = insert_ll
        t_loss = time.time() - t_loss_start

        time_dict = {
            "timings/flowception/prepare_train": t_flowception_prepare,
            "timings/flowception/forward": t_forward,
            "timings/flowception_loss": t_loss,
        }

        return vel_loss, rate_loss, means, means_y, time_dict

    def compute_model(self, x_t, time, cond_t, context_frames, model, frame_mask):
        b, d, c, h, w = x_t.shape

        global_velocity, lambda_ins, repa_output, means, means_y = self.denoiser(
            network=model,
            input=x_t,
            sigma=time,
            cond=cond_t,
            context_frames=context_frames,
            frame_mask=frame_mask,
            return_means=True,
            return_velocity=True,
        )

        lambda_ins = torch.exp(lambda_ins)

        global_velocity = rearrange(global_velocity, "b c d h w -> b d c h w", h=h, w=w, c=c)

        return global_velocity, lambda_ins.squeeze(-1), means, means_y

    def train_step(self, model, batch, extra_batch=None):
        if self.train_image_only:
            return self.image_train_step(model=model, batch=batch, extra_batch=extra_batch)
        else:
            return self.image_video_train_step(
                model=model,
                batch=batch,
                extra_batch=extra_batch,
            )

    def image_train_step(self, model, batch, extra_batch=None):
        timing_dict = {}
        with self.accelerator.autocast():
            loss_image = self.image_loss(model, batch)
            loss_all = loss_image

        t_backward_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_all)

        grad_norm = None
        if self.accelerator.sync_gradients:
            if (
                self.clip_grad
                and (getattr(self, "max_grad_norm", None) is not None)
                and self.max_grad_norm > 0
            ):
                grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        t_backward_end = time.time()
        timing_dict["timings/backward"] = t_backward_end - t_backward_start

        loss_dict = {
            "train/loss": loss_image.detach(),
            "train/loss_image": loss_image.detach(),
            "train/vel_loss": torch.tensor(0.0, device=loss_image.device),
            "train/rate_loss": torch.tensor(0.0, device=loss_image.device),
            "train/image_pretrain": torch.tensor(1.0, device=loss_image.device),
        }
        if grad_norm is not None:
            loss_dict["train/grad_norm"] = grad_norm.detach()

        loss_dict.update(timing_dict)

        return TrainTuple(
            loss=loss_image.detach(),
            latents=None,
            loss_dict=loss_dict,
            time_dict=timing_dict,
            vlb="-1",
            cond_t={},
        )

    def image_video_train_step(self, model, batch, extra_batch=None):
        timing_dict = {}

        img = batch.pixel_values

        c = batch.condition["class_id"]
        cond2 = batch.condition["crop_coords"].to(self.accelerator.device, non_blocking=True)
        cond = (
            c.to(self.accelerator.device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c
        )

        img = img.to(self.accelerator.device, non_blocking=True)
        if img.ndim == 4:
            img = img[:, :, None]

        img = self.gpu_aug(img, self.aug_gen)
        img = downsample_video(img, self.target_h, self.target_w)
        anchor_img = img[:, :, :1]

        with self.accelerator.autocast():
            with torch.no_grad():
                t_ae_start = time.time()
                latents = (self.vae.encode(img).latent_dist.sample() - self.vae_shift_factor).mul_(
                    self.vae_scale_factor
                )
                t_ae_end = time.time()
                timing_dict["timings/autoencoder"] = t_ae_end - t_ae_start

                latents, M1 = pick_latents_after_skip(
                    latents,
                    batch.condition["frame_mask"].to(self.accelerator.device, non_blocking=True),
                    group=self.temporal_factor,
                    num_start_frames=self.num_start_frames,
                )

                latents = latents.permute(0, 2, 1, 3, 4)

                t_cond_start = time.time()
                cond_t = self.conditioner(ids=cond, image=anchor_img, cond=cond2, drop=True)
                t_cond_end = time.time()
                timing_dict["timings/conditioner"] = t_cond_end - t_cond_start

            context_frames = latents[:, :1]
            if self.mask_first_frame:
                frames_mask = torch.rand_like(cond_t["cfg_mask"].to(context_frames)) > self.conditioner.cfg_p
            else:
                frames_mask = torch.rand_like(cond_t["cfg_mask"].to(context_frames)) > -1.0

            context_frames = context_frames * frames_mask[:, None, None, None, None]

            Y1 = latents
            vid_lengths = M1.long().sum(dim=1)

            vel_loss, rate_loss, means, means_y, flowception_timings = self.get_flowception_loss(
                Y1, M1, vid_lengths, cond_t, context_frames
            )
            timing_dict.update(flowception_timings)

            loss = vel_loss + self.poisson_loss_weight * rate_loss

            if extra_batch is not None:
                loss_image = self.image_loss(model, extra_batch)
            else:
                loss_image = torch.zeros_like(loss)

            loss_all = loss + self.image_loss_weight * loss_image

        # NaN detection
        local_nan = torch.isnan(loss.detach())

        if torch.distributed.is_initialized():
            nan_flag = torch.tensor(local_nan, device=loss.device, dtype=torch.int)
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            any_nan = nan_flag.item() > 0
        else:
            any_nan = local_nan.item()

        if any_nan:
            if local_nan.item():
                self.logger.warning(f"[Rank {self.accelerator.process_index}] Detected NaN, saving crash data")

                crash_dir = Path(self.results_folder) / "crash_data_rank{}".format(
                    self.accelerator.process_index
                )
                os.makedirs(crash_dir, exist_ok=True)

                torch.save(batch, crash_dir / "batch.pt")
                torch.save(
                    {
                        "latents": latents,
                        "cond_t": cond_t,
                        "context_frames": context_frames,
                    },
                    crash_dir / "inputs.pt",
                )

                self.logger.warning(f"[Rank {self.accelerator.process_index}] Saved crash data to {crash_dir}")

            self.accelerator.wait_for_everyone()

            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                Path(self.results_folder) / "crash_data" / "model_ckpt.bin",
                **get_fsdp_ckpt_kwargs(),
            )

            self.accelerator.wait_for_everyone()

            raise RuntimeError(f"[Rank {self.accelerator.process_index}] NaN detected. Exiting.")

        t_backward_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_all)

        grad_norm = None

        if self.accelerator.sync_gradients:
            if (
                self.clip_grad
                and (getattr(self, "max_grad_norm", None) is not None)
                and self.max_grad_norm > 0
            ):
                grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        t_backward_end = time.time()
        timing_dict["timings/backward"] = t_backward_end - t_backward_start

        loss_dict = {
            "train/loss": loss.detach(),
            "train/loss_image": loss_image.detach(),
            "train/vel_loss": vel_loss.detach(),
            "train/rate_loss": rate_loss.detach(),
        }

        if grad_norm is not None:
            loss_dict["train/grad_norm"] = grad_norm.detach()

        loss_dict.update(timing_dict)

        for layer_idx, val in enumerate(means):
            loss_dict[f"activations/x/layer_{layer_idx:02d}"] = val.detach()
        for layer_idx, val in enumerate(means_y):
            loss_dict[f"activations/y/layer_{layer_idx:02d}"] = val.detach()

        return TrainTuple(
            loss=loss.detach(),
            latents=None,
            loss_dict=loss_dict,
            time_dict=timing_dict,
            vlb="-1",
            cond_t=cond_t,
        )

    @torch.no_grad()
    def sampling_step(
        self,
        latents,
        epoch,
        model,
        use_cfg=True,
        s_text=None,
        s_img=None,
        s_offset=None,
        s_ins=None,
        num_steps=25,
        grid_size=8,
        keep_emb=False,
        save_img=True,
        return_data=False,
        return_feats=False,
    ):
        self.logger.info("Saving snapshots to GIF...")
        self.accelerator.wait_for_everyone()

        batch_size = self.batch_size
        latent_depth = self.latent_depth

        grid_latents = []
        grid_masks = []

        sample_idx = 0

        model.eval()
        for p in model.parameters():
            p.grad = None
        with self.accelerator.autocast():
            for _ in range(int(np.ceil(grid_size / (self.accelerator.num_processes * self.batch_size)))):
                idx = np.random.randint(0, len(self.val_dataloader.dataset), self.batch_size)

                samples = [self.val_dataloader.dataset.__getitem__(i) for i in idx]
                captions = [sample.condition["class_id"] for sample in samples]

                img = [sample.pixel_values for sample in samples]
                img = torch.stack(img)
                img = img.to(self.accelerator.device, non_blocking=True)
                if img.ndim == 4:
                    img = img[:, :, None]

                anchor_img = img.clone()

                with torch.no_grad():
                    context_frames = (
                        (self.vae.encode(img).latent_dist.sample() - self.vae_shift_factor)
                        .mul_(self.vae_scale_factor)
                        .permute(0, 2, 1, 3, 4)
                    )

                num_ctx = self.num_context_frames
                context_frames = select_random_frames(context_frames, num_ctx)

                if not isinstance(captions[0], str):
                    captions = None

                cond_t = self.conditioner.sample(
                    batch_size=self.batch_size,
                    idx=captions,
                    image=anchor_img,
                )
                cond_t = {k: v for k, v in cond_t.items() if k != "mask"}
                uc = self.conditioner.get_cfg_version(cond_t)
                if keep_emb:
                    uc["class_labels"][1] = cond_t["class_labels"][1].clone()

                if self.guider is not None:
                    self.guider.clean_buffer()

                s_text = s_text or self.cfg_scale
                s_img = s_img or self.image_guidance_scale
                s_offset = s_offset or self.guidance_offset
                s_ins = s_ins or self.guidance_insertion_scale

                Y_t, M_t, Y_t_l, M_t_l, ins_timemap, expected_l = vanilla_sample_interp_flowception(
                    first_frames=context_frames,
                    model=self.model,
                    forward_fn=self.compute_model,
                    num_steps=num_steps,
                    cond_t=cond_t,
                    context_frames=context_frames,
                    batch_size=batch_size,
                    input_length=self.latent_depth,
                    device=self.accelerator.device,
                    padding_index=self.padding_index,
                    start_frames=self.num_start_frames,
                    max_inserts=self.latent_depth - self.num_start_frames,
                    ins_start=0.0,
                    uc=uc,
                    s_text=s_text,
                    s_img=s_img,
                    s_offset=s_offset,
                    s_ins=s_ins,
                )

                self.accelerator.wait_for_everyone()
                grid_latents.append(self.accelerator.gather(Y_t)[:grid_size])
                grid_masks.append(self.accelerator.gather(M_t)[:grid_size])

        grid_latents = torch.cat(grid_latents)[:grid_size].permute(0, 2, 1, 3, 4)
        grid_masks = torch.cat(grid_masks)[:grid_size]

        z = self.vae_shift_factor + grid_latents / self.vae_scale_factor
        latent_mask = grid_masks
        if latent_mask.ndim > 2:
            latent_mask = latent_mask[..., 0]
        latent_mask = latent_mask.to(torch.bool).to(z.device)

        tcr = getattr(self.vae, "temporal_compression_ratio", self.temporal_factor)

        accepts_mask = vae_accepts_latent_mask(self.vae)

        restore_flags = {}
        if accepts_mask:
            for flag in ("use_tiling", "use_framewise_decoding"):
                if hasattr(self.vae, flag):
                    restore_flags[flag] = getattr(self.vae, flag)
                    if restore_flags[flag]:
                        setattr(self.vae, flag, False)

        try:
            with torch.no_grad(), self.accelerator.autocast():
                if accepts_mask:
                    img = self.vae.decode(z, latent_mask=latent_mask).sample
                else:
                    img = self.vae.decode(z).sample
        except NotImplementedError:
            with torch.no_grad(), self.accelerator.autocast():
                img = self.vae.decode(z).sample
        finally:
            for k, v in restore_flags.items():
                setattr(self.vae, k, v)

        G, C, T_img, H, W = img.shape
        T_lat = latent_mask.shape[1]

        repeats = torch.ones(T_lat, dtype=torch.long, device=img.device)
        if T_lat > 1:
            repeats[1:] = tcr

        mask_img = torch.repeat_interleave(latent_mask, repeats=repeats, dim=1)
        if mask_img.shape[1] > T_img:
            mask_img = mask_img[:, :T_img]
        elif mask_img.shape[1] < T_img:
            pad = mask_img[:, -1:].expand(-1, T_img - mask_img.shape[1])
            mask_img = torch.cat([mask_img, pad], dim=1)

        mask5 = mask_img[:, None, :, None, None].to(img.dtype).to(img.device)
        img = img * mask5 + (1 - mask5) * 1.0

        img = img.float().detach().cpu().numpy()
        img = (img + 1) / 2.0
        img = img.clip(0.0, 1.0)
        img_cat = np.concatenate([im for im in img], axis=3)
        if save_img:
            prefix = f"video_{sample_idx}.gif"
            savep = Path(self.results_dir) / "snapshots" / prefix
            self.logger.info(f"Saving to : {savep}")
            frames_to_gif(img_cat, save_path=savep, fps=self.sampling_fps)

        model.train()

        if return_feats:
            return img_cat, Y_t, M_t, Y_t_l, M_t_l, ins_timemap, expected_l

        if return_data:
            return (
                img_cat,
                grid_latents,
                grid_masks,
            )

        return img_cat

    @torch.inference_mode()
    def sample_text_to_image_flow_only(
        self,
        captions: list[str],
        idx: list[int],
        num_steps: int = 25,
        s_text: float | None = None,
        keep_emb: bool = False,
        use_cfg: bool = True,
    ):
        device = self.accelerator.device
        B = len(captions)
        s_text_eff = self.cfg_scale if s_text is None else s_text

        cond_t = self.conditioner.sample(batch_size=B, idx=captions, image=None)
        cond_t = {k: v for k, v in cond_t.items() if k != "mask"}
        uc = self.conditioner.get_cfg_version(cond_t)
        if keep_emb:
            uc["class_labels"][1] = cond_t["class_labels"][1].clone()

        ds = self.val_dataloader.dataset
        probe = [ds.__getitem__(i).pixel_values for i in idx]
        probe = torch.stack(probe).to(device, non_blocking=True)
        if probe.ndim == 4:
            probe = probe[:, :, None]
        probe = probe[:, :, :1]

        with self.accelerator.autocast():
            lat = (self.vae.encode(probe).latent_dist.sample() - self.vae_shift_factor).mul_(
                self.vae_scale_factor
            )
        _, C_z, _, H_z, W_z = lat.shape

        Y_t = torch.randn(B, 1, C_z, H_z, W_z, device=device, dtype=lat.dtype)
        M_t = torch.ones(B, 1, dtype=torch.bool, device=device)
        t = torch.zeros(B, 1, device=device, dtype=lat.dtype)
        h = torch.full((B, 1), 1.0 / num_steps, device=device, dtype=lat.dtype)

        for _ in range(num_steps):
            vel, _, _, _ = self.compute_model(
                x_t=Y_t,
                time=t,
                cond_t=cond_t,
                context_frames=torch.zeros_like(Y_t),
                model=self.model,
                frame_mask=M_t,
            )
            if use_cfg and s_text_eff > 1.0:
                vel_u, _, _, _ = self.compute_model(
                    x_t=Y_t,
                    time=t,
                    cond_t=uc,
                    context_frames=torch.zeros_like(Y_t),
                    model=self.model,
                    frame_mask=M_t,
                )
                vel = vel * s_text_eff + vel_u * (1.0 - s_text_eff)

            Y_t = Y_t + h[..., None, None, None] * vel
            t = (t + h).clamp(max=1.0)

        Yg = Y_t.permute(0, 2, 1, 3, 4).contiguous()
        return Yg, M_t

    @torch.no_grad()
    def generate_samples(
        self,
        num_batches: int = 10,
        num_steps: int = 25,
        out_root: str | Path | None = None,
        prefix: str = "val",
        fps: int | None = None,
        s_text: float | None = None,
        s_img: float | None = None,
    ):
        device = self.accelerator.device
        rank = self.accelerator.process_index

        model = self.model
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.grad = None

        if out_root is None:
            out_root = Path(self.results_folder) / "eval_samples"
        out_root = Path(out_root)
        gt_dir = out_root / "gt_mp4"
        gen_dir = out_root / "gen_mp4"
        prm_dir = out_root / "prompts"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)
        prm_dir.mkdir(parents=True, exist_ok=True)

        fps = int(fps or getattr(self, "sampling_fps", 16))

        def tensor_video_to_uint8(x: torch.Tensor) -> np.ndarray:
            x = x.clamp(-1, 1)
            x = (x + 1) / 2.0
            x = (x * 255.0).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            return x

        s_text = s_text or self.cfg_scale
        s_img = s_img or self.image_guidance_scale

        local_idx = 0
        pbar = range(num_batches)
        if self.accelerator.is_main_process:
            pbar = tqdm.tqdm(pbar, desc="[generate_samples]")

        iterator = iter(self.val_dataloader)

        for _ in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            img_gt = batch.pixel_values.to(device, non_blocking=True)
            frame_mask = batch.condition["frame_mask"].to(device, non_blocking=True)
            if frame_mask.ndim == 1:
                frame_mask = frame_mask[None, :].expand(img_gt.shape[0], -1)
            B, C, T_gt, H, W = img_gt.shape

            c = batch.condition["class_id"]
            cond2 = batch.condition["crop_coords"].to(device, non_blocking=True)
            cond_ids = c.to(device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c

            if isinstance(cond_ids, torch.Tensor):
                prompt_list = [f"label_id:{int(x)}" for x in cond_ids.detach().cpu().tolist()]
            elif isinstance(cond_ids, (list, tuple)):
                prompt_list = [str(x) for x in cond_ids]
            else:
                prompt_list = [str(cond_ids) for _ in range(B)]

            anchor_img = img_gt[:, :, :1].contiguous()
            with self.accelerator.autocast():
                context_lat = (self.vae.encode(anchor_img).latent_dist.sample() - self.vae_shift_factor).mul_(
                    self.vae_scale_factor
                )
            context_frames = context_lat.permute(0, 2, 1, 3, 4).contiguous()

            num_ctx = self.num_context_frames
            context_frames = select_random_frames(context_frames, num_ctx)

            captions = None if isinstance(cond_ids, torch.Tensor) else cond_ids
            cond_t = self.conditioner.sample(batch_size=B, idx=captions, image=anchor_img)
            cond_t = {k: v for k, v in cond_t.items() if k != "mask"}
            uc = self.conditioner.get_cfg_version(cond_t)

            if self.guider is not None:
                self.guider.clean_buffer()
            with self.accelerator.autocast():
                Y_t, M_t, _, _, _, _ = vanilla_sample_interp_flowception(
                    first_frames=context_frames,
                    model=self.model,
                    forward_fn=self.compute_model,
                    num_steps=num_steps,
                    cond_t=cond_t,
                    context_frames=context_frames,
                    batch_size=B,
                    input_length=self.latent_depth,
                    device=device,
                    padding_index=self.padding_index,
                    start_frames=self.num_start_frames,
                    max_inserts=self.latent_depth - self.num_start_frames,
                    ins_start=0.0,
                    uc=uc,
                    s_text=s_text,
                    s_img=s_img,
                )

            Y_t_bcdhw = Y_t.permute(0, 2, 1, 3, 4).contiguous()
            z = self.vae_shift_factor + Y_t_bcdhw / self.vae_scale_factor
            latent_mask = M_t
            if latent_mask.ndim > 2:
                latent_mask = latent_mask[..., 0]
            latent_mask = latent_mask.to(torch.bool).to(Y_t.device)

            tcr = getattr(self.vae, "temporal_compression_ratio", self.temporal_factor)

            accepts_mask = vae_accepts_latent_mask(self.vae)

            restore_flags = {}
            if accepts_mask:
                for flag in ("use_tiling", "use_framewise_decoding"):
                    if hasattr(self.vae, flag):
                        restore_flags[flag] = getattr(self.vae, flag)
                        if restore_flags[flag]:
                            setattr(self.vae, flag, False)

            try:
                with torch.no_grad(), self.accelerator.autocast():
                    if accepts_mask:
                        vid_gen = self.vae.decode(z, latent_mask=latent_mask).sample
                    else:
                        vid_gen = self.vae.decode(z).sample
            except NotImplementedError:
                with torch.no_grad(), self.accelerator.autocast():
                    vid_gen = self.vae.decode(z).sample
            finally:
                for k, v in restore_flags.items():
                    setattr(self.vae, k, v)

            D_lat = M_t.shape[1]
            repeats = torch.ones(D_lat, dtype=torch.long, device=device)
            if D_lat > 1:
                repeats[1:] = self.temporal_factor
            mask_frames_gen = torch.repeat_interleave(M_t.to(device), repeats=repeats, dim=1)
            Tg = vid_gen.shape[2]
            if mask_frames_gen.shape[1] > Tg:
                mask_frames_gen = mask_frames_gen[:, :Tg]
            elif mask_frames_gen.shape[1] < Tg:
                pad = mask_frames_gen[:, -1:].expand(-1, Tg - mask_frames_gen.shape[1])
                mask_frames_gen = torch.cat([mask_frames_gen, pad], dim=1)

            for b in range(B):
                file_id = f"{prefix}_r{rank:02d}_{local_idx:06d}"
                gt_path = gt_dir / f"{file_id}.mp4"
                gen_path = gen_dir / f"{file_id}.mp4"
                txt_path = prm_dir / f"{file_id}.txt"
                jsn_path = prm_dir / f"{file_id}.json"

                mask_b = frame_mask[b] if frame_mask.shape[0] == B else frame_mask
                real_T_gt = int(mask_b.long().sum().item())
                vid_b_gt = img_gt[b, :, :real_T_gt]
                frames_gt = tensor_video_to_uint8(vid_b_gt)

                real_T_gen = int(mask_frames_gen[b].long().sum().item())
                vid_b_gen = vid_gen[b, :, :real_T_gen]
                frames_gen = tensor_video_to_uint8(vid_b_gen)

                min_len = min(frames_gt.shape[0], frames_gen.shape[0])
                frames_gt, frames_gen = frames_gt[:min_len], frames_gen[:min_len]

                write_frames_ffmpeg(frames_gt, str(gt_path), fps=fps)
                write_frames_ffmpeg(frames_gen, str(gen_path), fps=fps)

                prompt_text = prompt_list[b] if b < len(prompt_list) else ""
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text if isinstance(prompt_text, str) else str(prompt_text))

                meta = {
                    "file_id": file_id,
                    "prompt": prompt_text,
                    "gt_path": str(gt_path),
                    "gen_path": str(gen_path),
                    "fps": fps,
                    "num_steps": num_steps,
                    "s_text": float(s_text),
                    "s_img": float(s_img),
                    "rank": int(rank),
                }
                with open(jsn_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                local_idx += 1

            del batch, img_gt, frame_mask, Y_t, M_t, Y_t_bcdhw, vid_gen
            torch.cuda.empty_cache()

        if was_training:
            model.train()

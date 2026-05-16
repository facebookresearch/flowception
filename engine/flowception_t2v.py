import gc
import inspect
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
    GPUBatchVideoAug,
    downsample_video,
    get_ema_loss_heatmap_to_wandb,
    logit_normal_weight,
    randomly_select_slice,
    sample_tau_g_edge_equalized,
    scale_snr,
    update_ema_loss_bins,
    vae_accepts_latent_mask,
)
from helpers.video_utils.visualization import frames_to_gif, write_frames_ffmpeg
from modules.flowception.align import (
    compute_insert_counts,
    left_align_by_mask,
    left_align_frames,
)
from modules.flowception.helpers import pick_latents_after_skip, sample_start_frames
from modules.flowception.losses import poisson_loss
from modules.flowception.sampling import (
    vanilla_sample_flowception_prescribed,
    vanilla_sample_flowception_t2v,
)
from modules.flowception.schedulers import get_kappa_scheduler
from modules.diffusion.denoiser import build_denoiser_wrapper
from modules.diffusion.denoiser_scaling import CondOTScaling
from modules.diffusion.sigma_sampling import CondOTUniformDiscretization
from modules.sampling.apg import APGGuider
from torch.distributions import Beta


class FlowceptionT2V(Trainer):
    """Flowception trainer for text-to-video generation.

    Similar to Flowception but generates videos conditioned on text prompts
    rather than anchor images. Uses the same temporal interpolation mechanism
    but with text-conditioned context frames.
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

        self.image_loss_weight = cfg.FLOWCEPTION.LOSS.IMG_WEIGHT

        # Add these in __init__ or global setup
        self.grad_ema = 0.02
        self.grad_ema_var = 0.0002
        self.grad_ema_beta = 0.99  # α in the paper

        self.kappa_scheduler = get_kappa_scheduler(cfg.FLOWCEPTION.KAPPA_SCHEDULER)
        self.num_start_frames = cfg.FLOWCEPTION.NUM_START_FRAMES
        self.padding_index = cfg.DATA.PADDING_INDEX

        self.sampling_fps = cfg.DATA.VIDEO.SAMPLING_FPS
        self.train_image_only = cfg.SOLVER.IMAGE_ONLY

        self.guidance_offset = cfg.FLOWCEPTION.SAMPLING.GUIDANCE_OFFSET
        self.guidance_insertion_scale = cfg.FLOWCEPTION.SAMPLING.GUIDANCE_INS

        self.insertion_rule = cfg.FLOWCEPTION.SAMPLING.INSERTION_RULE
        self.gpu_aug = GPUBatchVideoAug()
        self.aug_gen = torch.Generator(device=self.accelerator.device)
        self.target_h = cfg.SOLVER.IM_SIZE
        self.target_w = cfg.SOLVER.IM_SIZE

        self.poisson_loss_weight = cfg.FLOWCEPTION.LOSS.POISSON_WEIGHT

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def get_loss_per_bin(self, timesteps, losses):
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        # Calculate average loss for each range
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
        Single-image(-ish) EDM loss that works with the current denoiser API.
        - Passes anchor frame to the conditioner
        - Robustly handles denoiser's return signature (first item is the pred)
        - Fixes 'normalized_mse' (was using latents - latents)
        - Sets return_velocity only for *vel* losses
        """
        denoiser = self.denoiser

        # inputs
        img = batch.pixel_values.to(self.accelerator.device, non_blocking=True)
        if img.ndim == 4:
            # make [B, C, 1, H, W] for 2D images
            img = img[:, :, None]

        c = batch.condition["class_id"]
        cond2 = batch.condition["crop_coords"].to(self.accelerator.device, non_blocking=True)

        anchor_img = None
        cond = (
            c.to(self.accelerator.device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c
        )

        with self.accelerator.autocast():
            with torch.no_grad():
                latents = (self.vae.encode(img).latent_dist.sample() - self.vae_shift_factor).mul_(
                    self.vae_scale_factor
                )
                cond_t = self.conditioner(ids=cond, image=None, cond=cond2, drop=True)

            # mask out image features for image generation.
            if isinstance(cond_t["class_labels"], list):
                cond_t["class_labels"][1] = torch.zeros_like(cond_t["class_labels"][1])

            # Enforce a lognorm timespte for the image loss.
            u = torch.randn((latents.shape[0],), device=self.accelerator.device)
            timesteps = torch.sigmoid(u)
            timesteps = scale_snr(timesteps, sigma_scale=self.sigma_scale)
            t_unsq = append_dims(timesteps, latents.ndim)

            noise = torch.randn_like(latents)
            x_noisy = t_unsq * latents + (1 - t_unsq) * noise
            x_noisy = x_noisy.permute(0, 2, 1, 3, 4).contiguous()

            # no context for image loss
            context_frames = torch.zeros_like(x_noisy)

            # predict (supports both x0- and velocity-pred depending on self.loss_type)
            outs = denoiser(
                network=model,
                input=x_noisy,
                sigma=timesteps[:, None],
                cond=cond_t,
                context_frames=context_frames,
                return_means=False,
                return_velocity=True,  # self.loss_type in ["pseudo_huber_divergence", "mse_vel", "mse_vel_weighted"],
            )
            pred = outs[0]  # first item is the model prediction regardless of tuple length

            target = latents - noise
            loss = F.mse_loss(pred, target)

        return loss

    def _sample_tau_like(self, start_frames: torch.Tensor) -> torch.Tensor:
        """
        Reproduce your current tau sampler:
        tau0 ~ U[0,2) (shape [B,1])
        taui ~ tau0 - U[0,1) (shape [B,L])
        tau  = where(start_frames, tau0, taui)
        start_frames: [B, L] bool
        returns tau:  [B, L]
        """
        B, L = start_frames.shape
        device = start_frames.device
        tau0 = torch.rand(B, 1, device=device) * 2.0  # [0,2)
        taui = tau0 - torch.rand(B, L, device=device)  # tau0 - U[0,1)
        tau = torch.where(start_frames, tau0.expand_as(taui), taui)
        return tau

    def sample_tau_tau_max(
        self,
        start_frames: torch.Tensor,  # [B, L] bool
        M1: torch.Tensor,  # [B, L] (1==real/kept frame)
        ctx_mask: torch.Tensor,  # [B, L] (True==context to exclude from inserts)
        vid_lengths: torch.Tensor,  # [B]    valid video length per sample
        device=None,
        global_dist: str = "uniform",  # "uniform" | "lognorm" | "beta"
        max_tries: int = 8,
    ):
        if device is None:
            device = start_frames.device
        B, L = start_frames.shape

        # ----- candidate mask (where we *want* signal) -----
        frame_idx = torch.arange(L, device=device)[None, :]
        valid_len = frame_idx < vid_lengths[:, None]  # [B, L]
        cand = M1.bool() & (~ctx_mask.bool()) & valid_len  # [B, L]

        # rows with no candidates at all: can't enforce anything there
        has_any_cand = cand.any(dim=1)  # [B]

        # storage for outputs
        t_ins_all = torch.zeros(B, L, device=device)
        tau_all = torch.zeros(B, L, device=device)
        tau_global_all = torch.zeros(B, 1, device=device)

        # we only resample rows in `need`
        need = has_any_cand.clone()  # only rows that have candidates need the constraint
        tries = 0

        eps = 1e-3

        # params for beta schedule (you can move these to __init__ if you like)
        beta_alpha = 1.42
        beta_beta = 2.64
        beta_tail_weight = 0.05

        while need.any() and tries < max_tries:
            idx = torch.where(need)[0]  # indices of rows to resample
            Bbad = idx.numel()
            if Bbad == 0:
                break

            # --- 1) sample t_ins in [0,1) ---
            t_ins_bad = torch.rand(Bbad, L, device=device)
            # force start frames to behave like copies of tau_global (t_ins = 0 there)
            sf_bad = start_frames[idx]
            t_ins_bad = torch.where(sf_bad, torch.zeros_like(t_ins_bad), t_ins_bad)

            # --- 2) tau_max = max_{i in cand} t_ins^i + 1  (per sample) ---
            cand_bad = cand[idx]  # [Bbad, L]
            minus_inf = torch.full_like(t_ins_bad, -1e9)
            t_ins_cand = torch.where(cand_bad, t_ins_bad, minus_inf)
            t_max = t_ins_cand.max(dim=1).values  # [Bbad]
            has_any_here = t_max > -1e8  # sanity guard
            # if somehow a "cand" row is all False (shouldn't happen if has_any_cand),
            # fall back to 1.0
            t_max = torch.where(has_any_here, t_max, torch.ones_like(t_max))
            tau_max = t_max + 1.0 - eps  # [Bbad]

            # --- 3) tau_global ~ (0, tau_max) with chosen global_dist ---
            if global_dist == "lognorm":
                # logit-normal on (0,1) then scale by tau_max
                sigma = 1.0
                mu = 0.0
                e0 = torch.randn(Bbad, 1, device=device) * sigma + mu
                u = torch.sigmoid(e0)
                u = u.clamp(None, 1.0 - eps)

            elif global_dist == "beta":
                # Beta(α, β) on (0,1) + uniform tail mixture (pdf floor)
                beta_dist = Beta(beta_alpha, beta_beta)
                t_beta = beta_dist.sample((Bbad, 1)).to(device)  # [Bbad, 1]

                tail_samples = torch.rand(Bbad, 1, device=device)  # uniform [0,1]
                mask = torch.rand(Bbad, 1, device=device) < beta_tail_weight
                u = torch.where(mask, tail_samples, t_beta)
                u = u.clamp(None, 1.0 - eps)

            else:  # "uniform"
                u = torch.rand(Bbad, 1, device=device)
                u = u.clamp(None, 1.0 - eps)

            tau_global_bad = u * tau_max.unsqueeze(1)  # [Bbad, 1]

            # --- 4) tau_i = tau_global - t_ins^i ---
            tau_bad = tau_global_bad - t_ins_bad  # [Bbad, L]

            # store into full tensors
            t_ins_all[idx] = t_ins_bad
            tau_all[idx] = tau_bad
            tau_global_all[idx] = tau_global_bad

            # ----- check per-sample validity -----
            signal_mask_bad = cand_bad & (tau_bad >= 0.0) & (tau_bad < 1.0)  # [Bbad, L]
            ok_bad = signal_mask_bad.any(dim=1)  # [Bbad]

            # rows that are OK no longer need resample
            need[idx[ok_bad]] = False

            tries += 1

        # If any rows with candidates are *still* invalid, do a minimal repair:
        still_bad = has_any_cand & need
        if still_bad.any():
            idx = torch.where(still_bad)[0]
            Brepair = idx.numel()
            cand_bad = cand[idx]  # [Brepair, L]
            t_ins_bad = t_ins_all[idx]  # [Brepair, L]

            # pick one candidate index j per row (where cand==True)
            j = torch.multinomial(cand_bad.float() + 1e-8, num_samples=1).squeeze(1)  # [Brepair]
            # sample tau* ~ U(0,1) for that site
            tau_star = torch.rand(Brepair, device=device)  # [Brepair]

            # set τ at that site
            tau_all[idx, j] = tau_star
            # and adjust tau_global so that τ = tau_global - t_ins remains consistent at j
            tau_global_all[idx, 0] = tau_star + t_ins_bad[torch.arange(Brepair, device=device), j]

        return tau_all, tau_global_all.squeeze(1)

    def get_flowception_loss(self, Y1, M1, vid_lengths, cond_t, context_frames):
        t_flowception_start = time.time()

        batch_size = Y1.shape[0]
        num_frames = Y1.shape[1]
        device = self.accelerator.device

        Y0 = torch.randn_like(Y1)

        M0 = torch.zeros_like(M1)

        start_frames = sample_start_frames(M1, k=self.num_start_frames, skip_first=1)

        M0[start_frames] = 1
        M0[torch.arange(M0.shape[0], device=device), vid_lengths - 1] = (
            1  # last frame doesn't induce insertions.
        )

        # --- t2v: only the 1st frame is context ---
        ctx_mask = torch.zeros_like(M1, dtype=torch.bool)

        tau, tau_0 = self.sample_tau_tau_max(
            start_frames=start_frames,
            M1=M1,
            ctx_mask=ctx_mask,
            vid_lengths=vid_lengths,
            device=self.accelerator.device,
            global_dist=self.tau_g_sampling,
        )

        masking_cond = tau >= 0
        t_raw = torch.clip(tau, 0.0, 1.0)  # insertion clock (raw)
        t_den = scale_snr(t_raw, sigma_scale=self.sigma_scale)  # denoising clock (warped)
        Y_t = (1 - t_den[:, :, None, None, None]) * Y0 + t_den[:, :, None, None, None] * Y1

        M_t = torch.where(masking_cond, M1, M0)  # masking interpolation go from one mask to the other.

        M_t[:, 0] = 1

        Y_t_masked = torch.where(M_t[:, :, None, None, None], Y_t, torch.full_like(Y0, self.padding_index))

        # gather insert locations and flow locations.
        insert_site_mask = (M_t == 0) & (M1 == 1)
        valid_frame_mask = torch.arange(M_t.shape[1], device=M_t.device)[None, :] < vid_lengths[:, None]

        insert_site_mask = insert_site_mask & valid_frame_mask
        flow_site_mask = (
            M_t == 1
        ) & valid_frame_mask  # & (kappa_t > 0) # only backprop insertion after tmin_ins

        insert_counts = compute_insert_counts(insert_site_mask, flow_site_mask)

        # ======== 3. STRIP SEQUENCE ========
        X_t, left_aligned_indices = left_align_by_mask(Y_t, M_t)
        X_t_m, left_aligned_indices = left_align_by_mask(Y_t_masked, M_t)
        t_left_aligned_den, _ = left_align_by_mask(t_den, M_t)
        tau_left_aligned, _ = left_align_by_mask(tau, M_t)

        B, L, C, H, W = Y1.shape

        # Add singleton dims for H, W, C:
        #   [B, L] -> [B, L, 1, 1, 1]
        aug_indices = left_aligned_indices.view(B, L, 1, 1, 1)
        aug_indices = aug_indices.expand(-1, -1, C, H, W)

        aligned_flow_site_mask = torch.gather(flow_site_mask, dim=1, index=left_aligned_indices)
        aligned_insert_counts = torch.gather(insert_counts, dim=1, index=left_aligned_indices)
        t_flowception_prepare = time.time() - t_flowception_start

        # context is always empty.
        context_full = torch.zeros_like(Y1)

        t_forward_start = time.time()
        with self.accelerator.autocast():
            velocity_pred, lambda_ins_pred, means, means_y = self.compute_model(
                x_t=X_t_m,
                time=t_left_aligned_den,
                cond_t=cond_t,
                context_frames=context_full,
                model=self.model,
                frame_mask=aligned_flow_site_mask,
            )
        t_forward = time.time() - t_forward_start

        t_loss_start = time.time()

        target_velocity = torch.gather(Y1 - Y0, dim=1, index=aug_indices)

        # after computing aligned_flow_site_mask, tau_left_aligned, etc.
        valid_flow_mask = aligned_flow_site_mask & (tau_left_aligned >= 0.0) & (tau_left_aligned < 1.0)

        # ---- velocity loss ----
        mask5 = valid_flow_mask[:, :, None, None, None].float()  # [B,L,1,1,1]
        diff2 = (velocity_pred - target_velocity).pow(2)  # [B,L,C,H,W]
        denom = (
            mask5.sum() * velocity_pred.shape[2] * velocity_pred.shape[3] * velocity_pred.shape[4]
        ).clamp_min(1e-8)

        vel_loss = (diff2 * mask5).sum() / denom

        # after alignment
        valid_rate_mask = aligned_flow_site_mask & (tau_left_aligned < 1.0)
        insert_ll = poisson_loss(lambda_ins_pred, aligned_insert_counts, valid_flow_mask, tau_left_aligned)

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

        # standard backward/step block (unchanged)
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
            cond_t={},  # not needed for image-only
        )

    def image_video_train_step(self, model, batch, extra_batch=None):
        timing_dict = {}

        img = batch.pixel_values.to(self.accelerator.device, non_blocking=True)
        if img.ndim == 4:
            img = img[:, :, None]  # [B,C,1,H,W]

        img = self.gpu_aug(img, self.aug_gen)
        img = downsample_video(img, self.target_h, self.target_w)

        # anchor from augmented frames (cheap + consistent)
        anchor_img = img[:, :, :1]

        c = batch.condition["class_id"]
        cond2 = batch.condition["crop_coords"].to(self.accelerator.device, non_blocking=True)
        cond = (
            c.to(self.accelerator.device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c
        )

        with self.accelerator.autocast():
            with torch.no_grad():
                t_ae_start = time.time()
                latents = (
                    self.vae.encode(
                        img,
                    ).latent_dist.sample()
                    - self.vae_shift_factor
                ).mul_(self.vae_scale_factor)
                t_ae_end = time.time()
                timing_dict["timings/autoencoder"] = t_ae_end - t_ae_start

                latents, M1 = pick_latents_after_skip(
                    latents,
                    batch.condition["frame_mask"].to(self.accelerator.device, non_blocking=True),
                    group=self.temporal_factor,
                    num_start_frames=self.num_start_frames,
                    skip="none",
                )

                latents = latents.permute(0, 2, 1, 3, 4)

                t_cond_start = time.time()
                cond_t = self.conditioner(ids=cond, image=anchor_img, cond=cond2, drop=True)
                t_cond_end = time.time()
                timing_dict["timings/conditioner"] = t_cond_end - t_cond_start

            Y1 = latents
            vid_lengths = M1.long().sum(dim=1)

            vel_loss, rate_loss, means, means_y, flowception_timings = self.get_flowception_loss(
                Y1, M1, vid_lengths, cond_t, None
            )
            timing_dict.update(flowception_timings)

            loss = vel_loss + self.poisson_loss_weight * rate_loss

            if extra_batch is not None:
                loss_image = self.image_loss(model, extra_batch)
            else:
                loss_image = torch.zeros_like(loss)

            loss_all = loss + self.image_loss_weight * loss_image

        # 1. Local NaN check
        local_nan = torch.isnan(loss.detach())

        # 2. Communicate across ranks if any rank saw a NaN (skip if single-GPU)
        if torch.distributed.is_initialized():
            nan_flag = torch.tensor(local_nan, device=loss.device, dtype=torch.int)
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            any_nan = nan_flag.item() > 0
        else:
            any_nan = local_nan.item()

        if any_nan:
            # 3. The process that detected the NaN saves its batch/inputs
            if local_nan.item():
                self.logger.warning(
                    f"[Rank {self.accelerator.process_index}] Detected NaN, saving crash data"
                )

                crash_dir = Path(self.results_folder) / "crash_data_rank{}".format(
                    self.accelerator.process_index
                )
                os.makedirs(crash_dir, exist_ok=True)

                # Save batch and inputs
                torch.save(batch, crash_dir / "batch.pt")
                torch.save(
                    {
                        "latents": latents,
                        "cond_t": cond_t,
                    },
                    crash_dir / "inputs.pt",
                )

                self.logger.warning(
                    f"[Rank {self.accelerator.process_index}] Saved crash data to {crash_dir}"
                )

            # 4. Sync before saving model
            self.accelerator.wait_for_everyone()

            # 5. Save model checkpoint — all ranks participate
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                Path(self.results_folder) / "crash_data" / f"model_ckpt.bin",
                **get_fsdp_ckpt_kwargs(),
            )

            self.accelerator.wait_for_everyone()

            # 7. Exit cleanly
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
            self.optimizer.zero_grad(set_to_none=True)  # important with accumulation
        else:
            # Accumulating grads: no clipping, no stepping
            pass

        t_backward_end = time.time()
        timing_dict["timings/backward"] = t_backward_end - t_backward_start

        # Build logs safely: grad_norm exists only when we stepped
        loss_dict = {
            "train/loss": loss.detach(),
            "train/loss_image": loss_image.detach(),
            "train/vel_loss": vel_loss.detach(),
            "train/rate_loss": rate_loss.detach(),
        }

        # Either log it only when available...
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

                if self.num_context_frames > 0:
                    img = [sample.pixel_values for sample in samples]
                    img = torch.stack(img)
                    img = img.to(self.accelerator.device, non_blocking=True)
                    if img.ndim == 4:
                        img = img[:, :, None]  # img data
                    else:
                        img = img[:, :, :1]

                    anchor_img = img.clone()

                    with self.accelerator.autocast():
                        with torch.no_grad():
                            context_frames = (
                                (self.vae.encode(img).latent_dist.sample() - self.vae_shift_factor)
                                .mul_(self.vae_scale_factor)
                                .permute(0, 2, 1, 3, 4)
                            )
                else:
                    context_frames = None

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

                if self.insertion_rule == "learned":
                    Y_t, M_t, Y_t_l, M_t_l, ins_timemap, expected_l = vanilla_sample_flowception_t2v(
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
                        snr_shift=self.sigma_scale,
                    )
                else:
                    Y_t, M_t, Y_t_l, M_t_l, ins_timemap, expected_l = vanilla_sample_flowception_prescribed(
                        first_frames=context_frames,
                        model=self.model,
                        forward_fn=self.compute_model,
                        num_steps=num_steps,
                        cond_t=cond_t,
                        context_frames=context_frames,
                        batch_size=max(2, batch_size),
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
                        insertion_rule=self.insertion_rule,
                    )

                self.accelerator.wait_for_everyone()
                grid_latents.append(self.accelerator.gather(Y_t)[:grid_size])
                grid_masks.append(self.accelerator.gather(M_t)[:grid_size])

        grid_latents = torch.cat(grid_latents)[:grid_size].permute(0, 2, 1, 3, 4)  # (G, C, Tz, H, W)
        grid_masks = torch.cat(grid_masks)[:grid_size]  # (G, Tz) or (G, Tz, ...)

        # ---- decode (mask-aware if supported) ----
        z = self.vae_shift_factor + grid_latents / self.vae_scale_factor  # (G, C, Tz, H, W)
        latent_mask = grid_masks
        # ensure latent_mask is (G, Tz) bool on the right device
        if latent_mask.ndim > 2:  # e.g., (G, Tz, 1)
            latent_mask = latent_mask[..., 0]
        latent_mask = latent_mask.to(torch.bool).to(z.device)

        # derive temporal compression ratio (fallback to 8 if absent)
        tcr = getattr(self.vae, "temporal_compression_ratio", self.temporal_factor)

        # some VAEs don’t support latent_mask; some may have tiling enabled
        accepts_mask = vae_accepts_latent_mask(self.vae)

        # temporarily turn off tiling/framewise if we’re going to use a mask
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
            # guard if the VAE refuses masked decode (e.g., in a future config)
            with torch.no_grad(), self.accelerator.autocast():
                img = self.vae.decode(z).sample
        finally:
            # restore flags
            for k, v in restore_flags.items():
                setattr(self.vae, k, v)

        # ---- build display-time mask (white-out padded frames) ----
        G, C, T_img, H, W = img.shape
        T_lat = latent_mask.shape[1]

        # repeats: first latent -> 1 frame, others -> tcr frames
        repeats = torch.ones(T_lat, dtype=torch.long, device=img.device)
        if T_lat > 1:
            repeats[1:] = tcr

        # upsample latent mask to image time, then trim/pad to exactly T_img
        mask_img = torch.repeat_interleave(latent_mask, repeats=repeats, dim=1)  # (G, T≈(Tz-1)*tcr+1)
        if mask_img.shape[1] > T_img:
            mask_img = mask_img[:, :T_img]
        elif mask_img.shape[1] < T_img:
            pad = mask_img[:, -1:].expand(-1, T_img - mask_img.shape[1])
            mask_img = torch.cat([mask_img, pad], dim=1)

        # make masked frames white (use 0.0 for black if you prefer)
        mask5 = mask_img[:, None, :, None, None].to(img.dtype).to(img.device)  # (G,1,T,1,1)
        img = img * mask5 + (1 - mask5) * 1.0

        # ---- rest of your code (to numpy, gif writing, etc.) ----
        img = img.float().detach().cpu().numpy()
        img = (img + 1) / 2.0
        img = img.clip(0.0, 1.0)
        img_cat = np.concatenate([im for im in img], axis=3)
        if save_img:
            prefix = f"video_{sample_idx}.gif"
            savep = Path(self.results_dir) / "snapshots" / prefix
            self.logger.info(f"Saving to : {savep}")
            frames_to_gif(img_cat, save_path=savep, fps=self.sampling_fps)

        # Add image-level sampling.
        # detect T2I (strings) or allow explicit flag if you add one

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

        # --- build conditioning ---
        cond_t = self.conditioner.sample(batch_size=B, idx=captions, image=None)
        cond_t = {k: v for k, v in cond_t.items() if k != "mask"}
        uc = self.conditioner.get_cfg_version(cond_t)
        if keep_emb:
            uc["class_labels"][1] = cond_t["class_labels"][1].clone()

        # --- probe VAE ---
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

        # --- init ---
        Y_t = torch.randn(B, 1, C_z, H_z, W_z, device=device, dtype=lat.dtype)
        M_t = torch.ones(B, 1, dtype=torch.bool, device=device)

        # --- Compute schedule ONCE, as a Python list of floats ---
        # This ensures all ranks have identical values
        h_grid_cpu = torch.linspace(0, 1, num_steps + 1)
        h_grid_cpu = scale_snr(h_grid_cpu, sigma_scale=self.sigma_scale)
        h_grid_list = h_grid_cpu.tolist()  # Convert to Python floats

        # --- Pre-allocate t tensor (like working version) ---
        t = torch.zeros(B, 1, device=device, dtype=lat.dtype)

        for s in tqdm.tqdm(range(num_steps)):
            # Use Python floats, then fill tensor
            t.fill_(h_grid_list[s])
            h_eff = h_grid_list[s + 1] - h_grid_list[s]

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

            Y_t = Y_t + h_eff * vel  # h_eff is a scalar, broadcasts fine

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
        """
        Save paired GT and generated videos from the val loader to mp4 for later metrics.
        Also saves prompts next to them.
        - GT   -> <out_root>/gt_mp4/{prefix}_r{rank}_{idx:06d}.mp4
        - Gen  -> <out_root>/gen_mp4/{prefix}_r{rank}_{idx:06d}.mp4
        - Text -> <out_root>/prompts/{prefix}_r{rank}_{idx:06d}.txt
        - Meta -> <out_root>/prompts/{prefix}_r{rank}_{idx:06d}.json  (optional)
        """
        import json  # NEW

        device = self.accelerator.device
        rank = self.accelerator.process_index

        model = self.model
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.grad = None

        # IO setup
        if out_root is None:
            out_root = Path(self.results_folder) / "eval_samples"
        out_root = Path(out_root)
        gt_dir = out_root / "gt_mp4"
        gen_dir = out_root / "gen_mp4"
        prm_dir = out_root / "prompts"  # NEW
        gt_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)
        prm_dir.mkdir(parents=True, exist_ok=True)  # NEW

        # video fps
        fps = int(fps or getattr(self, "sampling_fps", 16))

        # convenience: to uint8 [T,H,W,3] from [-1,1] torch [C,T,H,W]
        def tensor_video_to_uint8(x: torch.Tensor) -> np.ndarray:
            x = x.clamp(-1, 1)
            x = (x + 1) / 2.0
            x = (x * 255.0).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            return x

        # guidance defaults
        s_text = s_text or self.cfg_scale
        s_img = s_img or self.image_guidance_scale

        local_idx = 0
        pbar = range(num_batches)
        if self.accelerator.is_main_process:
            pbar = tqdm.tqdm(pbar, desc="[generate_samples]")

        for _ in pbar:
            try:
                batch = next(iter(self.val_dataloader))
            except StopIteration:
                break

            # GT tensors
            img_gt = batch.pixel_values.to(device, non_blocking=True)  # [B,C,T,H,W]
            frame_mask = batch.condition["frame_mask"].to(device, non_blocking=True)
            if frame_mask.ndim == 1:
                frame_mask = frame_mask[None, :].expand(img_gt.shape[0], -1)
            B, C, T_gt, H, W = img_gt.shape

            # conditioning (we’ll save prompts from this)
            c = batch.condition["class_id"]
            cond2 = batch.condition["crop_coords"].to(device, non_blocking=True)
            cond_ids = c.to(device, torch.long, non_blocking=True) if isinstance(c, torch.Tensor) else c

            # build per-sample prompt strings (NEW)
            if isinstance(cond_ids, torch.Tensor):
                # numeric labels
                prompt_list = [f"label_id:{int(x)}" for x in cond_ids.detach().cpu().tolist()]
            elif isinstance(cond_ids, (list, tuple)):
                prompt_list = [str(x) for x in cond_ids]
            else:
                # single string replicated across batch (unlikely, but safe)
                prompt_list = [str(cond_ids) for _ in range(B)]

            # anchor / context
            anchor_img = img_gt[:, :, :1].contiguous()
            if self.num_context_frames > 0:
                with self.accelerator.autocast():
                    context_lat = (
                        self.vae.encode(anchor_img).latent_dist.sample() - self.vae_shift_factor
                    ).mul_(self.vae_scale_factor)
                context_frames = context_lat.permute(0, 2, 1, 3, 4).contiguous()
            else:
                context_frames = None

            captions = None if isinstance(cond_ids, torch.Tensor) else cond_ids
            cond_t = self.conditioner.sample(batch_size=B, idx=captions, image=anchor_img)
            cond_t = {k: v for k, v in cond_t.items() if k != "mask"}
            uc = self.conditioner.get_cfg_version(cond_t)

            # sample latents
            if self.guider is not None:
                self.guider.clean_buffer()
            with self.accelerator.autocast():
                s_offset = self.guidance_offset
                s_ins = self.guidance_insertion_scale

                Y_t, M_t, Y_t_l, M_t_l, ins_timemap, expected_l = vanilla_sample_flowception_t2v(
                    first_frames=context_frames,
                    model=self.model,
                    forward_fn=self.compute_model,
                    num_steps=num_steps,
                    cond_t=cond_t,
                    context_frames=context_frames,
                    batch_size=B,
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

            # decode latents to frames
            Y_t_bcdhw = Y_t.permute(0, 2, 1, 3, 4).contiguous()
            z = self.vae_shift_factor + Y_t_bcdhw / self.vae_scale_factor
            latent_mask = M_t
            # ensure latent_mask is (G, Tz) bool on the right device
            if latent_mask.ndim > 2:  # e.g., (G, Tz, 1)
                latent_mask = latent_mask[..., 0]
            latent_mask = latent_mask.to(torch.bool).to(Y_t.device)

            # derive temporal compression ratio (fallback to 8 if absent)
            tcr = getattr(self.vae, "temporal_compression_ratio", self.temporal_factor)

            # some VAEs don’t support latent_mask; some may have tiling enabled
            accepts_mask = vae_accepts_latent_mask(self.vae)

            # temporarily turn off tiling/framewise if we’re going to use a mask
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
                # guard if the VAE refuses masked decode (e.g., in a future config)
                with torch.no_grad(), self.accelerator.autocast():
                    vid_gen = self.vae.decode(z).sample
            finally:
                # restore flags
                for k, v in restore_flags.items():
                    setattr(self.vae, k, v)

            # latent mask -> frame mask
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

            # save pairs + prompts
            for b in range(B):
                file_id = f"{prefix}_r{rank:02d}_{local_idx:06d}"
                gt_path = gt_dir / f"{file_id}.mp4"
                gen_path = gen_dir / f"{file_id}.mp4"
                txt_path = prm_dir / f"{file_id}.txt"  # NEW
                jsn_path = prm_dir / f"{file_id}.json"  # NEW (optional)

                # GT frames (drop pad)
                mask_b = frame_mask[b] if frame_mask.shape[0] == B else frame_mask
                real_T_gt = int(mask_b.long().sum().item())
                vid_b_gt = img_gt[b, :, :real_T_gt]
                frames_gt = tensor_video_to_uint8(vid_b_gt)

                # GEN frames (drop pad)
                real_T_gen = int(mask_frames_gen[b].long().sum().item())
                vid_b_gen = vid_gen[b, :, :real_T_gen]
                frames_gen = tensor_video_to_uint8(vid_b_gen)

                # length parity (optional but handy for some metrics)
                min_len = min(frames_gt.shape[0], frames_gen.shape[0])
                frames_gt, frames_gen = frames_gt[:min_len], frames_gen[:min_len]

                # write videos (use your ffmpeg writer)
                write_frames_ffmpeg(frames_gt, str(gt_path), fps=fps)
                write_frames_ffmpeg(frames_gen, str(gen_path), fps=fps)

                # --- NEW: save prompt text + metadata ---
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

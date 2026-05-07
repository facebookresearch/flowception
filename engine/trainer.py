import gc
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from shutil import make_archive

import numpy as np
import torch
import tqdm
from accelerate.utils import save_fsdp_model
from data.datasets.dataloader import ConcatDataloader
from helpers.checkpoint import get_fsdp_ckpt_kwargs
from helpers.ema import update_ema
from PIL import Image
from torch.profiler import profile, ProfilerActivity


# Global flag for preemption detection
PREEMPTION_FLAG = {"flag": False}


def _preemption_handler(signum, frame):
    """Handler for SIGUSR1 signal sent by SLURM before preemption."""
    PREEMPTION_FLAG["flag"] = True
    logger = logging.getLogger("Flowception")
    logger.warning(f"Received preemption signal (SIGUSR1) on {socket.gethostname()}")


class Trainer:
    """Base trainer for diffusion models.

    Handles the training loop, EMA updates, checkpointing, evaluation, and
    snapshot generation. Subclassed by Flowception and FlowceptionT2V for
    video-specific training logic.
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
        self.accelerator = accelerator
        self.model = model
        self.ema = ema
        self.vae = vae

        self.logger = logger

        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.extra_dataloader = extra_dataloader
        self.dataloader_style = cfg.DATA.DATALOADER_STYLE
        assert self.dataloader_style in ["map_style", "iter_style"]

        self.image_size = cfg.SOLVER.IM_SIZE
        self.latent_size = self.image_size // cfg.MODEL.VAE.FACTOR
        self.latent_ch = cfg.MODEL.VAE.OUT_CH
        self.timesteps = cfg.FRAMEWORK.TIMESTEPS
        self.save_and_sample_every = cfg.SOLVER.SNAPSHOT_F

        self.gradient_accumulation_steps = cfg.SOLVER.ACCUMULATE
        self.save_and_sample_every = int(
            (self.save_and_sample_every // self.gradient_accumulation_steps)
            * self.gradient_accumulation_steps
        )

        self.log_every = cfg.SOLVER.LOG_EVERY
        self.device = device
        self.epochs = cfg.SOLVER.EPOCHS
        self.ckpt_freq = cfg.SOLVER.CKPT_EVERY
        self.permanent_ckpt_freq = cfg.SOLVER.KEEP_CKPT_EVERY
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.fid_sampler = cfg.EVAL.FID.SAMPLER
        self.fid_skip_factor = cfg.EVAL.FID.SKIP_F
        self.eval_fid = cfg.SOLVER.EVAL_FID
        self.ema_decay = cfg.SOLVER.EMA_DECAY
        self.ema_start = cfg.SOLVER.EMA_START
        self.log_target = "model"
        self.dec_batch_size = cfg.SOLVER.DEC_BATCH_SIZE
        self.text_zero_out = cfg.MODEL.TEXT_ENCODER.ZERO_OUT
        assert self.text_zero_out in ["none", "first", "last"]

        self.max_iter = cfg.SOLVER.MAX_ITER
        self.run_step = 0

        # in order not to kickoff sampling mid iteration when using gradient accumulation.
        self.global_step = int(
            (global_step // self.gradient_accumulation_steps)
            * self.gradient_accumulation_steps
        )
        self.local_step = int(
            (local_step // self.gradient_accumulation_steps)
            * self.gradient_accumulation_steps
        )

        self.vae_scale_factor = cfg.MODEL.VAE.SCALE_FACTOR
        self.vae_shift_factor = cfg.MODEL.VAE.SHIFT_FACTOR

        self.conditioner = conditioner
        self.start_epoch = start_epoch
        self.use_cfg = cfg.SOLVER.USE_CFG
        self.cfg_scale = cfg.SAMPLER.CFG_SCALE
        self.denoiser = None

        self.eval_freq = cfg.SOLVER.EVAL_FREQ

        self.clip_grad = cfg.SOLVER.CLIP_GRAD_NORM
        self.max_grad_norm = cfg.SOLVER.MAX_GRAD_NORM

        self.sigma_scale = cfg.FRAMEWORK.SIGMA_SCALE

        self.eval_groups = cfg.EVAL.GROUPS_COUNT

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.results_folder = output_dir
        if output_dir:
            self.results_dir = Path(output_dir)
            (self.results_dir / "snapshots").mkdir(parents=True, exist_ok=True)
            self.trace_dir = self.results_dir / "traces"
            self.trace_dir.mkdir(exist_ok=True)

        if self.permanent_ckpt_freq:
            assert self.permanent_ckpt_freq % self.ckpt_freq == 0

        # Setup preemption handler if running on SLURM
        if "SLURM_JOB_ID" in os.environ:
            signal.signal(signal.SIGUSR1, _preemption_handler)
            self.logger.info(
                f"Registered SIGUSR1 handler for job {os.environ['SLURM_JOB_ID']}"
            )

    def log_stats(self, loss_dict, tot_iter, epoch, step, timing=None):
        if torch.isnan(loss_dict["train/loss"]):
            raise ValueError("Loss has become nan!")
        if step % self.log_every == 0:
            log_dict = {
                "metrics/glob_iter": tot_iter,
                "metrics/learning_rate": self.optimizer.param_groups[-1]["lr"],
            }

            log_dict.update(
                {
                    k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in loss_dict.items()
                }
            )

            if timing is not None:
                log_dict.update(timing)

            self.accelerator.log(log_dict)

        if step % 100 == 0:
            self.logger.info(
                f"epoch : {epoch} - global iteration : {tot_iter} - local iteration : {step} "
                f"alloc_mem : {torch.cuda.memory_allocated()/1000**3:.2E} "
                f"max_alloc_mem : {torch.cuda.max_memory_allocated()/1000**3:.2E} "
                f"- loss : {loss_dict['train/loss']:.2E} "
                f"- learning rate : {self.optimizer.param_groups[-1]['lr']}"
            )

    @torch.no_grad()
    def checkpoint_state(self, epoch, local_step, global_step):
        self.accelerator.wait_for_everyone()

        ckpt_dir = self.results_dir / "checkpoint"
        if global_step % self.ckpt_freq == 0 and global_step > 0:
            # note: accelerator.save_state handles internally the multiprocess save
            self.accelerator.save_state(ckpt_dir)
            self.logger.info("Done saving state.")

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                with open(ckpt_dir / "epoch.txt", "w") as f:
                    f.write(str(epoch + 1))
                with open(ckpt_dir / "global_step.txt", "w") as f:
                    f.write(str(global_step + 1))
                with open(ckpt_dir / "local_step.txt", "w") as f:
                    f.write(str(local_step + 1))

            if (
                int(global_step) % int(self.permanent_ckpt_freq) == 0
                and global_step > 1
            ):
                os.makedirs(os.path.join(self.results_folder, "weights"), exist_ok=True)
                self.logger.info(f"Iter: {global_step} - Saving FSDP checkpoint.")
                save_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.model,
                    Path(self.results_folder)
                    / "weights"
                    / f"model_{global_step:06d}",
                    **get_fsdp_ckpt_kwargs(),
                )
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            gc.collect()
            # gc.freeze()

    def requeue_slurm_job(self) -> None:
        """Requeue the current SLURM job after checkpointing."""
        if "SLURM_PROCID" not in os.environ:
            self.logger.warning("Not running under SLURM, cannot requeue")
            return

        proc_id = int(os.environ["SLURM_PROCID"])
        self.logger.warning(f"Host: {socket.gethostname()} - Global rank: {proc_id}")

        # Only rank 0 should requeue
        if proc_id == 0:
            self.logger.warning(f"Requeuing job {os.environ['SLURM_JOB_ID']}")
            subprocess.run(
                ["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=False
            )
        else:
            self.logger.warning("Not the master process, no need to requeue.")

        sys.exit(0)

    def compute_val_metrics(self, model, epoch, sampler, skip_factor):
        raise NotImplementedError(
            "compute_val_metrics is not implemented. Set SOLVER.EVAL_FID=false to skip evaluation."
        )

    @torch.no_grad()
    def eval_step(self, epoch, tot_iter):
        """
        Performs an evaluation step.
        This function performs an evaluation step if the epoch count is divisible by
        the checkpoint frequency and the epoch count is greater than 0.
        It computes FID (Fréchet Inception Distance) for the model or the EMA depending on the log_target.
        Args:
            epoch (int): The current epoch count.
        """
        target_ = self.model if self.log_target == "model" else self.ema
        if self.eval_fid and tot_iter % self.eval_freq == 0 and tot_iter > 0:
            self.compute_val_metrics(
                model=target_,
                epoch=tot_iter,
                sampler=self.fid_sampler,
                skip_factor=self.fid_skip_factor,
            )

        self.accelerator.wait_for_everyone()

    def train(
        self,
    ):
        torch.backends.cudnn.benchmark = True

        self.logger.info("Start training.")
        self.model.train()
        self.ema.eval()

        # Initialization.
        self.logger.info("Initializing...")
        gc.disable()
        gc.freeze()

        self.logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0
            self.accelerator.wait_for_everyone()

            # Setup dataloader for epoch:
            if (
                isinstance(self.dataloader, ConcatDataloader)
                or self.dataloader_style == "map_style"
            ):
                self.dataloader.set_epoch(epoch + 1)
                if self.extra_dataloader is not None:
                    if hasattr(self.extra_dataloader, "pipeline"):
                        self.extra_dataloader.pipeline[0].dataset.pipeline[1].seed = (
                            epoch + 1 + self.accelerator.process_index
                        )
                        self.extra_dataloader.pipeline[0].dataset.pipeline[
                            0
                        ].reshuffle_shards(local_idx=self.accelerator.process_index)
                    else:
                        self.extra_dataloader.set_epoch(epoch + 1)
            else:
                if hasattr(self.dataloader, "pipeline"):
                    # setting seed during dataset instantiation instead.
                    self.dataloader.pipeline[0].dataset.pipeline[1].seed = (
                        epoch + 1 + self.accelerator.process_index
                    )
                    self.dataloader.pipeline[0].dataset.pipeline[0].reshuffle_shards(
                        local_idx=self.accelerator.process_index
                    )
                    if self.extra_dataloader is not None:

                        if hasattr(self.extra_dataloader, "pipeline"):
                            self.extra_dataloader.pipeline[0].dataset.pipeline[
                                1
                            ].seed = (epoch + 1 + self.accelerator.process_index)
                            self.extra_dataloader.pipeline[0].dataset.pipeline[
                                0
                            ].reshuffle_shards(local_idx=self.accelerator.process_index)
                        else:
                            self.extra_dataloader.set_epoch(epoch + 1)

            def none_iterator():
                while True:
                    yield None

            active_dataloader = self.dataloader
            extra_dataloader = (
                iter(self.extra_dataloader)
                if self.extra_dataloader is not None
                else none_iterator()
            )
            # Training iterations.
            t_data_start = time.time()
            for step, (batch, extra_batch) in enumerate(
                zip(active_dataloader, extra_dataloader)
            ):
                t_data_end = time.time()
                # Initialize EMA.
                if self.global_step == self.ema_start:
                    self.logger.info("Starting EMA updates.")
                    update_ema(self.ema, self.model, decay=0)

                with self.accelerator.accumulate(self.model):
                    if self.run_step == -1:
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            with_stack=True,
                            with_flops=True,
                        ) as prof:
                            train_tuple = self.train_step(
                                self.model,
                                batch=batch,
                                extra_batch=None,
                            )
                        prof.export_chrome_trace(
                            str(
                                self.trace_dir
                                / f"trace_{self.accelerator.process_index}.json"
                            )
                        )
                    else:
                        t_step_start = time.time()
                        train_tuple = self.train_step(
                            self.model,
                            batch=batch,
                            extra_batch=extra_batch,
                        )
                        t_step_end = time.time()
                        train_tuple.time_dict.update(
                            {"timings/train_step": t_step_end - t_step_start}
                        )
                train_tuple.time_dict.update(
                    {"timings/dataloading": t_data_end - t_data_start}
                )
                epoch_loss += train_tuple.loss

                self.local_step = step
                self.global_step += 1
                self.run_step += 1

                # Update EMA
                if self.global_step > self.ema_start:
                    t_ema_start = time.time()
                    update_ema(self.ema, self.model, decay=self.ema_decay)
                    t_ema_end = time.time()
                    train_tuple.time_dict.update(
                        {"timings/ema": t_ema_end - t_ema_start}
                    )

                # Save snapshots
                if (
                    self.global_step % self.save_and_sample_every == 0
                    and self.global_step > 1
                ):

                    if self.global_step >= self.ema_start:
                        self.log_target = "ema"
                    target_ = self.model if self.log_target == "model" else self.ema
                    self.logger.info(f"Generating snapshots with : {self.log_target}")
                    _ = self.sampling_step(
                        train_tuple.latents,
                        (self.global_step // self.save_and_sample_every) % 5,
                        model=target_,
                        use_cfg=self.use_cfg,
                    )

                # log training stats.
                self.log_stats(
                    loss_dict=train_tuple.loss_dict,
                    tot_iter=self.global_step,
                    epoch=epoch,
                    step=self.run_step,
                    timing=train_tuple.time_dict,
                )

                # Checkpoint state
                self.checkpoint_state(epoch, step, self.global_step)

                # Check for preemption signal
                if PREEMPTION_FLAG["flag"]:
                    self.logger.info(
                        f"Job pre-empted at step {self.global_step}! Checkpointing current state."
                    )
                    # Force a checkpoint regardless of schedule
                    self.accelerator.wait_for_everyone()
                    ckpt_dir = self.results_dir / "checkpoint"
                    self.accelerator.save_state(ckpt_dir)

                    if self.accelerator.is_main_process:
                        with open(ckpt_dir / "epoch.txt", "w") as f:
                            f.write(str(epoch + 1))
                        with open(ckpt_dir / "global_step.txt", "w") as f:
                            f.write(str(self.global_step + 1))
                        with open(ckpt_dir / "local_step.txt", "w") as f:
                            f.write(str(step + 1))

                    self.accelerator.wait_for_everyone()
                    self.requeue_slurm_job()

                # Evaluation.
                self.eval_step(epoch, self.global_step)

                if (self.global_step + 1) % 50 == 0:
                    gc_start_t = time.time()
                    gc.collect()
                    gc_end_t = time.time()
                    train_tuple.time_dict.update({"timings/gc": gc_end_t - gc_start_t})

                if self.global_step > self.max_iter:
                    raise ValueError("Reached max iteration !")
                t_data_start = time.time()

            # End of epoch
            epoch_loss = epoch_loss.detach().cpu().item()
            self.logger.info(f"end of epoch : {epoch} - epoch loss : {epoch_loss/step}")
            self.accelerator.log(
                {
                    "train/loss_epoch": epoch_loss / step,
                }
            )

    @torch.no_grad()
    def split_and_decode(self, latents):
        """
        Splits the latents into smaller batches and decodes them using the VAE.
        This function takes a tensor of latents, splits it into smaller batches to avoid
            out-of-memory errors, and then decodes each batch using the VAE.
        Args:
            latents (torch.Tensor): The tensor of latents to be split and decoded.
        Returns:
            torch.Tensor: The decoded latents.
        """
        return torch.cat(
            [
                self.vae.decode(
                    self.vae_shift_factor + latents_batch / self.vae_scale_factor
                ).sample
                for latents_batch in latents.split(self.dec_batch_size)
            ]
        )

    def save_snapshot(self, samples, prefix="snapshot_0.png"):
        images = samples[-1].cpu().numpy()
        del samples
        b, c, h, w = images.shape
        ch, cw = int(np.sqrt(b)), int(np.sqrt(b))
        images = images[: ch * cw]
        image_grid = images.reshape(ch, cw, c, h, w)
        image_grid = image_grid.transpose(0, 3, 1, 4, 2)
        image_grid = image_grid.reshape(ch * h, cw * w, c).clip(-1, 1)
        image_grid = (image_grid + 1) / 2.0

        if c == 1:
            image_grid = (image_grid[:, :, 0] * 255.0).astype(np.uint8)
            image_grid = Image.fromarray(image_grid, mode="L")
        else:
            image_grid = (image_grid * 255.0).astype(np.uint8)
            image_grid = Image.fromarray(image_grid)
        self.logger.info(f"Saving to : {Path(self.results_dir) / 'snapshots' / prefix}")
        image_grid.save(Path(self.results_dir) / "snapshots" / prefix)

    @torch.no_grad()
    def plot_progressive_grid(self, latents, num_samples, num_steps=6, save_to=None):
        timesteps = len(latents)
        steps = list(range(0, timesteps, timesteps // num_steps))
        if steps[-1] != timesteps - 1:
            steps[-1] = timesteps - 1
        images = []
        for t in steps:
            with self.accelerator.autocast():
                dec_img = (
                    self.split_and_decode(latents[t][:num_samples])
                    .detach()
                    .cpu()
                    .numpy()
                )
            dec_img = (dec_img + 1.0) / 2.0
            dec_img = dec_img.transpose(0, 2, 3, 1).clip(0, 1)
            images.append(dec_img)
        stacked_images = np.stack(images)
        stacked_images = stacked_images.transpose(1, 2, 0, 3, 4)
        b, w, ns, h, c = stacked_images.shape
        stacked_images = stacked_images.reshape(b * w, ns * h, c)
        return Image.fromarray((255 * stacked_images).astype(np.uint8))

    def save_images_for_fid(self, samples, step, directory):
        images = samples[-1].cpu().numpy()
        b, c, h, w = images.shape
        images = images.transpose(0, 2, 3, 1)
        images = (images + 1.0) / 2.0
        images = (255.0 * images).clip(0, 255).astype(np.uint8)
        for idx, img_np in enumerate(images):
            Image.fromarray(img_np).save(
                Path(directory)
                / f"sample_{step}_{idx}_dev_{self.accelerator.process_index}.png"
            )

    def renormalize(self, x):
        # inputs are scaled in [-1, 1]
        x = (x + 1.0) / 2.0
        x = x.clip(0, 1).to(torch.float32, non_blocking=True)
        return x

    def save_samples(self, samples, captions=None):
        os.makedirs(Path(self.results_folder) / "generations", exist_ok=True)
        os.makedirs(
            Path(self.results_folder) / "generations" / "captions", exist_ok=True
        )
        images = samples[-1].permute(0, 2, 3, 1).cpu().numpy()
        del samples
        images = (images + 1.0) / 2.0
        images = images.clip(0.0, 1.0)

        for i, image in tqdm.tqdm(enumerate(images)):
            Image.fromarray((image * 255.0).astype(np.uint8)).save(
                Path(self.results_folder) / "generations" / f"{i:06d}.png"
            )
            with open(
                Path(self.results_folder) / "generations" / "captions" / f"{i:06d}.txt",
                "w",
            ) as f:
                f.write(captions[i])

        fname = str(self.results_folder).split("/")[-1]
        make_zip_archive(
            source=Path(self.results_folder) / "generations",
            destination=Path(self.results_folder) / f"{fname}.zip",
        )


def make_zip_archive(source: Path, destination: Path) -> None:
    """Create a ZIP archive from *source* directory at *destination*."""
    base_name = destination.parent / destination.stem
    fmt = destination.suffix.replace(".", "")
    root_dir = source.parent
    base_dir = source.name
    make_archive(str(base_name), fmt, root_dir, base_dir)

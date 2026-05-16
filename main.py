import argparse
import logging
import os
import sys
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch._dynamo as dynamo
import yaml
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
)
from config.default import get_cfg_defaults
from data.datasets.dataloader import ConcatDataloader, get_train_dataloader
from data.datasets.dataset import get_all_datasets, get_extra_datasets
from engine.flowception import Flowception
from engine.flowception_interpolate import FlowceptionInterpolate
from engine.flowception_t2v import FlowceptionT2V
from helpers.checkpoint import analyze_and_prune_checkpoint
from helpers.lr_schedulers import get_lr_scheduler
from helpers.utils import parse_ckpt
from modules.conditioning import I2VConditioner, INConditioner
from modules.get_model import get_denoiser
from modules.vae import get_vae
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

dynamo.config.cache_size_limit = 64

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, default="output", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", action="store_true", help="resume from logdir checkpoint")
    parser.add_argument(
        "-rf",
        "--resume_from",
        type=str,
        default="",
        help="resume from specific checkpoint dir",
    )
    parser.add_argument("-c", "--config", metavar="config.yaml", help="path to config YAML", default="")
    parser.add_argument(
        "-a",
        "--append",
        nargs="*",
        help="config overrides as KEY VALUE pairs",
        default=list(),
    )
    parser.add_argument("-t", "--train", action="store_true", help="train")
    parser.add_argument("--no-test", action="store_true", help="disable test")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="data logging directory")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="scale lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument("--sample_only", action="store_true", help="generate samples and exit")
    return parser


def parse_arg_changes(arg_list):
    """Convert CLI config overrides to properly typed values."""
    for k in range(len(arg_list)):
        if arg_list[k].isnumeric():
            arg_list[k] = int(arg_list[k])
        elif arg_list[k].replace(".", "").isnumeric():
            arg_list[k] = float(arg_list[k])
        elif arg_list[k].lower() in ("yes", "true", "t", "y"):
            arg_list[k] = True
        elif arg_list[k].lower() in ("no", "false", "f", "n"):
            arg_list[k] = False
        elif arg_list[k].lower() == "[]":
            arg_list[k] = []
        elif arg_list[k] and arg_list[k][0] == "[" and arg_list[k][-1] == "]":
            arg_list[k] = arg_list[k][1:-1].split()
    return arg_list


# ──────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"


def setup_accelerator(cfg, args):
    """Create and configure the HuggingFace Accelerator with FSDP."""
    set_fsdp_env()

    kwargs = [
        InitProcessGroupKwargs(timeout=timedelta(seconds=500000)),
    ]

    accel_loggers = {}
    if cfg.SOLVER.LOG_WANDB:
        accel_loggers = {
            "wandb": {
                "name": args.name,
                "entity": os.environ.get("WANDB_ENTITY", None),
                "id": cfg.RUN_ID,
                "resume": "allow",
            }
        }

    fsdp_plugin = FullyShardedDataParallelPlugin(
        cpu_offload=False,
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(
        mixed_precision=cfg.SOLVER.AMP_TYPE if cfg.SOLVER.AMP else "no",
        gradient_accumulation_steps=cfg.SOLVER.ACCUMULATE,
        kwargs_handlers=kwargs,
        log_with=list(accel_loggers.keys()) or None,
        rng_types=[],
        fsdp_plugin=fsdp_plugin,
        device_placement=False,
    )
    accelerator.init_trackers(project_name="Flowception", config=cfg, init_kwargs=accel_loggers or None)
    return accelerator


def setup_dataloaders(cfg, args, accelerator, logger):
    """Load train/val/extra datasets and wrap in accelerator-prepared dataloaders."""
    batch_size = cfg.SOLVER.BATCH_SIZE
    extra_batch_size = cfg.SOLVER.EXTRA_BATCH_SIZE
    workers = cfg.DATA.WORKERS
    prefetch_factor = cfg.SOLVER.PREFETCH if workers > 0 else None

    train_datasets, val_dataset = get_all_datasets(
        cfg, logger=logger, num_gpus=accelerator.num_processes, seed=args.seed
    )

    if len(train_datasets) == 1:
        dataloader = accelerator.prepare(
            get_train_dataloader(
                dataset_name=cfg.DATA.DATASET,
                dataset=train_datasets[0],
                batch_size=batch_size,
                workers=workers,
                prefetch_factor=prefetch_factor,
            )
        )
    else:
        dataloader = ConcatDataloader(
            [
                accelerator.prepare(
                    get_train_dataloader(
                        dataset_name=dataset_name,
                        dataset=train_dataset,
                        batch_size=batch_size,
                        workers=(0 if workers == 0 else max(workers // len(train_datasets), 1)),
                        prefetch_factor=prefetch_factor,
                    )
                )
                for train_dataset, dataset_name in zip(
                    train_datasets, cfg.DATA.DATASET.lower().split(","), strict=False
                )
            ],
            seed=accelerator.process_index + args.seed,
        )

    val_dataloader = accelerator.prepare(
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=cfg.EVAL.FID.TARGET_DATASET == "train",
            num_workers=0,
            prefetch_factor=None,
            drop_last=True,
        )
    )

    extra_dataloader = None
    if cfg.DATA.EXTRA_DATASETS is not None and cfg.DATA.EXTRA_DATASETS.lower() != "none":
        extra_datasets = get_extra_datasets(
            cfg, logger=logger, num_gpus=accelerator.num_processes, seed=args.seed
        )
        if len(extra_datasets) == 1:
            extra_dataloader = accelerator.prepare(
                get_train_dataloader(
                    dataset_name=cfg.DATA.EXTRA_DATASETS,
                    dataset=extra_datasets[0],
                    batch_size=extra_batch_size,
                    workers=workers,
                    prefetch_factor=prefetch_factor,
                )
            )
        else:
            extra_dataloader = ConcatDataloader(
                [
                    accelerator.prepare(
                        get_train_dataloader(
                            dataset_name=cfg.DATA.DATASET.lower().split(",")[0],
                            dataset=train_dataset,
                            batch_size=extra_batch_size,
                            workers=(0 if workers == 0 else max(workers // len(train_datasets), 1)),
                            prefetch_factor=prefetch_factor,
                        )
                    )
                    for train_dataset, dataset_name in zip(
                        train_datasets,
                        cfg.DATA.DATASET.lower().split(","),
                        strict=False,
                    )
                ],
                seed=accelerator.process_index + args.seed,
            )

    return dataloader, val_dataloader, extra_dataloader


def setup_model(cfg, accelerator, logger):
    """Instantiate model, load pretrained weights if specified, and create EMA copy."""
    model = get_denoiser(cfg, device=accelerator.device)
    model.to(accelerator.device)
    logger.info(f"Instantiated model, parameter count: {sum(p.numel() for p in model.parameters())}")

    if cfg.MODEL.WEIGHTS != "":
        ckpt_target = model
        logger.info(f"Loading weights from {cfg.MODEL.WEIGHTS}")
        ckpt = parse_ckpt(cfg.MODEL.WEIGHTS, device="cpu")

        analyzed_ckpt = analyze_and_prune_checkpoint(
            ckpt_target,
            ckpt_state=ckpt,
            logger=logger,
            allow_dtype_mismatch=True,
            keep_only_matching=not cfg.MODEL.STRICT_WEIGHTS,
        )

        if cfg.MODEL.STRICT_WEIGHTS:
            ckpt_target.load_state_dict(ckpt, strict=True)
        else:
            tmp_report = ckpt_target.load_state_dict(analyzed_ckpt, strict=False)
            logger.info(
                f"Load report — missing: {len(tmp_report.missing_keys)}, "
                f"unexpected: {len(tmp_report.unexpected_keys)}"
            )

    ema = deepcopy(model).to(accelerator.device)
    ema.requires_grad_(False)

    return model, ema


def setup_conditioner(cfg, accelerator):
    """Instantiate the conditioning module based on trainer type."""
    trainer_name = cfg.SOLVER.TRAINER.lower()
    flowception_trainers = [
        "flowception",
        "flowception_t2v",
        "flowception_interpolate",
    ]

    if trainer_name in flowception_trainers:
        return I2VConditioner(cfg, device=accelerator.device)
    else:
        return INConditioner(cfg, device=accelerator.device)


def setup_fsdp(cfg, model, ema, conditioner, accelerator, logger):
    """Wrap model, EMA, and optionally conditioner in FSDP."""
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    if hasattr(model, "ltx") and hasattr(model.ltx, "transformer_blocks"):
        wrap_classes = {
            type(model.ltx.transformer_blocks[0]),
            type(model.ltx.transformer_blocks[-1]),
        }
    elif hasattr(model, "blocks"):
        wrap_classes = {type(model.blocks[0]), type(model.blocks[-1])}
    else:
        raise ValueError("Cannot determine transformer block type for FSDP wrapping.")

    block_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=wrap_classes,
    )

    fsdp_kwargs = dict(
        auto_wrap_policy=block_auto_wrap_policy,
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        sync_module_states=True,
    )

    if cfg.SOLVER.AMP:
        fsdp_mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        fsdp_kwargs["mixed_precision"] = fsdp_mp

    model = FSDP(model, **fsdp_kwargs)
    ema = FSDP(ema, **fsdp_kwargs)

    if accelerator.is_main_process:
        logger.info(f"Denoiser architecture:\n{model}")

    # Optionally wrap conditioner text encoder in FSDP
    wrap_conditioner = cfg.SOLVER.FSDP.WRAP_CONDITIONER and (conditioner._type == "text")
    if wrap_conditioner:
        llama_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
        for attr in ("transformer", "llama_embedder"):
            if hasattr(conditioner.embedder, attr):
                setattr(
                    conditioner.embedder,
                    attr,
                    FSDP(
                        getattr(conditioner.embedder, attr),
                        auto_wrap_policy=llama_policy,
                        use_orig_params=False,
                        sharding_strategy=ShardingStrategy.FULL_SHARD,
                        forward_prefetch=True,
                    ),
                )

    # torch.compile
    if cfg.SOLVER.COMPILE_MODELS:
        if hasattr(conditioner, "embedder"):
            for attr in ("transformer", "llama_embedder", "t5_embedder", "encoder"):
                if hasattr(conditioner.embedder, attr):
                    setattr(
                        conditioner.embedder,
                        attr,
                        torch.compile(getattr(conditioner.embedder, attr)),
                    )

    model, ema = accelerator.prepare(model, ema)
    if cfg.SOLVER.COMPILE_MODELS:
        model = torch.compile(model)
        ema = torch.compile(ema)

    return model, ema


def setup_optimizer(cfg, model, accelerator):
    """Instantiate optimizer and LR scheduler."""
    optim_name = cfg.SOLVER.OPTIM.lower()
    optim_kwargs = dict(
        lr=cfg.SOLVER.LR,
        betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        fused=True,
    )

    if optim_name == "adam":
        optimizer = Adam(model.parameters(), **optim_kwargs)
    elif optim_name == "adamw":
        optimizer = AdamW(model.parameters(), weight_decay=cfg.SOLVER.WEIGHT_DECAY, **optim_kwargs)
    else:
        raise ValueError(f"Optimizer {cfg.SOLVER.OPTIM} not recognized")

    scheduler = get_lr_scheduler(cfg.SOLVER.LR_SCHEDULER.NAME)(
        optimizer=optimizer,
        start_epoch=cfg.SOLVER.LR_SCHEDULER.START_ITER,
        decay_length=cfg.SOLVER.LR_SCHEDULER.DECAY_LENGTH,
        min_lr=cfg.SOLVER.LR_SCHEDULER.MIN_LR,
        warmup_length=cfg.SOLVER.LR_SCHEDULER.WARMUP_LENGTH,
        num_processes=accelerator.num_processes,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    return optimizer, scheduler


def resume_from_checkpoint(cfg, args, accelerator, ema, logger):
    """Attempt to resume training state from a checkpoint directory."""
    start_epoch = 0
    local_step = 0
    global_step = 0

    output_dir = Path(args.logdir) / args.name
    ckpt_dir = Path(output_dir) / "checkpoint"

    resume_dir = None
    if Path(ckpt_dir).exists():
        resume_dir = ckpt_dir
    elif args.resume_from != "" and Path(args.resume_from).exists():
        resume_dir = Path(args.resume_from)

    if resume_dir is not None:
        logger.info(f"Resuming from: {resume_dir}")
        accelerator.load_state(resume_dir)

        epoch_file = resume_dir / "epoch.txt"
        if epoch_file.exists():
            start_epoch = int(epoch_file.read_text())

        global_step_file = resume_dir / "global_step.txt"
        if global_step_file.exists():
            global_step = int(global_step_file.read_text())

        local_step_file = resume_dir / "local_step.txt"
        if local_step_file.exists():
            local_step = int(local_step_file.read_text())

        ema_file = resume_dir / "ema.pt"
        if ema_file.exists():
            ema_ckpt = torch.load(ema_file, map_location="cpu")
            ema.load_state_dict(ema_ckpt["ema"])
            del ema_ckpt

    return start_epoch, local_step, global_step


_TRAINER_REGISTRY = {
    "flowception": Flowception,
    "flowception_t2v": FlowceptionT2V,
    "flowception_interpolate": FlowceptionInterpolate,
}


def get_trainer_class(cfg):
    """Look up the trainer class from the config."""
    name = cfg.SOLVER.TRAINER.lower()
    if name not in _TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer '{name}'. Available: {list(_TRAINER_REGISTRY.keys())}")
    return _TRAINER_REGISTRY[name]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def train(cfg, args):
    logger = get_logger("Flowception")
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m/%d/%Y-%H:%M:%S",
        level=logging.INFO if cfg.LOGGING.LOG_LEVEL == "INFO" else logging.DEBUG,
        stream=sys.stdout,
    )

    accelerator = setup_accelerator(cfg, args)
    accelerator.print(f"\nCommand line args:\n{args}\n")
    accelerator.print(cfg.dump())

    logger.info(f"Using random seed: {args.seed}")
    set_seed(seed=args.seed, device_specific=True)

    # Data
    logger.info("Loading datasets.")
    dataloader, val_dataloader, extra_dataloader = setup_dataloaders(cfg, args, accelerator, logger)

    # Model + EMA
    logger.info("Instantiating model.")
    model, ema = setup_model(cfg, accelerator, logger)

    # VAE
    logger.info("Instantiating autoencoder.")
    vae = get_vae(cfg, device=accelerator.device).eval().to(accelerator.device)
    vae.requires_grad_(False)

    # Conditioner
    logger.info("Instantiating conditioner.")
    conditioner = setup_conditioner(cfg, accelerator)

    # FSDP wrapping (requires torch.distributed; skip for single-GPU runs)
    output_dir = Path(args.logdir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(cfg.dump())

    if torch.distributed.is_initialized():
        model, ema = setup_fsdp(cfg, model, ema, conditioner, accelerator, logger)
    else:
        logger.info("Skipping FSDP (torch.distributed not initialized, single-GPU mode).")
    if cfg.SOLVER.COMPILE_MODELS:
        vae = torch.compile(vae)

    # Optimizer + scheduler
    logger.info("Instantiating optimizer.")
    optimizer, scheduler = setup_optimizer(cfg, model, accelerator)

    # Resume
    logger.info("Checking for checkpoint to resume.")
    start_epoch, local_step, global_step = resume_from_checkpoint(cfg, args, accelerator, ema, logger)

    # Instantiate trainer
    logger.info("Instantiating trainer.")
    trainer_fn = get_trainer_class(cfg)
    trainer = trainer_fn(
        cfg,
        accelerator,
        model=model,
        ema=ema,
        conditioner=conditioner,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        extra_dataloader=extra_dataloader,
        optimizer=optimizer,
        device=accelerator.device,
        vae=vae,
        scheduler=scheduler,
        output_dir=output_dir,
        start_epoch=start_epoch,
        logger=logger,
        local_step=local_step,
        global_step=global_step,
    )

    # Run
    if args.sample_only:
        logger.info(f"Generating {cfg.EVAL.NUM_SAMPLES} samples.")
        trainer.generate_samples(
            num_batches=cfg.EVAL.NUM_SAMPLES // cfg.SOLVER.BATCH_SIZE,
            num_steps=cfg.SAMPLER.NUM_STEPS,
        )
    else:
        logger.info("Starting training.")
        trainer.train()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    arg_changes = parse_arg_changes(args.append)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(arg_changes)
    cfg.freeze()

    train(cfg, args)

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Fastest FSDP-aware EMA update for use_orig_params=True.
    Each rank updates only its local parameter shards - zero communication overhead.
    """
    # Unwrap torch.compile if present
    actual_model = getattr(model, "_orig_mod", model)
    actual_ema = getattr(ema_model, "_orig_mod", ema_model)

    # Direct parameter iteration - with use_orig_params=True, each rank
    # only sees its shard, so this naturally only updates local data
    ema_params = dict(actual_ema.named_parameters())
    model_params = dict(actual_model.named_parameters())

    for name in ema_params.keys():
        if name in model_params:
            # In-place multiply-add: fastest operation
            ema_params[name].mul_(decay).add_(model_params[name], alpha=1.0 - decay)

"""Checkpoint loading and analysis utilities."""

import inspect
from collections import defaultdict

import torch
from accelerate.utils import save_fsdp_model
from transformers.utils import is_accelerate_available


def get_fsdp_ckpt_kwargs():
    """Return extra kwargs accepted by ``save_fsdp_model`` (version-dependent)."""
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    return {}


def analyze_and_prune_checkpoint(
    ckpt_target,
    ckpt_state,
    logger,
    *,
    allow_dtype_mismatch=True,
    keep_only_matching=True,
    max_examples=20,
):
    """
    Compare checkpoint keys against model and optionally filter to matching ones.

    Args:
        ckpt_target: nn.Module to load into.
        ckpt_state: flat dict (key -> Tensor) from checkpoint.
        logger: logger instance.
        allow_dtype_mismatch: keep same-shape tensors even if dtypes differ.
        keep_only_matching: return a filtered dict of only shape-matching keys.

    Returns:
        Filtered or original checkpoint dict.
    """
    model_state = ckpt_target.state_dict()

    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())

    only_in_ckpt = sorted(list(ckpt_keys - model_keys))
    only_in_model = sorted(list(model_keys - ckpt_keys))
    in_both = sorted(list(model_keys & ckpt_keys))

    shape_mismatch = []
    dtype_mismatch = []
    ok_keys = []

    for k in in_both:
        v_ck = ckpt_state[k]
        v_md = model_state[k]
        if v_ck.shape != v_md.shape:
            shape_mismatch.append((k, tuple(v_ck.shape), tuple(v_md.shape)))
        else:
            if v_ck.dtype != v_md.dtype and not allow_dtype_mismatch:
                dtype_mismatch.append((k, str(v_ck.dtype), str(v_md.dtype)))
            else:
                ok_keys.append(k)

    def group_by_prefix(keys):
        buckets = defaultdict(list)
        for k in keys:
            prefix = k.rsplit(".", 1)[0] if "." in k else k
            buckets[prefix].append(k)
        return {p: sorted(v) for p, v in buckets.items()}

    def _log_examples(title, items, formatter=lambda x: x, limit=max_examples):
        logger.info(f"{title}: {len(items)}")
        for itm in items[:limit]:
            logger.info("  " + formatter(itm))
        if len(items) > limit:
            logger.info(f"  ... and {len(items) - limit} more")

    _log_examples("Only in checkpoint (unexpected)", only_in_ckpt)
    _log_examples("Only in model (missing)", only_in_model)
    _log_examples(
        "Shape mismatches",
        shape_mismatch,
        formatter=lambda t: f"{t[0]}: ckpt{t[1]} vs model{t[2]}",
    )
    _log_examples(
        "Dtype mismatches",
        dtype_mismatch,
        formatter=lambda t: f"{t[0]}: ckpt {t[1]} vs model {t[2]}",
    )

    ckpt_prefix_counts = {p: len(v) for p, v in group_by_prefix(only_in_ckpt).items()}
    model_prefix_counts = {p: len(v) for p, v in group_by_prefix(only_in_model).items()}
    if ckpt_prefix_counts:
        logger.info(
            "Unexpected key prefixes (prefix: count): "
            + ", ".join(f"{p}: {n}" for p, n in sorted(ckpt_prefix_counts.items()))
        )
    if model_prefix_counts:
        logger.info(
            "Missing key prefixes (prefix: count): "
            + ", ".join(f"{p}: {n}" for p, n in sorted(model_prefix_counts.items()))
        )

    logger.info(f"Matched keys (ok to load): {len(ok_keys)}")
    logger.info(
        f"Summary — total ckpt keys: {len(ckpt_keys)}, "
        f"only_in_ckpt: {len(only_in_ckpt)}, only_in_model: {len(only_in_model)}, "
        f"shape_mismatch: {len(shape_mismatch)}, dtype_mismatch: {len(dtype_mismatch)}, "
        f"loadable: {len(ok_keys)}"
    )

    if keep_only_matching:
        return {k: ckpt_state[k] for k in ok_keys}
    else:
        return ckpt_state

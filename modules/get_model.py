import importlib

try:
    from diffusers.models import UNet2DModel
except ImportError:
    UNet2DModel = None

from modules.metadit_flowception import FlowceptionV1_models


def _try_import_models(module_path, attr_name):
    """Import an optional model registry dict; return an empty registry if it is unavailable."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name, {})
    except (ImportError, ModuleNotFoundError):
        return {}


ltxn_98_models = _try_import_models("modules.ltx_flowception", "ltxn_98_models")


def _hidden_dim_2(cfg):
    hidden_dim_2 = cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM_2
    return hidden_dim_2 if hidden_dim_2 and hidden_dim_2 > 0 else None


def _flowception_kwargs(cfg, device):
    return {
        "num_classes": "text_cond",
        "learn_sigma": False,
        "class_dropout_prob": 0.0,
        "text_encoder_dim": cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM,
        "text_encoder_dim_2": _hidden_dim_2(cfg),
        "input_size": cfg.SOLVER.IM_SIZE // cfg.MODEL.VAE.FACTOR,
        "embed_type": cfg.MODEL.EMBED_TYPE,
        "act_checkpoint": cfg.SOLVER.CKPT_ACT,
        "in_channels": cfg.MODEL.VAE.OUT_CH,
        "add_refiner": cfg.MODEL.ADD_REFINER,
        "depth_patch_size": cfg.MODEL.TEMPORAL_DOWNSCALING,
        "attention_mask": cfg.MODEL.VIDEO.ATTENTION_MASK,
        "device": device,
        "repa_layer": cfg.FRAMEWORK.VIDEO.REPA.LAYER,
        "repa_dim": cfg.FRAMEWORK.VIDEO.REPA.EMB_DIM,
        "rope_theta": cfg.MODEL.VIDEO.ROPE_THETA,
        "rope_mode": cfg.MODEL.VIDEO.ROPE_MODE,
        "h_mult": cfg.MODEL.VIDEO.ROPE_H_MULT,
        "w_mult": cfg.MODEL.VIDEO.ROPE_W_MULT,
        "t_mult": cfg.MODEL.VIDEO.ROPE_T_MULT,
        "add_y_emb": cfg.MODEL.VIDEO.POOL_Y_EMB,
    }


def _ltx_kwargs(cfg):
    return {
        "in_channels": cfg.MODEL.VAE.OUT_CH,
        "act_checkpoint": cfg.SOLVER.CKPT_ACT,
        "fps": cfg.DATA.VIDEO.SAMPLING_FPS,
        "checkpoint_path": cfg.MODEL.WEIGHTS or None,
        "fetch_pretrained": cfg.MODEL.FETCH_PRETRAINED,
    }


def get_denoiser(cfg, device="cpu"):
    """Instantiate one of the public denoiser backbones."""
    condition = cfg.MODEL.CONDITION
    public_key = condition.replace("T2I-", "")

    if condition.lower() == "class":
        if UNet2DModel is None:
            raise ImportError("diffusers is required for the class-conditional UNet fallback.")
        return UNet2DModel(
            sample_size=cfg.SOLVER.IM_SIZE,
            in_channels=cfg.MODEL.VAE.OUT_CH,
            out_channels=cfg.MODEL.VAE.OUT_CH,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            num_class_embeds=cfg.DATA.NUM_CLASSES + 1 if cfg.SOLVER.USE_CFG else cfg.DATA.NUM_CLASSES,
        )

    if public_key in FlowceptionV1_models:
        return FlowceptionV1_models[public_key](**_flowception_kwargs(cfg, device))

    if public_key in ltxn_98_models:
        return ltxn_98_models[public_key](**_ltx_kwargs(cfg))

    supported = sorted(
        set(FlowceptionV1_models)
        | {f"T2I-{name}" for name in FlowceptionV1_models}
        | set(ltxn_98_models)
    )
    raise ValueError(f"Unsupported MODEL.CONDITION '{condition}'. Supported public models: {supported}")

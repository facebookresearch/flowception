"""Conditioning modules for Flowception.

Provides text/image conditioners that wrap various text encoders (CLIP, LLaMA,
T5, DINOv2) and produce conditioning dicts consumed by the denoiser.

Classes:
    BaseConditioner: Shared init, CFG dropout, crop params generation.
    INConditioner: Class/text-conditioned (no image input).
    I2VConditioner: Image-to-video conditioned (text + image input).
    I2V2Conditioner: Like I2VConditioner but with cached unconditional embeddings.
"""

import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

try:
    from data.datasets.imagenet_classes import id2txt
except (ImportError, ModuleNotFoundError):
    id2txt = {}  # dummy fallback
try:
    from modules.size_embed import ConcatTimestepEmbedderND
except (ImportError, ModuleNotFoundError):
    ConcatTimestepEmbedderND = None  # only needed for sdxl conditioning

try:
    from modules.text_embedders.t5 import ByT5Embedder
except (ImportError, ModuleNotFoundError):
    ByT5Embedder = None  # only needed for Llama3_and_ByT5 embedder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Text / image embedders
# ---------------------------------------------------------------------------


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """CLIP ViT-L/14 text encoder (frozen)."""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        return self.transformer(input_ids=tokens).last_hidden_state

    def encode(self, text):
        return self(text)


class T5Embedder(AbstractEncoder):
    """T5 encoder-only text embedder (frozen)."""

    def __init__(self, version="google/flan-t5-xl", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        return self.transformer(input_ids=tokens).last_hidden_state

    def encode(self, text):
        return self(text)


class Llama3Embedder8B(AbstractEncoder):
    """LLaMA 3 8B decoder-only text embedder (frozen)."""

    def __init__(self, version="meta-llama/Meta-Llama-3-8B", device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_ = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.transformer = AutoModelForCausalLM.from_pretrained(version).get_decoder().to(device, dtype_)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        return self.transformer(input_ids=tokens).last_hidden_state

    def encode(self, text):
        return self(text)


class Llama3P2Embedder(AbstractEncoder):
    """LLaMA 3.2 3B-Instruct text embedder (frozen)."""

    def __init__(
        self, version="meta-llama/Llama-3.2-3B-Instruct", device="cuda", max_length=250, dtype="bf16"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_ = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.transformer = AutoModelForCausalLM.from_pretrained(version).get_decoder().to(device, dtype_)
        self.device = device
        self.max_length = max_length
        self.freeze()
        self.pre_text = ""
        self.post_text = ""

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def process_text(self, text):
        return [self.pre_text + line + self.post_text for line in text]

    def forward(self, text, image=None):
        text = self.process_text(text)
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        return self.transformer(input_ids=tokens).last_hidden_state

    def encode(self, text, image=None):
        return self(text, image)


class Llama3P2_and_DINOV2_Embedder(AbstractEncoder):
    """LLaMA 3.2 text + DINOv2 image embedder for I2V conditioning."""

    def __init__(
        self,
        version="meta-llama/Llama-3.2-3B-Instruct",
        device="cuda",
        max_length=77,
        dtype="bf16",
        img_res=252,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_ = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.transformer = AutoModelForCausalLM.from_pretrained(version).get_decoder().to(device, dtype_)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.img_embed_dim = 1024
        self.img_res = img_res
        self.pre_text = ""
        self.post_text = ""

        self.dino = torch.compile(
            torch.hub.load(
                "facebookresearch/dinov2:main",
                "dinov2_vitl14_reg",
                force_reload=False,
            ).to(self.device)
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def process_text(self, text):
        return [self.pre_text + str(line) + self.post_text for line in text]

    def forward(self, text, image):
        text = self.process_text(text)
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        z_txt = self.transformer(input_ids=tokens).last_hidden_state

        if image is not None:
            img = F.interpolate(
                image[:, :, 0], size=(self.img_res, self.img_res), mode="bicubic", align_corners=True
            )
            mu = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
            sigma = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
            img = (img + 1.0) / 2.0
            img = (img - mu) / sigma
            with torch.no_grad():
                o = self.dino.forward_features(img)
            z_im = o["x_norm_patchtokens"][:, None:]
        else:
            z_im = torch.zeros(z_txt.shape[0], 324, self.img_embed_dim, device=z_txt.device)

        return [z_txt, z_im]

    def encode(self, text, image=None):
        return self(text, image)


class Llama3_and_T5(AbstractEncoder):
    """LLaMA 3 + T5 dual text embedder."""

    def __init__(self, device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.llama_embedder = Llama3Embedder8B(device=device, max_length=max_length, dtype=dtype)
        self.t5_embedder = T5Embedder(device=device, max_length=max_length)

    def forward(self, text):
        return [self.llama_embedder(text), self.t5_embedder(text)]

    def encode(self, text):
        return self(text)


class Llama3_and_T5_XXL(AbstractEncoder):
    """LLaMA 3 + T5-XXL dual text embedder."""

    def __init__(self, device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.llama_embedder = Llama3Embedder8B(device=device, max_length=max_length, dtype=dtype)
        self.t5_embedder = T5Embedder(version="google/flan-t5-xxl", device=device, max_length=max_length)

    def forward(self, text):
        return [self.llama_embedder(text), self.t5_embedder(text)]

    def encode(self, text):
        return self(text)


class Llama3_and_ByT5(AbstractEncoder):
    """LLaMA 3 + ByT5 dual text embedder."""

    def __init__(self, device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.llama_embedder = Llama3Embedder8B(device=device, max_length=max_length, dtype=dtype)
        self.t5_embedder = ByT5Embedder(device=device, max_length=max_length * 6)

    def forward(self, text):
        return [self.llama_embedder(text), self.t5_embedder(text)]

    def encode(self, text):
        return self(text)


class Llama3_and_CLIP(AbstractEncoder):
    """LLaMA 3 + CLIP (concatenated) text embedder."""

    def __init__(self, device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.llama_embedder = Llama3Embedder8B(device=device, max_length=max_length, dtype=dtype)
        self.clip_embedder = FrozenCLIPEmbedder(device=device, max_length=max_length)

    def forward(self, text):
        return torch.cat([self.llama_embedder(text), self.clip_embedder(text)], dim=-1)

    def encode(self, text):
        return self(text)


class Llama3_and_CLIP_SEQ(AbstractEncoder):
    """LLaMA 3 + CLIP (as list) text embedder."""

    def __init__(self, device="cuda", max_length=77, dtype="bf16"):
        super().__init__()
        self.llama_embedder = Llama3Embedder8B(device=device, max_length=max_length, dtype=dtype)
        self.clip_embedder = FrozenCLIPEmbedder(device=device, max_length=max_length)

    def forward(self, text):
        return [self.llama_embedder(text), self.clip_embedder(text)]

    def encode(self, text):
        return self(text)


class T5XXLEmbedder(nn.Module):
    """T5-XXL encoder-only (as used by LTX-Video). Returns [B, S, 4096]."""

    def __init__(self, version="google/t5-v1_1-xxl", device="cuda", max_length=256, dtype="bf16"):
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(version)
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.encoder = T5EncoderModel.from_pretrained(version, torch_dtype=torch_dtype).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts, image=None):
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        return self.encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask).last_hidden_state

    def encode(self, text, image=None):
        return self(text, image=image)


class DummyEmbedder(AbstractEncoder):
    """Zero-cost dummy text+image embedder for toy experiments.

    Returns fixed-size zero tensors so the rest of the pipeline (conditioner,
    denoiser, model) runs without downloading any real encoder weights.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        hidden_dim_2: int = 1024,
        max_length: int = 77,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.max_length = max_length
        self.device = device

    def forward(self, text, image=None):
        B = len(text) if isinstance(text, (list, tuple)) else 1
        z_txt = torch.zeros(B, self.max_length, self.hidden_dim, device=self.device)
        z_im = torch.zeros(B, 324, self.hidden_dim_2, device=self.device)
        return [z_txt, z_im]

    def encode(self, text, image=None):
        return self(text, image)


# ---------------------------------------------------------------------------
# Conditioners
# ---------------------------------------------------------------------------

# Text embedder registry: MODEL.TEXT_ENCODER.VERSION -> (class, kwargs_fn)
_IN_EMBEDDERS = {
    "clip-vit-large-p14": lambda cfg, dev: FrozenCLIPEmbedder(device=dev),
    "llama3-8b": lambda cfg, dev: Llama3Embedder8B(device=dev, dtype=cfg.SOLVER.AMP_TYPE),
    "llama3p2": lambda cfg, dev: Llama3P2Embedder(device=dev, dtype=cfg.SOLVER.AMP_TYPE),
    "llama3-8b-and-clip": lambda cfg, dev: Llama3_and_CLIP(
        device=dev, max_length=77, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "llama3-8b-and-clip-seq": lambda cfg, dev: Llama3_and_CLIP_SEQ(
        device=dev, max_length=77, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "llama3-8b-and-flan-t5": lambda cfg, dev: Llama3_and_T5(
        device=dev, max_length=77, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "llama3-8b-and-by-t5": lambda cfg, dev: Llama3_and_ByT5(
        device=dev, max_length=77, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "llama3-8b-and-t5-xxl": lambda cfg, dev: Llama3_and_T5_XXL(
        device=dev, max_length=77, dtype=cfg.SOLVER.AMP_TYPE
    ),
}

_I2V_EMBEDDERS = {
    "llama3p2_and_dinov2": lambda cfg, dev: Llama3P2_and_DINOV2_Embedder(
        device=dev,
        max_length=cfg.MODEL.TEXT_ENCODER.TOKEN_LIMIT,
        dtype=cfg.SOLVER.AMP_TYPE,
        img_res=cfg.MODEL.VIDEO.IMG_ENC_RES,
    ),
    "llama3p2": lambda cfg, dev: Llama3P2Embedder(
        device=dev, max_length=cfg.MODEL.TEXT_ENCODER.TOKEN_LIMIT, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "t5_xxl": lambda cfg, dev: T5XXLEmbedder(
        device=dev, max_length=cfg.MODEL.TEXT_ENCODER.TOKEN_LIMIT, dtype=cfg.SOLVER.AMP_TYPE
    ),
    "dummy": lambda cfg, dev: DummyEmbedder(
        hidden_dim=cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM,
        hidden_dim_2=cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM_2
        if cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM_2 and cfg.MODEL.TEXT_ENCODER.HIDDEN_DIM_2 > 0
        else 1024,
        max_length=cfg.MODEL.TEXT_ENCODER.TOKEN_LIMIT,
        device=dev,
    ),
}


def _resolve_condition_type(model_condition: str) -> str:
    """Map MODEL.CONDITION string to an internal conditioning type."""
    low = model_condition.lower()
    if low in (
        "text",
        "sdxl_text_clip",
        "sdxl_text_large",
        "ltx2b",
        "ltx2b-distilled",
        "ltx13b-distilled",
    ) or low.startswith("t2i-"):
        return "text"
    elif low in ("sdxl", "sdxls"):
        return "sdxl"
    elif low in ("sd_unet", "sdxl_inside", "sdxl_inside2", "mdt"):
        return "sdxl_inside"
    else:
        return "sdxl_inside"


def _count_control_vars(cfg) -> int:
    """Count the number of conditioning control variables based on config flags."""
    n = 4
    if cfg.DATA.AESTHETIC_COND:
        n += 1
    if cfg.DATA.FLIP_COND:
        n += 1
    if cfg.DATA.BLUR_COND:
        n += 1
    return n


def _sample_crop_params(
    batch_size: int,
    num_vars: int,
    aes_cond: bool,
    flip_cond: bool,
    blur_cond: bool,
    explicit_aspect_ratio: bool,
) -> np.ndarray:
    """Generate random crop/size/aspect-ratio parameters for sampling.

    This was previously duplicated 7 times across conditioner classes.
    """
    s = np.random.randint(512, 1024, (batch_size, 1))
    aspect_ratio = np.random.rand(batch_size, 1) * 0.6 + 0.7
    cr = np.zeros((batch_size, num_vars))

    # Set control variable indices based on which flags are active.
    idx = 4  # first 4 slots are always crop coords
    if aes_cond:
        cr[:, idx] = 6.2 * 100
        idx += 1
    if flip_cond:
        cr[:, idx] = 0.0
        idx += 1
    if blur_cond:
        cr[:, idx] = 0.0

    if explicit_aspect_ratio:
        return np.concatenate((s, 300 * aspect_ratio, cr), axis=1)
    else:
        return np.concatenate((s, aspect_ratio * s, cr), axis=1)


class BaseConditioner(nn.Module):
    """Base class with shared init, CFG dropout, and sampling logic.

    Subclasses override forward() and optionally sample() to handle
    different input signatures (text-only vs text+image).
    """

    def __init__(self, cfg, device, embedder_registry):
        super().__init__()
        self.device = device
        self._type = _resolve_condition_type(cfg.MODEL.CONDITION)

        # Instantiate embedder
        if self._type == "text":
            version = cfg.MODEL.TEXT_ENCODER.VERSION
            if version not in embedder_registry:
                raise ValueError(
                    f"Unknown text encoder '{version}'. Available: {list(embedder_registry.keys())}"
                )
            self.embedder = embedder_registry[version](cfg, device).to(device, non_blocking=True)
            self.embedder.eval()
            self.embedder.requires_grad_(False)
        elif self._type == "sdxl":
            self.embedder = ConcatTimestepEmbedderND(outdim=cfg.MODEL.DENOISER.CONCAT_DIM).to(
                device, non_blocking=True
            )
        elif self._type == "sdxl_inside":
            self.embedder = nn.Identity()

        self.num_classes = cfg.DATA.NUM_CLASSES
        self.use_cfg = cfg.SOLVER.USE_CFG
        self.cfg_p = cfg.SOLVER.CFG_DROP_P

        self.num_control_vars = _count_control_vars(cfg)
        self.aes_cond = cfg.DATA.AESTHETIC_COND
        self.flip_cond = cfg.DATA.FLIP_COND
        self.blur_cond = cfg.DATA.BLUR_COND
        self.explicit_aspect_ratio = cfg.DATA.POWER_COSINE

    # -- shared helpers --

    def convert_to_text(self, ids):
        return [id2txt[k] for k in ids.detach().cpu().numpy()]

    def drop_labels(self, ids, cond):
        mask = torch.rand(ids.shape, device=ids.device) > self.cfg_p
        return torch.where(mask, ids, 0), torch.where(mask.unsqueeze(1), cond, 0), mask

    def drop_text(self, txt_emb, cond):
        """Apply CFG dropout by zeroing out text embeddings with probability cfg_p."""
        if isinstance(txt_emb, torch.Tensor):
            mask = torch.rand(txt_emb.shape[0], device=txt_emb.device) > self.cfg_p
            masked_text = torch.where(mask[:, None, None], txt_emb, 0)
            mask_cond = torch.rand(txt_emb.shape[0], device=txt_emb.device) > self.cfg_p
        elif isinstance(txt_emb, list):
            masked_text = []
            for inp in txt_emb:
                mask = torch.rand(inp.shape[0], device=inp.device) > self.cfg_p ** (1 / len(txt_emb))
                masked_text.append(torch.where(mask[:, None, None], inp, 0))
            mask_cond = torch.rand(txt_emb[0].shape[0], device=txt_emb[0].device) > self.cfg_p
        return masked_text, torch.where(mask_cond.unsqueeze(1), cond, 0), mask

    def get_cfg_version(self, cond):
        """Return an unconditional version of cond (all zeros) for CFG."""
        cond2 = deepcopy(cond)
        for k, v in cond2.items():
            if isinstance(v, torch.Tensor):
                cond2[k] = torch.zeros_like(v)
            elif isinstance(v, list):
                cond2[k] = [torch.zeros_like(vl) for vl in v]
        return cond2

    def _make_crop_params(self, batch_size):
        """Generate random crop parameters for sampling."""
        return _sample_crop_params(
            batch_size,
            self.num_control_vars,
            self.aes_cond,
            self.flip_cond,
            self.blur_cond,
            self.explicit_aspect_ratio,
        )

    def _forward_non_text(self, ids, cond, drop):
        """Shared forward logic for non-text conditioning types."""
        ids_ = ids[:, 0].clone()
        if self.use_cfg:
            ids_ += 1
            mask = torch.ones(ids_.shape[0], device=ids_.device).bool()
            if drop:
                ids_, cond, mask = self.drop_labels(ids_, cond)
            else:
                mask = torch.ones(ids_.shape[0], device=ids_.device).bool()
        else:
            mask = torch.ones(ids_.shape[0], device=ids_.device).bool()

        if self._type == "sdxl":
            z = self.embedder(cond)
            return {"encoder_hidden_states": z, "class_labels": ids_, "cfg_mask": mask}
        elif self._type == "sdxl_inside":
            return {"encoder_hidden_states": cond.clone(), "class_labels": ids_, "cfg_mask": mask}
        else:
            return {"class_labels": ids_, "cfg_mask": mask}


class INConditioner(BaseConditioner):
    """Conditioner for class/text-conditioned generation (no image input)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device, embedder_registry=_IN_EMBEDDERS)

    @torch.no_grad()
    def forward(self, ids, cond=None, drop=True):
        if self._type == "text":
            txt_emb = self.embedder.encode(ids)
            device = txt_emb.device if isinstance(txt_emb, torch.Tensor) else txt_emb[0].device
            mask = torch.ones(
                txt_emb.shape[0] if isinstance(txt_emb, torch.Tensor) else len(ids), device=device
            ).bool()
            if self.use_cfg and drop:
                txt_emb, cond, mask = self.drop_text(txt_emb, cond)
            return {"class_labels": txt_emb, "encoder_hidden_states": cond.clone(), "cfg_mask": mask}
        else:
            return self._forward_non_text(ids, cond, drop)

    @torch.no_grad()
    def sample(self, batch_size, idx=None, crop_params=None):
        if self._type != "text":
            idx = (
                torch.randint(0, self.num_classes, (batch_size, 1), device=self.device).long()
                if idx is None
                else torch.tensor(idx).to(self.device).long()
            )
        else:
            if idx is None:
                from data.datasets.example_prompts import prompts

                idx = random.sample(prompts, batch_size)

        if self._type in ("sdxl", "sdxl_inside"):
            crop = self._make_crop_params(batch_size) if crop_params is None else crop_params
            cond = torch.zeros_like(torch.tensor(crop, device=self.device).float())
        elif self._type == "text":
            crop = self._make_crop_params(batch_size) if crop_params is None else crop_params
            cond = torch.tensor(crop, device=self.device).float()
        else:
            cond = None

        return self.forward(idx, cond=cond, drop=False)


class I2VConditioner(BaseConditioner):
    """Conditioner for image-to-video generation (text + image input)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device, embedder_registry=_I2V_EMBEDDERS)

    @torch.no_grad()
    def forward(self, ids, image, cond=None, drop=True):
        if self._type == "text":
            txt_emb = self.embedder.encode(ids, image)
            device = txt_emb.device if isinstance(txt_emb, torch.Tensor) else txt_emb[0].device
            mask = torch.ones(len(ids), device=device).bool()
            if self.use_cfg and drop:
                txt_emb, cond, mask = self.drop_text(txt_emb, cond)
            return {"class_labels": txt_emb, "encoder_hidden_states": cond.clone(), "cfg_mask": mask}
        else:
            return self._forward_non_text(ids, cond, drop)

    @torch.no_grad()
    def sample(self, batch_size, idx=None, image=None, crop_params=None):
        if self._type != "text":
            idx = (
                torch.randint(0, self.num_classes, (batch_size, 1), device=self.device).long()
                if idx is None
                else torch.tensor(idx).to(self.device).long()
            )
        else:
            if idx is None:
                from data.datasets.example_prompts import prompts

                idx = random.sample(prompts, batch_size)

        if self._type in ("sdxl", "sdxl_inside"):
            crop = self._make_crop_params(batch_size) if crop_params is None else crop_params
            cond = torch.zeros_like(torch.tensor(crop, device=self.device).float())
        elif self._type == "text":
            crop = self._make_crop_params(batch_size) if crop_params is None else crop_params
            cond = torch.tensor(crop, device=self.device).float()
        else:
            cond = None

        return self.forward(idx, cond=cond, image=image, drop=False)


class I2V2Conditioner(I2VConditioner):
    """Like I2VConditioner but uses cached unconditional embeddings for CFG.

    Instead of zeroing out text embeddings during CFG dropout, this encodes
    an empty string once and reuses it. This improves CFG quality for
    text-conditioned models.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._cached_uncond_emb = None

    @torch.no_grad()
    def _get_uncond_embedding(self, batch_size, device, dtype):
        """Get unconditional embedding (encoded empty string, cached)."""
        if self._cached_uncond_emb is None or self._cached_uncond_emb.device != device:
            self._cached_uncond_emb = self.embedder.encode([""], image=None).detach()
        return self._cached_uncond_emb.expand(batch_size, -1, -1).contiguous().to(device=device, dtype=dtype)

    def drop_text(self, txt_emb, cond):
        """CFG dropout using cached unconditional embeddings instead of zeros."""
        if isinstance(txt_emb, torch.Tensor):
            mask = torch.rand(txt_emb.shape[0], device=txt_emb.device) > self.cfg_p
            uncond_emb = self._get_uncond_embedding(txt_emb.shape[0], txt_emb.device, txt_emb.dtype)
            masked_text = torch.where(mask[:, None, None], txt_emb, uncond_emb)
            mask_cond = torch.rand(txt_emb.shape[0], device=txt_emb.device) > self.cfg_p
        elif isinstance(txt_emb, list):
            masked_text = []
            uncond_emb = self._get_uncond_embedding(txt_emb[0].shape[0], txt_emb[0].device, txt_emb[0].dtype)
            for inp in txt_emb:
                mask = torch.rand(inp.shape[0], device=inp.device) > self.cfg_p ** (1 / len(txt_emb))
                masked_text.append(torch.where(mask[:, None, None], inp, uncond_emb))
            mask_cond = torch.rand(txt_emb[0].shape[0], device=txt_emb[0].device) > self.cfg_p
        return masked_text, torch.where(mask_cond.unsqueeze(1), cond, 0), mask

    def get_cfg_version(self, cond):
        """Unconditional version using cached empty-string embeddings."""
        cond2 = deepcopy(cond)
        if self._type == "text" and "class_labels" in cond2:
            txt_emb = cond2["class_labels"]
            if isinstance(txt_emb, torch.Tensor):
                cond2["class_labels"] = self._get_uncond_embedding(
                    txt_emb.shape[0], txt_emb.device, txt_emb.dtype
                )
            elif isinstance(txt_emb, list):
                uncond = self._get_uncond_embedding(txt_emb[0].shape[0], txt_emb[0].device, txt_emb[0].dtype)
                cond2["class_labels"] = [uncond.clone() for _ in txt_emb]
        else:
            for k, v in cond2.items():
                if isinstance(v, torch.Tensor):
                    cond2[k] = torch.zeros_like(v)
                elif isinstance(v, list):
                    cond2[k] = [torch.zeros_like(vl) for vl in v]
        return cond2

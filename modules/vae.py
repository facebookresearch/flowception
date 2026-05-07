import torch
import torch.nn as nn

try:
    from modules.video.video_vae.cosmos_ae import CosmosVAE
except (ImportError, ModuleNotFoundError):
    CosmosVAE = None

try:
    from modules.video.video_vae.kl_ltx.vae_ltx import AutoencoderKLLTXVideo
except (ImportError, ModuleNotFoundError):
    AutoencoderKLLTXVideo = None


SUPPORTED_VAES = {
    "IDENTITY",
    "LTX_AE",
    "LTX_AE_0_9_5",
    "LTX_AE_0_9_8",
    "COSMOS_1_X8",
}


class _IdentityLatentDist:
    """Small shim so toy runs can use the same encode API as real VAEs."""

    def __init__(self, z: torch.Tensor):
        self._z = z

    def sample(self) -> torch.Tensor:
        return self._z


class _IdentityEncodeResult:
    def __init__(self, z: torch.Tensor):
        self.latent_dist = _IdentityLatentDist(z)


class IdentityVAE(nn.Module):
    """Identity autoencoder used by the synthetic toy-coloring configs."""

    def __init__(self):
        super().__init__()
        self.normalize_latents = False
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x, **kwargs):
        return _IdentityEncodeResult(x)

    def decode(self, z, **kwargs):
        class _Out:
            sample = z

        return _Out()

    def forward(self, x, **kwargs):
        return x


def _require_ltx():
    if AutoencoderKLLTXVideo is None:
        raise ImportError("AutoencoderKLLTXVideo could not be imported; LTX VAE support is unavailable.")
    return AutoencoderKLLTXVideo


def _require_cosmos():
    if CosmosVAE is None:
        raise ImportError("CosmosVAE could not be imported; COSMOS_1_X8 VAE support is unavailable.")
    return CosmosVAE


def get_vae(cfg, device):
    """Instantiate the configured public VAE.

    Supported production VAEs are LTX_AE, LTX_AE_0_9_5, LTX_AE_0_9_8, and
    COSMOS_1_X8. IDENTITY is kept only for the synthetic toy-coloring configs.
    """
    name = cfg.MODEL.VAE.NAME

    if name == "IDENTITY":
        vae = IdentityVAE().to(device)
    elif name == "LTX_AE":
        ltx_vae = _require_ltx()
        vae = ltx_vae.from_pretrained("Lightricks/LTX-Video", subfolder="vae").to(device).eval()
    elif name == "LTX_AE_0_9_5":
        ltx_vae = _require_ltx()
        vae = ltx_vae.from_pretrained(
            "Lightricks/LTX-Video-0.9.5",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
    elif name == "LTX_AE_0_9_8":
        ltx_vae = _require_ltx()
        vae = ltx_vae.from_pretrained(
            "Lightricks/LTX-Video-0.9.8-13B-distilled",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
    elif name == "COSMOS_1_X8":
        cosmos_vae = _require_cosmos()
        if cfg.MODEL.VAE.CHECKPOINT == "":
            raise ValueError("COSMOS_1_X8 requires MODEL.VAE.CHECKPOINT to point to the Cosmos tokenizer directory.")
        vae = cosmos_vae("Cosmos-1.0-Tokenizer-CV8x8x8", cfg.MODEL.VAE.CHECKPOINT, device=device)
    else:
        supported = ", ".join(sorted(SUPPORTED_VAES))
        raise ValueError(f"Unsupported VAE '{name}'. Supported VAEs: {supported}")

    vae.normalize_latents = cfg.MODEL.VAE.NORMALIZE_LATENTS
    return vae

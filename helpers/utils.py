import os
import struct
import subprocess
import warnings
from argparse import Namespace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import numpy as np
import safetensors
import torch
import torch.distributed as dist
from PIL import Image, ImageFilter

try:
    from safe_blur.pipeline.blur import CompositePIL, DP_CropPixGaussBlurPIL, GaussianBlurPIL

    safe_blur_available = True
except ImportError:
    warnings.warn(
        "safe_blur is not available. "
        "Please install the missing dependency if you want to use pixelated blur. "
        "Falling back to Gaussian blur.",
        stacklevel=2,
    )
    safe_blur_available = False

StrOrBytesPath: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
FileDescriptorOrPath: TypeAlias = int | StrOrBytesPath


def parse_ckpt(filename, device="cpu"):
    if filename.endswith(".safetensors"):
        tensors = {}
        with safetensors.safe_open(filename, framework="pt", device=device) as f:
            for key in f.keys():  # noqa
                tensors[key] = f.get_tensor(key)
        return tensors
    elif filename.endswith(".bin"):
        tensors = torch.load(filename, map_location=device)
        return tensors
    else:
        tensors = torch.load(filename, map_location=device)
        return tensors["ema"]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        image = Image.open(f)
        return image.convert("RGB")


def pil_loader_v2(fp: FileDescriptorOrPath, max_size: tuple | None = None, mode: str = "RGB") -> Image.Image:
    if max_size is None:
        if isinstance(fp, str | Path):
            with open(fp, "rb") as f:
                return Image.open(f).convert(mode)
        elif isinstance(fp, bytes):
            return Image.open(BytesIO(fp)).convert(mode)
        else:
            raise ValueError("Invalid input type for fp.")

    if isinstance(fp, str | Path):
        with open(fp, "rb") as f:
            image = Image.open(BytesIO(f.read()))
    elif isinstance(fp, bytes):
        image = Image.open(fp)
    else:
        raise ValueError("Invalid input type for fp.")
    image.draft("RGB", (max_size[0], max_size[1]))
    return image


def masked_blur(
    image: Image.Image, blur_mask: Image.Image, image_blur_radius: float, mask_blur_radius: float
) -> Image.Image:
    """
    Blurs parts of the image according to the mask provided.
    In fair_blur the way face masks are computed is as follows:
        max_diff = 0
        for face in face_data.values():
            x1, y1, x2, y2 = face['facial_area']
            x_diff, y_diff = x2-x1, y2-y1
            max_diff = max(max(x_diff, y_diff), max_diff)
        image_blur_radius = max_diff/17
        mask_blur_radius = image_blur_radius/3
    """
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    if blur_mask.mode != "1":
        blur_mask = blur_mask.convert(mode="1")
    assert image.mode == "RGB"
    assert blur_mask.mode == "1"
    mask_filter = ImageFilter.GaussianBlur(radius=mask_blur_radius)
    soft_mask = blur_mask.convert(mode="L").filter(mask_filter)
    assert soft_mask.mode == "L"
    image_filter = ImageFilter.GaussianBlur(radius=image_blur_radius)
    blurred_image = image.filter(image_filter)
    if soft_mask.size != image.size:
        soft_mask = soft_mask.resize(image.size)
    return Image.composite(blurred_image, image, soft_mask)


def masked_safeblur(
    image: Image.Image, blur_mask: Image.Image, image_blur_radius: float, mask_blur_radius: float
) -> Image.Image:
    """
    Blurs parts of the image using pixelated (safe) blur when available.
    """
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    if blur_mask.mode != "1":
        blur_mask = blur_mask.convert(mode="1")
    assert image.mode == "RGB"
    assert blur_mask.mode == "1"

    pixel_size = 4

    max_diagonal = max(np.array(blur_mask).sum(0).max(), np.array(blur_mask).sum(1).max())
    kernel_multiplier = 0.5
    num_pixels = max_diagonal // pixel_size
    kernel_size = kernel_multiplier * num_pixels + 1

    composite = CompositePIL(
        DP_CropPixGaussBlurPIL(
            radius=kernel_size,
            pixel_size=[pixel_size, pixel_size],
            pixel_offset=[0, 0],
            noise_type="Gaussian",
            noise_std=0.04,
            seed=None,  # if the same image is seen twice, the faces will be slightly different
        ),
        GaussianBlurPIL(mask_blur_radius),
        smooth=True,
        smooth_radius=0.5,
        is_img_blur_with_mask=True,
    )
    if blur_mask.size != image.size:
        blur_mask = blur_mask.resize(image.size)  # resample=0)
    return composite.overlay(image, blur_mask)  # np.array(blur_mask))


def pil_loader_v2_blur(
    fp: FileDescriptorOrPath,
    max_size: tuple | None = None,
    mode: str = "RGB",
    blur_mask_fp: FileDescriptorOrPath | None = None,
    blur_radius_fp: FileDescriptorOrPath | None = None,
    return_mask: bool = False,
) -> Image.Image | tuple[Image.Image, Image.Image]:
    """
    Loads an image from a file path or a file-like object using the PIL library.
    If a blur mask is provided, it applies a Gaussian blur to the image based on the mask.
    """
    im = pil_loader_v2(fp, max_size=max_size, mode=mode)
    mask = Image.new("L", im.size)
    if blur_mask_fp is not None:
        mask = pil_loader_v2(blur_mask_fp, max_size=None, mode="1")
        if mask.getextrema() != (0, 0):
            if blur_radius_fp is None and isinstance(blur_mask_fp, str | Path):
                blur_radius_fp = str(blur_mask_fp).replace("-mask.png", "-blur-radius.bin")
            if isinstance(blur_radius_fp, str):
                with open(blur_radius_fp, "rb") as f:
                    blur_radius_fp = f.read()
            assert isinstance(blur_radius_fp, bytes)
            image_radius, mask_radius = struct.unpack("ff", blur_radius_fp)

            if safe_blur_available:
                im = masked_safeblur(im, mask, image_radius, mask_radius)
            else:
                im = masked_blur(im, mask, image_radius, mask_radius)
    if return_mask:
        return im, mask
    else:
        return im


def nvidia_smi_gpu_memory_stats(logger):
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    try:
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        out_dict = {}
        for item in out_list:
            if " MiB" in item:
                gpu_idx, mem_used = item.split(",")
                gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
                out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024
    except FileNotFoundError:
        logger.error("Failed to find the 'nvidia-smi' executable for printing GPU stats")
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def get_nvidia_smi_gpu_memory_stats_str():
    return f"nvidia-smi stats: {nvidia_smi_gpu_memory_stats()}"


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def convert_to_namespace(dict_):
    # Converts nested dictionary to namespace.
    for k in dict_:
        if isinstance(dict_[k], dict):
            dict_[k] = convert_to_namespace(dict_[k])
    dict_ = Namespace(**dict_)
    return dict_


def generate_run_id(cluster: str) -> str:
    username = os.getenv("USERNAME") or os.getenv("USER")
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"{now}_{cluster}_{username}"

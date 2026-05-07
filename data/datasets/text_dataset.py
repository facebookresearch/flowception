import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from helpers.utils import pil_loader_v2_blur
from engine.data_classes import Datapoint


def get_indexor(choice, ratio):
    if choice == "first":

        def indexor():
            return 0
    elif choice == "last":

        def indexor():
            return -1
    elif choice == "mix":

        def indexor():
            return np.random.choice([0, -1], p=[1 - ratio, ratio])

    return indexor


class NPYTxtDataset(Dataset):
    def __init__(
        self,
        annotations_dir,
        img_dir,
        crop=True,
        flip=True,
        img_size=(256, 256),
        crop_scale=0.9,
        npy_format="train",
        img_key=None,
        caption_idx="first",
        recap_ratio=0.75,
        blur_dir=None,
        flip_cond=False,
        blur_cond=False,
        entropy_ths=-1.0,
        explicit_aspect_ratio=False,
    ):
        super().__init__()
        print("Loading labels")
        self.labels = np.load(annotations_dir, allow_pickle=True)
        print("Loaded labels")
        # filter the labels
        if entropy_ths > 0.0:
            self.labels = [sample for sample in self.labels if sample["entropy"] > entropy_ths]
        print("filtered labels")
        if img_key:
            self.img_key = img_key
        else:
            self.img_key = "image_name" if npy_format == "train" else "image_id"
        self.img_dir = img_dir
        self.caption_idx = caption_idx
        self.indexor = get_indexor(caption_idx, recap_ratio)
        self.blur_dir = blur_dir
        print("loading blur")
        # if blur_dir:
        # self.blurred_set = set(np.load(Path(blur_dir) / "blurred_namelist.npy", allow_pickle=True))
        if blur_dir:
            # self.blurred_set = set(np.load(Path(blur_dir) / "blurred_namelist.npy", allow_pickle=True))
            all_blurred_paths = os.listdir(blur_dir)
            self.blurred_set = set(
                [p[: -len("-mask.png")] for p in all_blurred_paths if p.endswith("-mask.png")]
            ).intersection(
                set(
                    [
                        p[: -len("-blur-radius.bin")]
                        for p in all_blurred_paths
                        if p.endswith("-blur-radius.bin")
                    ]
                )
            )
        print("loaded blur")

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.crop_scale = crop_scale
        self.flip = flip
        self.crop = crop
        self.flip_cond = flip_cond
        self.blur_cond = blur_cond
        self.explicit_aspect_ratio = explicit_aspect_ratio

        self.new_size = (int(self.img_size[0] / self.crop_scale), int(self.img_size[1] / self.crop_scale))

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])
        print("done init")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, captions_idx=None):
        item = self.labels[idx]

        img_path = Path(self.img_dir) / str(item[self.img_key])
        blur_path = None
        if self.blur_dir and (img_path.name in self.blurred_set):
            blur_path = Path(self.blur_dir) / (img_path.name + "-mask.png")
        img, mask = pil_loader_v2_blur(
            img_path, mode="RGB", max_size=self.new_size, blur_mask_fp=blur_path, return_mask=True
        )

        w0, h0 = img.size
        img = img.resize(self.new_size).convert("RGB")
        mask = mask.resize(self.new_size, resample=0)

        flipped = self.flip and random.random() > 0.5
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.crop:
            w, h = self.new_size
            dw = random.randint(self.img_size[0], w)
            dh = random.randint(self.img_size[1], h)
            x0 = random.randint(0, w - dw)
            y0 = random.randint(0, h - dh)

            scale = int(w0 * self.crop_scale / self.img_size[0])

            img_cropped = img.crop((x0, y0, x0 + dw, y0 + dh))
            img_cropped = img_cropped.resize(self.img_size)

            mask_cropped = mask.crop((x0, y0, x0 + dw, y0 + dh))
            mask_cropped = mask_cropped.resize(self.img_size)
        else:
            w, h = w0, h0
            dw, dh, x0, y0 = w, h, 0, 0
            scale = int(w0 / self.img_size[0])
            img_cropped = img.resize(self.img_size)
            mask_cropped = mask.resize(self.img_size)

        img_tensor = self.to_tensor(img_cropped)
        mask_tensor = 1.0 - transforms.ToTensor()(mask_cropped)

        if not self.explicit_aspect_ratio:
            crop_coords = [w0, h0, scale * x0, scale * y0, scale * (w - dw - x0), scale * (h - dh - y0)]
        else:
            crop_coords = [
                w0,
                300.0 * w0 / h0,
                scale * x0,
                scale * y0,
                scale * (w - dw - x0),
                scale * (h - dh - y0),
            ]

        if self.flip_cond:
            crop_coords.append(float(flipped) * 500.0)
        if self.blur_cond:
            crop_coords.append(500.0 * float(((1.0 - mask_tensor).sum() > 1).item()))

        if captions_idx is None:
            captions_idx = np.random.randint(0, len(item["captions"]))

        return Datapoint(
            pixel_values=img_tensor,
            condition={
                "class_id": item["captions"][captions_idx],  # np.random.choice(item["captions"]),
                "caption_idx": torch.tensor(captions_idx),
                "crop_coords": torch.tensor(crop_coords),
                "mask": mask_tensor,
            },
        )

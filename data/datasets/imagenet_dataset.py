import os
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from helpers.utils import pil_loader_v2_blur
from engine.data_classes import Datapoint


class JSONDataset(Dataset):
    def __init__(self, annotations_dir, transforms):
        super().__init__()
        with open(annotations_dir) as f:
            self.labels = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels["items"])

    def __getitem__(self, idx):
        item = self.labels["items"][idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = self.transforms(img)
        return {
            "pixel_values": img_tensor,
            "class_id": torch.as_tensor([item["class_id"]]),
            "crop_coords": torch.tensor([item["width"], item["height"], 0, 0, item["width"], item["height"]]),
        }


# class JSONExtDataset(Dataset):
#     def __init__(
#         self,
#         annotations_dir,
#         crop=True,
#         flip=True,
#         img_size=(256, 256),
#         crop_scale=0.9,
#         blur_dir=None,
#         img_root = "",
#         flip_cond=False,
#         blur_cond=False,
#         explicit_aspect_ratio=False,
#     ):
#         super().__init__()
#         with open(annotations_dir) as file:
#             self.labels = json.load(file)
#         self.blur_dir = blur_dir
#         print("blur dir ", self.blur_dir)
#         if self.blur_dir:
#             all_blurred_paths = os.listdir(blur_dir)
#             print("getting blurred set")
#             self.blurred_set = set(
#                 [p[: -len("-mask.png")] for p in all_blurred_paths if p.endswith("-mask.png")]
#             ).intersection(
#                 set(
#                     [
#                         p[: -len("-blur-radius.bin")]
#                         for p in all_blurred_paths
#                         if p.endswith("-blur-radius.bin")
#                     ]
#                 )
#             )
#             print("Got blurred set")

#         self.img_root = img_root

#         if isinstance(img_size, tuple):
#             self.img_size = img_size
#         else:
#             self.img_size = (img_size, img_size)
#         assert len(self.img_size) == 2

#         self.crop_scale = crop_scale
#         self.flip = flip
#         self.crop = crop
#         self.new_size = (int(self.img_size[0] / self.crop_scale), int(self.img_size[1] / self.crop_scale))

#         self.flip_cond = flip_cond
#         self.blur_cond = blur_cond
#         self.explicit_aspect_ratio = explicit_aspect_ratio

#         self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

#     def __len__(self):
#         return len(self.labels["items"])

#     def __getitem__(self, idx):
#         item = self.labels["items"][idx]
#         img_path = Path(self.img_root) / Path(item["image_path"])
#         blur_path = None
#         # if self.blur_dir:
#         #     blur_path = Path(self.blur_dir) / (img_path.name + "-mask.png")
#         if self.blur_dir and (img_path.name in self.blurred_set):
#             blur_path = Path(self.blur_dir) / (img_path.name + "-mask.png")
#         # img = pil_loader_v2_blur(img_path, max_size=self.new_size, blur_path=blur_path)
#         img, mask = pil_loader_v2_blur(
#             img_path, mode="RGB", max_size=self.new_size, blur_mask_fp=blur_path, return_mask=True
#         )
#         # print("got image and mask")
#         print(item)
#         print(img.size, mask.size, Image.open(img_path).size)
#         img = img.convert("RGB")
#         flipped = self.flip and random.random() > 0.5
#         if flipped:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

#         w, h = img.size
#         w0,h0=w,h

#         if self.crop:
#             # w, h = self.new_size
#             # dw = random.randint(self.img_size[0], w)
#             # dh = random.randint(self.img_size[1], h)
#             # x0 = random.randint(0, w - dw)
#             # y0 = random.randint(0, h - dh)

#             # scale = int(w0 * self.crop_scale / self.img_size[0])

#             # img_cropped = img.crop((x0, y0, x0 + dw, y0 + dh))
#             # img_cropped = img_cropped.resize(self.img_size)

#             # mask_cropped = mask.crop((x0, y0, x0 + dw, y0 + dh))
#             # mask_cropped = mask_cropped.resize(self.img_size)


#             dw = random.randint(int(self.crop_scale * w), w)
#             dh = random.randint(int(self.crop_scale * h), h)
#             x0 = random.randint(0, w - dw)
#             y0 = random.randint(0, h - dh)

#             img_cropped = img.crop((x0, y0, x0 + dw, y0 + dh))
#             img_cropped = img_cropped.resize(self.img_size)

#             mask_cropped = mask.crop((x0, y0, x0 + dw, y0 + dh))
#             mask_cropped = mask_cropped.resize(self.img_size, resample=0)
#         else:
#             # w, h = w0, h0
#             # dw, dh, x0, y0 = w, h, 0, 0
#             # scale = int(w0 / self.img_size[0])
#             # img_cropped = img.resize(self.img_size).convert("RGB")
#             # mask_cropped = mask.resize(self.img_size)

#             dw, dh, x0, y0 = w, h, 0, 0
#             img_cropped = img.resize(self.img_size)
#             mask_cropped = mask.resize(self.img_size, resample=0)

#         img_tensor = self.to_tensor(img_cropped)
#         mask_tensor = 1.0 - transforms.ToTensor()(mask_cropped)

#         if not self.explicit_aspect_ratio:
#             crop_coords = [item["width"], item["height"], x0, y0, (w - dw - x0), (h - dh - y0)]
#         else:
#             crop_coords = [
#                 item["width"],
#                 300.0 * item["width"] / item["height"],
#                 x0,
#                 y0,
#                 (w - dw - x0),
#                 (h - dh - y0),
#             ]

#         if self.flip_cond:
#             crop_coords.append(float(flipped) * 500.0)
#         if self.blur_cond:
#             crop_coords.append(500.0 * float(((1.0 - mask_tensor).sum() > 1).item()))

#             # print(im_tensor.shape, torch.tensor([item["class_id"]]).shape, torch.tensor(crop_coords).shape, mask_tensor.shape)

#         # print(img_tensor.shape)
#         return Datapoint(
#             pixel_values=img_tensor,  # Pixel values are in the range [0, 1].
#             condition={
#                 "class_id": torch.tensor([item["class_id"]]),
#                 # "crop_coords": torch.tensor(
#                 #     [item["width"], item["height"], x0, y0, w - dw - x0, h - dh - y0]
#                 # ),
#                 "crop_coords": torch.tensor(crop_coords),
#                 # "mask": mask_tensor,
#             },
#         )


class JSONExtDataset(Dataset):
    def __init__(
        self,
        annotations_dir,
        crop=True,
        flip=True,
        img_size=(256, 256),
        crop_scale=0.9,
        blur_dir=None,
        img_root="",
        flip_cond=False,
        blur_cond=False,
        explicit_aspect_ratio=False,
    ):
        super().__init__()
        with open(annotations_dir) as file:
            self.labels = json.load(file)
        self.blur_dir = blur_dir
        print("blur dir ", self.blur_dir)
        if self.blur_dir:
            all_blurred_paths = os.listdir(blur_dir)
            print("getting blurred set")
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
            print("Got blurred set")

        self.img_root = img_root

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.crop_scale = crop_scale
        self.flip = flip
        self.crop = crop
        self.new_size = (int(self.img_size[0] / self.crop_scale), int(self.img_size[1] / self.crop_scale))

        self.flip_cond = flip_cond
        self.blur_cond = blur_cond
        self.explicit_aspect_ratio = explicit_aspect_ratio

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        return len(self.labels["items"])

    def __getitem__(self, idx, caption_idx=None):
        item = self.labels["items"][idx]
        img_path = Path(self.img_root) / Path(item["image_path"])
        blur_path = None
        # if self.blur_dir:
        #     blur_path = Path(self.blur_dir) / (img_path.name + "-mask.png")
        if self.blur_dir and (img_path.name in self.blurred_set):
            blur_path = Path(self.blur_dir) / (img_path.name + "-mask.png")
        # img = pil_loader_v2_blur(img_path, max_size=self.new_size, blur_path=blur_path)
        img, mask = pil_loader_v2_blur(
            img_path, mode="RGB", max_size=self.new_size, blur_mask_fp=blur_path, return_mask=True
        )
        img = img.convert("RGB")
        flipped = self.flip and random.random() > 0.5
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        w, h = img.size
        w0, h0 = w, h

        dw, dh, x0, y0 = w, h, 0, 0
        img_cropped = img.resize(self.img_size, resample=3)
        mask_cropped = mask.resize(self.img_size, resample=0)

        img_tensor = self.to_tensor(img_cropped)
        mask_tensor = 1.0 - transforms.ToTensor()(mask_cropped)

        if not self.explicit_aspect_ratio:
            crop_coords = [item["width"], item["height"], x0, y0, (w - dw - x0), (h - dh - y0)]
        else:
            crop_coords = [
                item["width"],
                300.0 * item["width"] / item["height"],
                x0,
                y0,
                (w - dw - x0),
                (h - dh - y0),
            ]

        if self.flip_cond:
            crop_coords.append(float(flipped) * 500.0)
        if self.blur_cond:
            crop_coords.append(500.0 * float(((1.0 - mask_tensor).sum() > 1).item()))

        return Datapoint(
            pixel_values=img_tensor,  # Pixel values are in the range [0, 1].
            condition={
                "class_id": torch.tensor([item["class_id"]]),
                "crop_coords": torch.zeros_like(torch.tensor(crop_coords)),
                "caption_idx": torch.tensor([0]),
            },
        )

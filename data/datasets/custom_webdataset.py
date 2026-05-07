import json
import pickle
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from engine.data_classes import Datapoint

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomWebDatasetProcessor:
    def __init__(
        self,
        crop=True,
        flip=True,
        img_size=(256, 256),
        crop_scale=0.9,
        recap_ratio=0.0,
        captions_dir=None,
        root_dir=None,
        aes_cond=False,
        flip_cond=False,
        blur_cond=False,
        entropy_dir=None,
        explicit_aspect_ratio=False,
    ):
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2
        # assert self.recap_ratio >= 0.0 and self.recap_ratio <= 1.0

        self.crop_scale = crop_scale
        self.flip = flip
        self.crop = crop
        self.recap_ratio = recap_ratio
        self.aes_cond = aes_cond
        self.flip_cond = flip_cond
        self.blur_cond = blur_cond
        self.new_size = (int(self.img_size[0] / self.crop_scale), int(self.img_size[1] / self.crop_scale))

        self.root_dir = root_dir
        self.captions_dir = captions_dir
        self.num_choices = len(self.captions_dir) + 1
        self.explicit_aspect_ratio = explicit_aspect_ratio

        self.entropy_dir = entropy_dir

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2.0) - 1.0)]
        )

    def transform(self, sample):
        imgb = sample["jpg"]
        jsonb = sample["json"]

        # we sample uniformly randomly between the available captions.
        choice = np.random.randint(0, self.num_choices)
        # choice = np.random.choice([0,1,2], p=[0.1, 0.45, 0.45])
        if choice == 0:
            label = str(sample["txt"].decode())
        else:
            cap_path = (
                sample["__url__"].replace(self.root_dir, self.captions_dir[choice - 1]).replace(".tar", ".pt")
            )

            with open(cap_path, "rb") as f:
                label = pickle.load(f)[sample["__key__"]].replace("The image shows a", "A")

        # read image entropy
        entropy_path = sample["__url__"].replace(self.root_dir, self.entropy_dir).replace(".tar", ".pt")
        # print("entr ", entropy_path, os.path.isfile(entropy_path))
        with open(entropy_path, "rb") as f:
            ets = pickle.load(f)
            # print("keys ", sample["__key__"], sample["__key__"] in ets.keys())
            entropy = ets[sample["__key__"]]

        img = Image.open(BytesIO(imgb))
        img.draft("RGB", self.new_size)
        img = img.resize(self.new_size).convert("RGB")

        json_ = json.loads(jsonb)
        w0, h0 = json_["width"], json_["height"]

        flipped = self.flip and random.random() > 0.5
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.crop:
            w, h = self.new_size
            dw = random.randint(self.img_size[0], w)
            dh = random.randint(self.img_size[1], h)
            x0 = random.randint(0, w - dw)
            y0 = random.randint(0, h - dh)

            img_cropped = img.crop((x0, y0, x0 + dw, y0 + dh))
            img_cropped = img_cropped.resize(self.img_size)
        else:
            w, h = w0, h0
            dw, dh, x0, y0 = w, h, 0, 0
            img_cropped = img.resize(self.img_size)

        img_tensor = self.to_tensor(img_cropped)
        scale = int(w0 * self.crop_scale / self.img_size[0])

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

        if self.aes_cond:
            aes = json_["AESTHETIC_SCORE"]
            crop_coords.append(aes * 100.0)
        if self.flip_cond:
            crop_coords.append(float(flipped) * 500.0)
        if self.blur_cond:
            crop_coords.append(0.0)

        # face masks are always empty
        mask = torch.ones(1, self.new_size[1], self.new_size[0]).float()

        return Datapoint(
            pixel_values=img_tensor,  # Pixel values are in the range [0, 1].
            condition={
                "class_id": label,
                "crop_coords": torch.tensor(crop_coords),
                "mask": mask,
                "entropy": entropy,
                "aesthetic_score": json_["AESTHETIC_SCORE"],
            },
        )

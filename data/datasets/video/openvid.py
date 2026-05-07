import torch
import os
import json
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
from engine.data_classes import Datapoint

from decord import VideoReader, cpu, gpu

class OpenVid1MDataset(Dataset):
    def __init__(
        self,
        annotations_dir,
        width,
        height,
        num_frames: int = 72,
        min_motion_score=-1,
    ):
        self.annots_dir = Path(annotations_dir)
        self.annots_csv = pd.read_csv(Path(annotations_dir) / "data" / "train" / "OpenVid-1M.csv")
        if min_motion_score > 0:
            self.annots_csv = self.annots_csv[self.annots_csv['motion score'] > min_motion_score]
            print(f"OpenVid1M: filtered for motion score superior to {min_motion_score}")
            

        # all_entries = []
        # for jp in tqdm.tqdm(os.listdir(annotations_dir)[:2]):
        #     if jp.endswith("jsonl"):
        #         with open(os.path.join(annotations_dir, jp), 'r') as f:
        #             data = [json.loads(line) for line in f]
        #             all_entries.extend(data)
        # self.all_entries = all_entries
        
        self.num_frames = num_frames
        self.width = width
        self.height = height


    def __len__(self):
        # return 10000000 #len(self.all_entries)
        return len(self.annots_csv)
    
    def __getitem__(self, idx):
        try:
            # j = np.random.randint(0, len(self.annots_csv))
            j = idx
            
            vid_path = str(self.annots_dir / "video" / self.annots_csv.iloc[j]['video'])
            item = self.annots_csv.iloc[j]
            caption = str(item['caption'])
            mscore = item["motion score"]
            
            if os.path.isfile(vid_path):
                reader = VideoReader(
                    vid_path,
                    num_threads=-1,
                    ctx=cpu(0),
                    width=self.width,
                    height=self.height
                )
                start = np.random.randint(0, len(reader) - self.num_frames)
                frames = reader.get_batch(np.arange(start, start + self.num_frames)).asnumpy()
                frames = frames/127.5 - 1.0
                img_tensor = torch.tensor(frames).float().permute(3,0,1,2)
            
            # if not self.explicit_aspect_ratio:
            #     crop_coords = [w0, h0, scale * x0, scale * y0, scale * (w - dw - x0), scale * (h - dh - y0)]
            # else:
            #     crop_coords = [
            #         w0,
            #         300.0 * w0 / h0,
            #         scale * x0,
            #         scale * y0,
            #         scale * (w - dw - x0),
            #         scale * (h - dh - y0),
            #     ]
            crop_coords = torch.zeros(8)

            # if self.flip_cond:
            #     crop_coords.append(float(flipped) * 500.0)
            # if self.blur_cond:
            #     crop_coords.append(500.0 * float(((1.0 - mask_tensor).sum() > 1).item()))
            captions_idx = 0
            return Datapoint(
                pixel_values=img_tensor,  # Pixel values are in the range [-1, 1].
                condition={
                    "class_id": caption,
                    "caption_idx": torch.tensor(captions_idx),
                    "crop_coords": crop_coords,
                    "motion_score": mscore
                    # "mask": None,
                },
            )

        except:
            new_idx = np.random.randint(self.__len__())
            return self.__getitem__(new_idx)

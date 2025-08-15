# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Union
import torch
from torchvision.datasets.vision import VisionDataset

MAX_RETRIES = 100


class VOSRawDataset:
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError()


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, Image.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""
    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


class VOSDataset(VisionDataset):
    def __init__(
        self,
        training: bool,
        video_dataset: VOSRawDataset,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        self.training = training
        self.video_dataset = video_dataset

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):
        # 1) load the raw image+mask
        img_np, mask_np = self.video_dataset[idx]
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).float()
        #    img_tensor: [C,H,W], mask_tensor: [1,H,W] float or bool

        # 2) convert to a PIL Image so your Frame class is happy
        C, h, w = img_tensor.shape

        # 3) wrap into a Frame with exactly one Object
        frame = Frame(
            data=img_tensor,
            objects=[
                Object(
                    object_id=0,
                    frame_index=0,
                    segment=mask_tensor.to(torch.uint8),
                )
            ],
        )

        # 4) build your VideoDatapoint
        datapoint = VideoDatapoint(
            frames=[frame],
            video_id=idx,
            size=(h, w),
        )

        return datapoint

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)
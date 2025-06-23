import os
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from tensordict import TensorDict
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ToTensor,
    AsDiscrete,
    SpatialPad,
    RandFlip,
    RandRotate90,
    Identity,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    AsDiscreted
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from utils.utils import calculate_porosity, calculate_surface_area


def convert_segmentation(seg):

    seg = torch.where(seg == 0, 2, seg)
    seg = torch.where(seg == 155, 1, seg)
    seg = torch.where(seg == 255, 0, seg)

    return seg


class SegmentationDataset:
    """Dataset for 3D tomogram segmentation data."""

    def __init__(
        self,
        segmentation_paths: List[str],
        num_samples: int,
        volume_shape: List[int],
        pad_size: int = 0,
        characteristics: Dict = None,
        transforms: Optional[Compose] = Identity
    ):
        self.volume_shape = OmegaConf.to_container((volume_shape))
        self.len = num_samples
        load_img = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            ToTensor(),
            SpatialPad(pad_size, mode="circular")
        ])
        self.samples = [convert_segmentation(load_img(x)) for x in segmentation_paths]
        self.transforms = transforms
        self.characteristics = characteristics

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        idx = i % len(self.samples)
        volume = self.transforms(self.samples[idx]).float()
        
        volume = volume / volume.sum(dim=0).unsqueeze(0)
        assert volume.shape == torch.Size(self.volume_shape)
        volume = volume.unsqueeze(0)

        # eps = 1e-8
        # noise = torch.clip(torch.rand_like(volume), eps, 1.0)
        # gumbel_noise = -torch.log(-torch.log(noise))
        # volume = F.one_hot(
        #     torch.argmax(torch.log(volume) + gumbel_noise, dim=0), 3
        # ).permute(3, 0, 1, 2).float()
        
        return volume, self.characteristics(volume)


class SimpleDataset:
    """Dataset for 3D tomogram segmentation data."""

    def __init__(
        self,
        segmentation_dir: List[str],
        volume_shape: List[int],
        num_samples: int
    ):
        self.volume_shape = OmegaConf.to_container((volume_shape))
        self.len = num_samples
        self.load_img = Compose([
            RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandRotate90(prob=0.5, spatial_axes=[1, 2]),
            RandRotate90(prob=0.5, spatial_axes=[0, 2]),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2)
        ])
        self.paths = glob(f"{segmentation_dir}/*")

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        path = np.random.choice(self.paths)
        volume = self.load_img(torch.load(path, weights_only=False)).float()
        
        volume = volume / volume.sum(dim=0).unsqueeze(0)
        
        characteristics = calculate_porosity(volume), calculate_surface_area(volume.argmax(dim=0) == 2)
        assert volume.shape == torch.Size(self.volume_shape)
        return volume, characteristics
    

class TomogramDataset:
    """Dataset for 3D tomogram segmentation data."""

    def __init__(
        self,
        segmentation_paths: List[str],
        num_samples: int,
        volume_shape: List[int],
        transforms: Optional[Compose] = None
    ):
        self.volume_shape = OmegaConf.to_container((volume_shape))
        self.len = num_samples
        load_img = Compose([
            LoadImaged(keys=["tomo", "segm"], image_only=True),
            EnsureChannelFirstd(keys=["tomo", "segm"], channel_dim="no_channel"),
            ToTensord(keys=["tomo", "segm"])
        ])
        self.to_onehot = AsDiscreted(keys="segm", to_onehot=3)
        self.samples = [convert_segmentation(load_img(x)) for x in segmentation_paths]
        self.transforms = transforms

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        idx = i % len(self.samples)
        data = self.transforms(self.samples[idx])
        tomo, segm = data["tomo"], data["segm"]
        
        tomo[segm[[0]].to(torch.bool)] -= 10305
        tomo[segm[[1]].to(torch.bool)] -= 8472
        tomo[segm[[2]].to(torch.bool)] -= 7168
        tomo = (tomo - 34.54711526953125) / 994.8743318327591
        # tomo = (tomo - 1697) / (58091 - 1679)
        # tomo = tomo * 2 - 1
        
        assert segm.shape == torch.Size(self.volume_shape)
        return tomo, segm

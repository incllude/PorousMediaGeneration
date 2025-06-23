import torch
import torch.nn.functional as F
import numpy as np
import os
import random
from scipy.spatial.transform import Rotation
from monai.visualize import plot_2d_or_3d_image
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from lightning import seed_everything
from monai.transforms import Transform, Affine
from monai.utils import set_determinism
import lightning as l
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import porespy as ps


def custom_collate_fn(batch):
    
    volume, characteristics = zip(*batch)
    return torch.cat(volume, dim=0), torch.cat(characteristics, dim=0)

def calculate_surface_area_batched(binary_segmentations):
    return torch.tensor([calculate_surface_area(x) for x in binary_segmentations]).float()

def calculate_porosity_batched(binary_segmentations):
    return torch.tensor([calculate_porosity(x) for x in binary_segmentations]).float()

def calculate_surface_area(segmentation):

    binary_segmentation = (segmentation.argmax(dim=1) == 2).numpy()
    
    try:
        mesh = ps.tools.mesh_region(region=binary_segmentation)
        surface_area = ps.metrics.mesh_surface_area(mesh=mesh)
    except:
        surface_area = 0.
    material_volume = np.sum(1 - binary_segmentation)
    
    specific_surface_area = surface_area / material_volume if material_volume > 0 else 0
    
    return specific_surface_area


def calculate_porosity(segmentation):
    binary_segmentation = segmentation[2].numpy()
    return binary_segmentation.mean().item()


class SaveRangeEpochsCallback(l.Callback):
    
    def __init__(self, range_epochs, prefix="", dirpath='my_checkpoints'):
        super().__init__()
        self.range = range_epochs
        self.dirpath = dirpath
        self.prefix = prefix
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer, l_module):
        epoch = trainer.current_epoch
        if epoch in self.range:
            ckpt_path = f"{self.dirpath}/{self.prefix + '_' if self.prefix != '' else ''}epoch={epoch}.pt"
            torch.save(l_module.model.state_dict(), ckpt_path)


def set_seed(seed):

    seed_everything(seed)
    set_determinism(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandAxisRotation(Transform):

    def __init__(self, n_rotations, angle_range):
        super(RandAxisRotation, self).__init__()
        
        directions = np.random.uniform(low=-1.0, high=1.0, size=(n_rotations, 3))
        directions /= np.linalg.norm(directions, axis=-1)[:, np.newaxis]
        angles = np.random.uniform(low=angle_range[0], high=angle_range[1], size=n_rotations)
        rotation_vectors = angles[:, np.newaxis] * directions
        rotation_matrices = Rotation.from_rotvec(rotation_vectors, degrees=True).as_matrix()
        rotation_matrices = np.pad(
            rotation_matrices,
            ((0, 0), (0, 1), (0, 1)),
            mode='constant',
            constant_values=0
        )
        
        self.rotations = [
            Affine(
                mode="nearest",
                affine=rotation_matrices[i],
            )
            for i in range(n_rotations)
        ]

    def __call__(self, x):

        rot = np.random.choice(self.rotations)
        return rot(x)[0]


def visualize_3d_slices(volume, slice_indices=None, num_slices=3, axis=2, return_slices=False):
    """
    Visualize slices from a 3D volume
    Args:
        volume: 3D tensor or numpy array
        slice_indices: list of slice indices to visualize
        num_slices: number of slices to visualize if slice_indices is None
        axis: axis along which to take slices (0=sagittal, 1=coronal, 2=axial)
    Returns:
        fig: matplotlib figure
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.permute(1, 2, 3, 0).detach().cpu().numpy()
    
    # if volume.ndim == 4:  # [C, D, H, W]
    #     volume = volume[0]  # Take first channel
    assert volume.ndim == 4
    
    if slice_indices is None:
        dim_size = volume.shape[axis]
        step = dim_size // (num_slices + 1)
        slice_indices = [step * (i + 1) for i in range(num_slices)]

    if return_slices:
        return [(volume[idx] * 255).astype(np.uint8) for idx in slice_indices]
    
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(4 * len(slice_indices), 4))
    if len(slice_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(volume[idx, ...])
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

"""
src/transforms.py
All image transformations before modeling
BoMeyering 2025
"""

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import numpy as np

def get_marker_transforms():

    return A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1)
        ]
    )

def get_seg_transforms() -> A.Compose:
    """
    Return a transform function for validation transforms, i.e. just resize and normalize.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose(
        [
            A.Normalize([0.3454, 0.4017, 0.2357], [0.1928, 0.2044, 0.1658]),
            ToTensorV2(p=1)
        ]
    )

    return transforms
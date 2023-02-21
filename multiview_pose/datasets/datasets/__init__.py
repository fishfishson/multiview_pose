# Copyright (c) OpenMMLab. All rights reserved.
from .body3d import CustomPanopticDataset
from .body3d import CustomCHI3DDataset
from mmpose.datasets.datasets import *  # noqa
__all__ = [
    'CustomPanopticDataset',
    'CustomCHI3DDataset'
]

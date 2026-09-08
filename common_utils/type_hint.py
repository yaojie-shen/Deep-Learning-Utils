# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : type_hint.py

import os
from typing import Union

import numpy as np
import torch

PathLike = Union[str, os.PathLike]
TorchOrNumpy = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[TorchOrNumpy, list, tuple]
Scalar = Union[int, float]
ArrayOrScalar = Union[ArrayLike, Scalar]

__all__ = [
    "PathLike",
    "TorchOrNumpy",
    "ArrayLike",
    "Scalar",
    "ArrayOrScalar",
]

# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : array.py

import numpy as np
import torch

from ..type_hint import ArrayLike, ArrayOrScalar


def to_numpy(array: ArrayOrScalar) -> np.ndarray:
    """
    Convert array-like object or scalar to NumPy array.

    Args:
        array: Array-like object or scalar to be converted.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        array = array.detach().cpu()
        # Convert unsupported data type
        if array.dtype is torch.bfloat16:
            array = array.float()
        return array.numpy()
    elif isinstance(array, (list, tuple, int, float)):
        return np.array(array)
    else:
        raise TypeError(f"Unsupported type: {type(array)}")


def to_tensor(array: ArrayOrScalar, *args, **kwargs) -> torch.Tensor:
    """
    Convert a scalar or array-like object to a PyTorch tensor.

    Args:
        array: Numeric scalar or array-like object to convert.
        *args: Additional arguments passed to torch.Tensor.to()
        **kwargs: Additional keyword arguments passed to torch.Tensor.to()
    """
    if isinstance(array, torch.Tensor):
        pass
    elif isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, (list, tuple, int, float)):
        array = torch.tensor(array)
    else:
        raise TypeError(f"Unsupported type: {type(array)}")

    if args or kwargs:
        array = array.to(*args, **kwargs)

    return array


def to_original(array: ArrayLike, ori_dtype) -> ArrayLike:
    """
    Convert array-like object to original type.
    """
    if ori_dtype is np.ndarray:
        return to_numpy(array)
    elif ori_dtype is torch.Tensor:
        return to_tensor(array)
    else:
        raise TypeError(f"Unsupported type: {ori_dtype}")


def is_tensor(x):
    """Is `torch.Tensor`"""
    return isinstance(x, torch.Tensor)


def is_ndarray(x):
    """Is `np.ndarray`"""
    return isinstance(x, np.ndarray)


__all__ = [
    "to_numpy",
    "to_tensor",
    "to_original",
    "is_tensor",
    "is_ndarray",
]

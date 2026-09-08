# -*- coding: utf-8 -*-
# @Time    : 9/8/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : normalize.py

__all__ = ["normalize", "inv_normalize", "convert_image_video_range"]

from typing import Tuple

import numpy as np
import torch

from .. import to_numpy, to_tensor
from ..type_hint import ArrayLike, ArrayOrScalar, TorchOrNumpy


def _prepare_mean_std(
    mean: TorchOrNumpy, std: TorchOrNumpy, dim: int, ndim: int
) -> Tuple[TorchOrNumpy, TorchOrNumpy]:
    """A helper function to prepare mean and std for normalization.
    Reshape mean and std to broadcast along the specified dim
    """
    shape = [1] * ndim
    shape[dim] = -1
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return mean, std


def _normalize(data, mean, std):
    return (data - mean) / std


def _inv_normalize(data, mean, std):
    return data * std + mean


def normalize(
    data: ArrayLike, mean: ArrayOrScalar, std: ArrayOrScalar, dim: int = -1
) -> ArrayLike:
    """Normalize the input array (usually image or video).

    Args:
        data: Input array, can be a NumPy array or a PyTorch tensor.
        mean: Numeric scalar or vector of means for each channel.
        std: Numeric scalar or vector of standard deviations for each channel.
        dim: The channel dimension to normalize along. Default is -1 (last dimension).

    Returns:
        Normalized image or video in the same type as input (NumPy array or PyTorch tensor).

    Examples:
        >>> import numpy as np
        >>> from common_utils import normalize
        >>> img = np.array([[[0, 128, 255]]], dtype=np.float32)
        >>> normalize(img, mean=128, std=64)
        array([[[-2.      ,  0.      ,  1.984375]]])

        >>> import torch
        >>> from common_utils import normalize
        >>> img_t = torch.tensor([[[0, 128, 255]]], dtype=torch.float32)
        >>> normalize(img_t, mean=torch.tensor([0, 128, 255]), std=torch.tensor([1, 64, 255]))
        tensor([[[0., 0., 0.]]], dtype=torch.float64)
    """

    if isinstance(data, list):
        return normalize(np.array(data), mean, std, dim).tolist()
    if isinstance(data, tuple):
        return tuple(normalize(np.array(data), mean, std, dim).tolist())
    elif isinstance(data, np.ndarray):
        mean, std = to_numpy(mean), to_numpy(std)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _normalize(data, mean, std)
    elif isinstance(data, torch.Tensor):
        # NOTE: Move std and mean to same device and convert to same dtype
        mean, std = to_tensor(mean, data), to_tensor(std, data)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _normalize(data, mean, std)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def inv_normalize(
    data: ArrayLike, mean: ArrayOrScalar, std: ArrayOrScalar, dim=-1
) -> ArrayLike:
    """Inverse normalize the input array (usually image or video).

    Args:
        data: Input array, can be a NumPy array or a PyTorch tensor,
              which has been previously normalized.
        mean: Numeric scalar or vector of means used in the original normalization.
        std: Numeric scalar or vector of standard deviations used in the original normalization.
        dim: The channel dimension along which normalization was applied.
             Default is -1 (last dimension).

    Returns:
        Denormalized image or video in the same type as input (NumPy array or PyTorch tensor).
    """
    if isinstance(data, list):
        return inv_normalize(np.array(data), mean, std, dim).tolist()
    if isinstance(data, tuple):
        return tuple(inv_normalize(np.array(data), mean, std, dim).tolist())
    elif isinstance(data, np.ndarray):
        mean, std = to_numpy(mean), to_numpy(std)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _inv_normalize(data, mean, std)
    elif isinstance(data, torch.Tensor):
        # NOTE: Move std and mean to same device and convert to same dtype
        mean, std = to_tensor(mean, data), to_tensor(std, data)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _inv_normalize(data, mean, std)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def convert_image_video_range(data: TorchOrNumpy, pattern: str):
    """
    Convert torch/numpy images/videos between flexible dtype and range.

    Note:
        - When converting from `uint8`, inputs may be integers or floating point values.
        - When converting from `-1_1` or `0_1`, inputs must be floating point.
        - When converting from `uint8` to `-1_1` or `0_1`, floating-point inputs preserve their dtype; integer inputs are promoted to float32.
        - When converting between `-1_1` and `0_1`, the output keeps the input’s dtype.

    Examples:

        >>> img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        >>> convert_image_video_range(img, "uint8->0_1")
        array([[[0.       , 0.5019608, 1.       ]]], dtype=float32)
        >>> img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        >>> convert_image_video_range(img, "0_1->uint8")
        array([[[0, 128, 255]]], dtype=uint8)
        >>> img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        >>> convert_image_video_range(img, "0_1->-1_1")
        array([[[-1.,  0.,  1.]]], dtype=float32)
    """
    RANGE_SPECS = {
        "uint8": ("uint8", 0.0, 255.0),
        "-1_1": ("float", -1.0, 1.0),
        "0_1": ("float", 0.0, 1.0),
    }
    try:
        in_spec, out_spec = [s.strip() for s in pattern.split("->")]
    except:
        raise ValueError(f"Invalid pattern: {pattern}")

    if in_spec not in RANGE_SPECS or out_spec not in RANGE_SPECS:
        raise ValueError(f"Unknown spec(s): {pattern}")

    assert isinstance(data, (np.ndarray, torch.Tensor)), (
        f'Data type "{type(data)}" is not supported, expected np.ndarray or torch.Tensor'
    )

    in_dtype, in_low, in_high = RANGE_SPECS[in_spec]
    out_dtype, out_low, out_high = RANGE_SPECS[out_spec]

    # check input dtype
    if in_dtype in ("uint8",):
        pass
    elif in_dtype in ("float",):
        if isinstance(data, torch.Tensor):
            assert data.is_floating_point(), "Expected floating-point tensor"
        else:
            assert np.issubdtype(data.dtype, np.floating), (
                "Expected floating-point array"
            )
    else:
        raise NotImplementedError(f"Unknown dtype: {in_dtype}")

    if in_spec == out_spec:
        return data  # If same, return directly

    if in_dtype == "uint8":
        if isinstance(data, torch.Tensor) and not data.is_floating_point():
            data = data.to(torch.float32)
        elif isinstance(data, np.ndarray) and not np.issubdtype(
            data.dtype, np.floating
        ):
            data = data.astype(np.float32)

    _ratio = (out_high - out_low) / (in_high - in_low)
    data = data * _ratio + (out_low - in_low * _ratio)
    # NOTE: A simplified version is:
    # # Normalize to [0, 1]
    # data = (data - in_low) / (in_high - in_low)
    # data = data * (out_high - out_low) + out_low

    if out_dtype == "uint8":
        data = data.round().clip(0, 255)
        return (
            data.astype(np.uint8)
            if isinstance(data, np.ndarray)
            else data.to(torch.uint8)
        )
    else:
        return data

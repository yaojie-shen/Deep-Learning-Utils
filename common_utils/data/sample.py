# -*- coding: utf-8 -*-
# @Time    : 7/21/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : sample.py

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import numpy as np

try:
    import torch
except ImportError:
    # These functions can work without torch
    torch = None


def sample_evenly(
    input_data: Union[List[Any], np.ndarray, "torch.Tensor", Sequence[Any]], n: int
) -> Union[np.ndarray, List[Any], "torch.Tensor"]:
    """
    Evenly sample N elements from input_data. Supports list, numpy array, or torch tensor.
    The input_data can be empty, and n can be less than or equal to 0, in which case it will return empty data.

    Args:
        input_data: List, numpy array, or torch tensor to sample from.
        n: Number of elements to sample.

    Returns:
        Sampled data in the same type as input_data.
    """
    if isinstance(input_data, np.ndarray):
        length = input_data.shape[0]
        if length == 0 or n <= 0:
            return input_data[:0]
        indices = np.linspace(0, length - 1, num=n).round().astype(int)
        return input_data[indices]
    elif torch and isinstance(input_data, torch.Tensor):
        length = input_data.size(0)
        if length == 0 or n <= 0:
            return input_data[:0]
        indices = torch.linspace(0, length - 1, steps=n).round().long()
        return input_data[indices]
    else:
        # Treat as a list or other iterable data structure with length.
        length = len(input_data)
        if length == 0 or n <= 0:
            return []
        indices = np.linspace(0, length - 1, num=n).round().astype(int)
        return [input_data[i] for i in indices]


def sample_randomly(
    input_data: Union[List[Any], np.ndarray, "torch.Tensor", Sequence[Any]],
    n: int,
    ordered: bool = False,
    seed: int = None,
    put_back: bool = False,
) -> Union[np.ndarray, List[Any], "torch.Tensor"]:
    """
    Randomly sample N elements from input_data. Supports list, numpy array, or torch tensor.

    Args:
        input_data: List, numpy array, or torch tensor to sample from.
        n: Number of elements to sample.
        ordered: Whether to return sampled elements in the original order.
        seed: Random seed for reproducibility.
        put_back: If True, sample with replacement.

    Returns:
        Sampled data in the same type as input_data.
    """
    assert n > 0, f"n must be positive, got {n}"
    assert len(input_data) > 0, (
        f"input_data must have at least one element, got {len(input_data)}"
    )
    if not put_back:
        assert n <= len(input_data), (
            f"n must be less than or equal to the length of input_data (without replacement), got {n} vs. {len(input_data)}"
        )

    rng = np.random.default_rng(seed)

    if isinstance(input_data, np.ndarray):
        if put_back:
            indices = rng.integers(0, len(input_data), size=n)
        else:
            indices = rng.choice(len(input_data), size=n, replace=False)
        if ordered:
            indices = np.sort(indices)
        return input_data[indices]
    elif torch and isinstance(input_data, torch.Tensor):
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        if put_back:
            indices = torch.randint(0, input_data.size(0), (n,), generator=generator)
        else:
            indices = torch.randperm(input_data.size(0), generator=generator)[:n]
        if ordered:
            indices, _ = torch.sort(indices)
        return input_data[indices]
    else:
        length = len(input_data)
        if put_back:
            indices = rng.integers(0, length, size=n)
        else:
            indices = rng.choice(length, size=n, replace=False)
        if ordered:
            indices = np.sort(indices)
        return [input_data[i] for i in indices]


def sample_contiguous(
    input_data: Union[List[Any], np.ndarray, "torch.Tensor", Sequence[Any]],
    n: int,
    seed: Optional[int] = None,
):
    """
    Randomly sample a contiguous slice of length n from input_data.

    Args:
        input_data: List, numpy array, or torch tensor to sample from.
        n: Length of the contiguous sequence to sample.
        seed: Random seed for reproducibility.

    Returns:
        A contiguous slice of length n from input_data, in the same type as input_data.
    """
    assert 0 <= n <= len(input_data), (
        f"The length of the slice must be between 0 and {len(input_data)}, but got {n}."
    )

    if n == 0:
        return input_data[:0]

    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(input_data) - n + 1)
    return input_data[start : start + n]


__all__ = [
    "sample_evenly",
    "sample_randomly",
    "sample_contiguous",
]

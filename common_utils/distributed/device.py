# -*- coding: utf-8 -*-
# @Time    : 9/22/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : device.py

__all__ = [
    "recursive_to",
]

from typing import Any

import torch


def recursive_to(obj: Any, *args, **kwargs) -> Any:
    """
    Recursively move all torch.Tensor in obj to the given device/dtype following the same behavior as torch.Tensor.to().
    Supports: Tensor, list, tuple, dict, set. Leaves other objects intact.

    Args:
        obj: The object to move.
        *args: Arguments to pass to torch.Tensor.to().
        **kwargs: Keyword arguments to pass to torch.Tensor.to().

    Returns:
        The object with all torch.Tensor moved to the given device.

    Note:
        If no device is specified, the current (gpu) device will be used. If no gpu is available, the cpu will be used.

    Note:
        This operation is in-place. You should **copy** if you want to keep the original object.
    """
    if (not args) and (not kwargs):
        kwargs["device"] = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

    if isinstance(obj, torch.Tensor):
        return obj.to(*args, **kwargs)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to(v, *args, **kwargs) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(v, *args, **kwargs) for v in obj)
    elif isinstance(obj, set):
        # sets are unordered; converting back to set might lose type/sort
        return {recursive_to(v, *args, **kwargs) for v in obj}
    else:
        return obj

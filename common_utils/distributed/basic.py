# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : basic.py

import os
from typing import Any, Callable, Union

import torch
import torch.distributed as dist


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get (global) world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def get_master_addr() -> Union[str, None]:
    return os.environ.get("MASTER_ADDR", None)


def get_master_port() -> Union[int, None]:
    port = os.environ.get("MASTER_PORT", None)
    if port is not None:
        return int(port)
    else:
        return None


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)


def rank0() -> bool:
    """Global rank 0"""
    return get_global_rank() == 0


def local_rank0() -> bool:
    """Local rank 0 (of each node)"""
    return get_local_rank() == 0


def rank0_print(*args, **kwargs):
    """Print only on rank 0"""
    if rank0():
        print(*args, **kwargs)


def rank0_log(fn, *args, **kwargs):
    """Log only on rank 0."""
    if rank0():
        fn(*args, **kwargs)


def rank0_wrapper(fn):
    """
    Wrap any function to only run on rank 0.
    """

    def wrapper(*args, **kwargs):
        if rank0():
            return fn(*args, **kwargs)

    return wrapper


def dist_info(print_fn: Callable[[str], Any] = print, prefix: str = ""):
    """
    Print torch distributed information for debugging.
    """
    is_initialized = dist.is_initialized()
    msg = f"{str(prefix) + ' ' if prefix else ''}Distributed Info".strip()
    print_fn(f"***** {msg} *****")
    if is_initialized:
        lines = [
            f"Initialized    : {is_initialized}",
            f"Backend        : {dist.get_backend()}",
            f"World size     : {dist.get_world_size()}",
            f"Rank           : {dist.get_rank()}",
            f"Master address : {get_master_addr()}",
            f"Master port    : {get_master_port()}",
        ]
        print_fn("\n".join(lines))
    else:
        print_fn("Not initialized.")
    print_fn("*" * (len(msg) + 12))


__all__ = [
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "get_device",
    "get_master_addr",
    "get_master_port",
    "barrier_if_distributed",
    "rank0",
    "local_rank0",
    "rank0_print",
    "rank0_log",
    "rank0_wrapper",
    "dist_info",
]

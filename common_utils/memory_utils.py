# -*- coding: utf-8 -*-
# @Time    : 10/29/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : memory_utils.py

__all__ = [
    "MemoryStats",
    "get_gpu_memory_state",
    "get_cpu_memory_state",
    "gc_and_empty_cache",
    "measure_memory",
]

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemoryStats:
    tag: str
    gpu_before_mb: Optional[float]
    gpu_after_mb: Optional[float]
    cpu_before_mb: Optional[float]
    cpu_after_mb: Optional[float]
    elapsed_s: float

    def as_dict(self):
        return {
            "tag": self.tag,
            "gpu_before_mb": self.gpu_before_mb,
            "gpu_after_mb": self.gpu_after_mb,
            "gpu_delta_mb": self.gpu_delta_mb,
            "cpu_before_mb": self.cpu_before_mb,
            "cpu_after_mb": self.cpu_after_mb,
            "cpu_delta_mb": self.cpu_delta_mb,
            "elapsed_s": self.elapsed_s,
        }

    @property
    def gpu_delta_mb(self) -> Optional[float]:
        if self.gpu_before_mb is None or self.gpu_after_mb is None:
            return None
        return self.gpu_after_mb - self.gpu_before_mb

    @property
    def cpu_delta_mb(self) -> Optional[float]:
        if self.cpu_before_mb is None or self.cpu_after_mb is None:
            return None
        return self.cpu_after_mb - self.cpu_before_mb

    def format_message(self, include_cpu: bool = True) -> str:
        """Return a formatted string summary, safe even if some values are None."""

        def fmt(v, precision=1):
            return f"{v:.{precision}f}" if v is not None else "N/A"

        msg = f"[{self.tag}] " if self.tag else ""

        gpu_delta = fmt(self.gpu_delta_mb)
        msg += f"GPU Δ{gpu_delta} MB"

        gpu_before = fmt(self.gpu_before_mb, precision=0)
        gpu_after = fmt(self.gpu_after_mb, precision=0)
        msg += f" (from {gpu_before} → {gpu_after} MB)"

        if include_cpu:
            cpu_delta = fmt(self.cpu_delta_mb)
            msg += f", CPU Δ{cpu_delta} MB"

        msg += f", time={fmt(self.elapsed_s, 3)}s"
        return msg


def get_gpu_memory_state(
    device: torch.device | None = None, sync: bool = True
) -> float | None:
    """Get current GPU memory state.

    Args:
        device: Device to measure memory usage. If None, the current device will be used.
        sync: Whether to synchronize the device before measuring memory usage.

    Returns:
        Current GPU memory usage in bytes. If no GPU is available, return None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available() and device.type == "cuda":
        if sync:
            torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated(device)
    else:
        return None


def get_cpu_memory_state() -> float | None:
    """Get current CPU memory state in bytes. Usually this is not important, so we allow it to fail silently."""
    try:
        import psutil
    except ImportError:
        return None
    gc.collect()
    return psutil.Process().memory_info().rss


def gc_and_empty_cache():
    """Simply a combination of `gc.collect()` and `torch.cuda.empty_cache()`."""
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def measure_memory(
    tag: str = "",
    device: torch.device | None = None,
    sync: bool = True,
    verbose: bool = True,
    report_cpu: bool = True,
):
    """
    Context manager to measure GPU/CPU memory usage during a code block.

    Args:
        tag: Tag to identify the memory usage.
        device: Device to measure memory usage. If None, the current device will be used.
        sync: Whether to synchronize the device before measuring memory usage.
        verbose: Whether to print the memory usage.
        report_cpu: Whether to report CPU memory usage.

    Examples:
        >>> with measure_memory("forward") as m:
        ...     out = model(x)
        >>> print(m.gpu_delta_mb)
    """
    start_time = time.time()

    gpu_before = get_gpu_memory_state(device, sync)
    cpu_before = get_cpu_memory_state() if report_cpu else None

    stats = MemoryStats(
        tag=tag,
        gpu_before_mb=gpu_before / 1024**2 if gpu_before is not None else None,
        gpu_after_mb=0,
        cpu_before_mb=cpu_before / 1024**2 if cpu_before is not None else None,
        cpu_after_mb=0,
        elapsed_s=0,
    )

    try:
        yield stats
    finally:
        gpu_after = get_gpu_memory_state(device, sync)
        cpu_after = get_cpu_memory_state() if report_cpu else None
        elapsed = time.time() - start_time

        stats.gpu_after_mb = gpu_after / 1024**2 if gpu_after is not None else None
        stats.cpu_after_mb = cpu_after / 1024**2 if cpu_after is not None else None
        stats.elapsed_s = elapsed

        if verbose:
            print(stats.format_message(include_cpu=report_cpu))

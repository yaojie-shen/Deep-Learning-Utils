# -*- coding: utf-8 -*-
# @Time    : 10/29/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_memory_utils.py

from dl_utils import MemoryStats, measure_memory


def test_memory_stats_as_dict_and_format_message():
    stats = MemoryStats(
        tag="test",
        gpu_before_mb=None,
        gpu_after_mb=None,
        cpu_before_mb=100.0,
        cpu_after_mb=112.5,
        elapsed_s=0.25,
    )

    assert stats.as_dict() == {
        "tag": "test",
        "gpu_before_mb": None,
        "gpu_after_mb": None,
        "gpu_delta_mb": None,
        "cpu_before_mb": 100.0,
        "cpu_after_mb": 112.5,
        "cpu_delta_mb": 12.5,
        "elapsed_s": 0.25,
    }
    assert stats.format_message(include_cpu=False) == (
        "[test] GPU ΔN/A MB (from N/A → N/A MB), time=0.250s"
    )
    assert "CPU Δ12.5 MB" in stats.format_message(include_cpu=True)


def test_measure_memory_returns_stats_without_printing():
    with measure_memory("test", verbose=False, report_cpu=False) as stats:
        data = [0] * 10

    assert data == [0] * 10
    assert stats.tag == "test"
    assert stats.cpu_before_mb is None
    assert stats.cpu_after_mb is None
    assert stats.elapsed_s >= 0

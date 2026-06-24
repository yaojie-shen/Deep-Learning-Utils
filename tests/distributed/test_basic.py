# -*- coding: utf-8 -*-
# @Time    : 10/10/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_basic.py

from dataclasses import dataclass

import pytest

from dl_utils import (
    barrier_if_distributed,
    dist_info,
    get_global_rank,
    get_local_rank,
    get_master_addr,
    get_master_port,
    get_world_size,
    local_rank0,
    rank0,
    rank0_log,
    rank0_wrapper,
)


@dataclass
class MockDist:
    _is_initialized: bool

    def is_initialized(self):
        return self._is_initialized

    def get_backend(self):
        return "nccl"

    def get_world_size(self):
        return 8

    def get_rank(self):
        return 2

    def get_master_addr(self):
        return "127.0.0.1"

    def get_master_port(self):
        return 12345


@pytest.fixture
def mock_dist(monkeypatch):
    """Mock torch.distributed functions."""

    mock = MockDist(True)
    monkeypatch.setattr("torch.distributed.is_initialized", mock.is_initialized)
    monkeypatch.setattr("torch.distributed.get_backend", mock.get_backend)
    monkeypatch.setattr("torch.distributed.get_world_size", mock.get_world_size)
    monkeypatch.setattr("torch.distributed.get_rank", mock.get_rank)
    monkeypatch.setattr(
        "dl_utils.distributed.basic.get_master_addr", mock.get_master_addr
    )
    monkeypatch.setattr(
        "dl_utils.distributed.basic.get_master_port", mock.get_master_port
    )
    return mock


@pytest.fixture
def mock_dist_not_initialized(monkeypatch):
    """Mock torch.distributed functions when NOT initialized."""

    mock = MockDist(False)
    monkeypatch.setattr("torch.distributed.is_initialized", mock.is_initialized)
    return mock


def test_rank_helpers_read_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12345")

    assert get_global_rank() == 2
    assert get_local_rank() == 0
    assert get_world_size() == 8
    assert get_master_addr() == "127.0.0.1"
    assert get_master_port() == 12345
    assert not rank0()
    assert local_rank0()


def test_rank0_log_and_wrapper_only_run_on_rank0(monkeypatch):
    calls = []

    monkeypatch.setenv("RANK", "1")
    assert rank0_wrapper(lambda: calls.append("wrapped"))() is None
    rank0_log(lambda value: calls.append(value), "logged")
    assert calls == []

    monkeypatch.setenv("RANK", "0")
    assert rank0_wrapper(lambda: "result")() == "result"
    rank0_log(lambda value: calls.append(value), "logged")
    assert calls == ["logged"]


def test_barrier_if_distributed_calls_barrier_only_when_initialized(monkeypatch):
    calls = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.barrier", lambda: calls.append("barrier"))
    assert barrier_if_distributed() is None
    assert calls == []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    barrier_if_distributed()
    assert calls == ["barrier"]


def test_dist_info_when_initialized(mock_dist):
    lines = []

    dist_info(print_fn=lines.append, prefix="Worker")

    assert lines == [
        "***** Worker Distributed Info *****",
        "Initialized    : True\nBackend        : nccl\nWorld size     : 8\nRank           : 2\nMaster address : 127.0.0.1\nMaster port    : 12345",
        "*" * len("Worker Distributed Info" + "************"),
    ]


def test_dist_info_when_not_initialized(mock_dist_not_initialized):
    lines = []

    dist_info(print_fn=lines.append)

    assert lines == [
        "***** Distributed Info *****",
        "Not initialized.",
        "*" * len("Distributed Info" + "************"),
    ]

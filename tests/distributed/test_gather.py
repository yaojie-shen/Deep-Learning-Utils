# -*- coding: utf-8 -*-
# @Time    : 6/24/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_gather.py

import pytest

from dl_utils import gather_objects


def test_gather_objects_flattens_lists_from_all_ranks(monkeypatch):
    def all_gather_object(output, input_list):
        output[0] = ["rank0"]
        output[1] = input_list
        output[2] = ["rank2-a", "rank2-b"]

    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 3)
    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    assert gather_objects(["rank1-a", "rank1-b"]) == [
        "rank0",
        "rank1-a",
        "rank1-b",
        "rank2-a",
        "rank2-b",
    ]


def test_gather_objects_requires_list_input():
    with pytest.raises(AssertionError):
        gather_objects(("not", "a", "list"))

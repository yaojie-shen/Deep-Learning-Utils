# -*- coding: utf-8 -*-
# @Time    : 10/23/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_device.py

import torch

from common_utils import recursive_to


def test_recursive_to():
    obj = {
        "a": torch.randn(1, 2, 3),
        "b": [torch.randn(1, 2, 3), torch.randn(1, 2, 3)],
        "c": (torch.randn(1, 2, 3), torch.randn(1, 2, 3)),
    }

    assert obj["a"].dtype is torch.float32

    recursive_to(obj)
    recursive_to(obj, "cpu")

    data = recursive_to(obj, torch.uint8)
    assert data["a"].dtype is torch.uint8
    assert obj["a"].dtype is not torch.uint8

    data = recursive_to(obj, torch.randn(1, 2, 3))
    assert data["a"].dtype is torch.float32

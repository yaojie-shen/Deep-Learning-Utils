# -*- coding: utf-8 -*-
# @Time    : 6/24/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_array.py

import numpy as np
import pytest
import torch

from common_utils import is_ndarray, is_tensor, to_numpy, to_original, to_tensor


def test_to_numpy_preserves_numpy_array_instance():
    array = np.array([1, 2, 3])

    assert to_numpy(array) is array


def test_to_numpy_converts_tensor_and_bfloat16():
    tensor = torch.tensor([1, 2], dtype=torch.bfloat16)
    array = to_numpy(tensor)

    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, np.array([1, 2], dtype=np.float32))


def test_to_tensor_converts_numpy_and_moves_dtype():
    array = np.array([1, 2, 3], dtype=np.int64)
    tensor = to_tensor(array, dtype=torch.float32)

    assert is_tensor(tensor)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))


def test_to_original_converts_back_to_requested_type():
    tensor = torch.tensor([1, 2, 3])
    array = np.array([1, 2, 3])

    assert is_ndarray(to_original(tensor, np.ndarray))
    assert is_tensor(to_original(array, torch.Tensor))


@pytest.mark.parametrize("value", [{"x": 1}, object()])
def test_array_converters_reject_unsupported_types(value):
    with pytest.raises(TypeError):
        to_numpy(value)

    with pytest.raises(TypeError):
        to_tensor(value)

    with pytest.raises(TypeError):
        to_original(value, list)

# -*- coding: utf-8 -*-
# @Time    : 7/21/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_sample.py

import numpy as np
import pytest
import torch

from common_utils import sample_contiguous, sample_evenly, sample_randomly


@pytest.mark.parametrize(
    "input_data, N, expected_type",
    [
        ([1, 2, 3, 4, 5], 3, list),
        (np.array([1, 2, 3, 4, 5]), 3, np.ndarray),
        (torch.tensor([1, 2, 3, 4, 5]), 3, torch.Tensor),
    ],
)
def test_sample_evenly_basic(input_data, N, expected_type):
    result = sample_evenly(input_data, N)

    assert isinstance(result, expected_type)
    assert len(result) == N
    # Check for even distribution
    if isinstance(result, list):
        assert result[0] == input_data[0]
        assert result[-1] == input_data[-1]
    elif isinstance(result, (np.ndarray, torch.Tensor)):
        assert result[0].item() == input_data[0].item()
        assert result[-1].item() == input_data[-1].item()


@pytest.mark.parametrize(
    "input_data, n",
    [
        ([], 3),
        (np.array([]), 3),
        (torch.tensor([]), 3),
        ([1, 2, 3, 4], 0),
        (np.array([1, 2, 3, 4]), 0),
        (torch.tensor([1, 2, 3, 4]), 0),
    ],
)
def test_sample_evenly_empty(input_data, n):
    assert len(sample_evenly(input_data, n)) == 0


def test_sample_evenly_more_N_than_input():
    data = [10, 20]
    result = sample_evenly(data, 5)

    assert len(result) == 5


@pytest.mark.parametrize(
    "input_data, N, put_back",
    [
        ([1, 2, 3, 4, 5], 3, False),
        ([1, 2, 3, 4, 5], 6, True),
        (np.array([1, 2, 3, 4, 5]), 3, False),
        (np.array([1, 2, 3, 4, 5]), 6, True),
        (torch.tensor([1, 2, 3, 4, 5]), 3, False),
        (torch.tensor([1, 2, 3, 4, 5]), 6, True),
    ],
)
def test_sample_randomly_basic(input_data, N, put_back):
    # Use a fixed seed to ensure reproducibility
    seed = 42
    result1 = sample_randomly(input_data, N, put_back=put_back, seed=seed)
    result2 = sample_randomly(input_data, N, put_back=put_back, seed=seed)

    # The two results should be identical with the same seed
    if isinstance(result1, list):
        assert result1 == result2
        assert len(result1) == N
    elif isinstance(result1, np.ndarray):
        assert np.array_equal(result1, result2)
        assert result1.shape[0] == N
    elif torch and isinstance(result1, torch.Tensor):
        assert torch.equal(result1, result2)
        assert result1.size(0) == N


@pytest.mark.parametrize("ordered", [True, False])
def test_sample_randomly_ordered(ordered):
    data = np.arange(10)
    result = sample_randomly(data, 5, ordered=ordered, seed=123)
    assert len(result) == 5
    if ordered:
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


@pytest.mark.parametrize(
    "input_data",
    [
        [0, 1, 2, 3, 4, 5],
        np.arange(6),
        torch.arange(6),
    ],
)
@pytest.mark.parametrize("n", [0, 1, 3, 6])
def test_sample_contiguous(input_data, n):
    seed = 123

    out = sample_contiguous(input_data, n, seed=seed)
    out2 = sample_contiguous(input_data, n, seed=seed)

    # length correctness
    assert len(out) == n

    # edge case
    if n == 0:
        return

    # contiguity in the underlying sequence
    # works for list/np/tensor because slicing preserves order
    full = (
        np.array(input_data) if not isinstance(input_data, torch.Tensor) else input_data
    )
    # convert output to array for comparison
    out_arr = np.array(out) if not isinstance(out, torch.Tensor) else out

    # find start index in the original data
    start = (full == out_arr[0]).nonzero()[0]
    assert len(start) > 0
    start = start[0]
    expected = full[start : start + n]
    assert np.allclose(out_arr, expected)

    # reproducibility with seed
    if isinstance(out, torch.Tensor):
        assert torch.equal(out, out2)
    else:
        assert np.allclose(out, out2)

    # type preservation
    assert isinstance(out, type(input_data))


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (sample_evenly, {"n": 4}),
        (sample_randomly, {"n": 4, "seed": 42}),
        (sample_contiguous, {"n": 4, "seed": 42}),
    ],
)
@pytest.mark.parametrize(
    "input_data",
    [
        list(range(100)),
        np.arange(100),
        pytest.param(torch.arange(100)),
    ],
)
def test_seed_reproducibility(func, kwargs, input_data):
    """
    Ensure that using the same random seed produces the same output
    across multiple runs for all sampling functions.
    """

    # Convert kwargs to avoid mutating shared dictionaries across parameter sets
    local_kwargs = dict(kwargs)

    # First call
    out1 = func(input_data, **local_kwargs)
    # Second call
    out2 = func(input_data, **local_kwargs)

    # Convert outputs for numerical comparison if needed
    if isinstance(out1, np.ndarray):
        assert np.array_equal(out1, out2)
    elif torch and isinstance(out1, torch.Tensor):
        assert torch.equal(out1, out2)
    else:
        assert out1 == out2

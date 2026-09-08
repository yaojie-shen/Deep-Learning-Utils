# -*- coding: utf-8 -*-
# @Time    : 10/10/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_normalize.py


import numpy as np
import pytest
import torch

from common_utils import convert_image_video_range, inv_normalize, normalize


@pytest.mark.parametrize("dtype", [np.float32, torch.float32])
@pytest.mark.parametrize("mean,std", [
    (128, 64),  # scalar
    (np.array([0, 128, 255]), np.array([1, 64, 255])),  # vector
    (torch.tensor([0, 128, 255]), torch.tensor(
        [1, 64, 255]))  # torch tensor vector
])
def test_normalize_inv_normalize(dtype, mean, std):
    # Create a dummy image/video: [2, 2, 3] shape
    if dtype == np.float32:
        img = np.array([[[0, 128, 255],
                         [64, 128, 192]]], dtype=np.float32)
    else:
        img = torch.tensor([[[0, 128, 255],
                             [64, 128, 192]]], dtype=torch.float32)

    # Normalize
    norm_img = normalize(img, mean, std, dim=-1)

    # Inverse normalize
    recovered_img = inv_normalize(norm_img, mean, std, dim=-1)

    # Convert both to numpy for comparison
    img_np = img.numpy() if isinstance(img, torch.Tensor) else img
    rec_np = recovered_img.numpy() if isinstance(
        recovered_img, torch.Tensor) else recovered_img

    # Check that recovered image matches original
    np.testing.assert_allclose(rec_np, img_np, rtol=1e-5)


@pytest.mark.parametrize("img, mean, std, dim", [
    # NumPy image, scalar mean/std, last dim
    (np.array([[[0, 128, 255]]], dtype=np.float32), 128, 64, -1),
    # NumPy image, vector mean/std, last dim
    (np.array([[[0, 128, 255]]], dtype=np.float32),
     np.array([0, 128, 255]), np.array([1, 64, 255]), -1),
    # Torch tensor, scalar mean/std, last dim
    (torch.tensor([[[0, 128, 255]]], dtype=torch.float32), 128, 64, -1),
    # Torch tensor, vector mean/std, last dim
    (torch.tensor([[[0, 128, 255]]], dtype=torch.float32),
     torch.tensor([0, 128, 255]), torch.tensor([1, 64, 255]), -1),
])
def test_normalize(img, mean, std, dim):
    result = normalize(img, mean, std, dim=dim)

    # Convert to numpy for checking
    result_np = result.numpy() if isinstance(result, torch.Tensor) else result
    mean_np = mean.numpy() if isinstance(mean, torch.Tensor) else mean
    std_np = std.numpy() if isinstance(std, torch.Tensor) else std

    # Broadcast if needed
    if np.ndim(mean_np) == 1:
        shape = [1] * result_np.ndim
        shape[dim] = -1
        mean_np = mean_np.reshape(shape)
        std_np = std_np.reshape(shape)

    expected = ((img.numpy() if isinstance(
        img, torch.Tensor) else img) - mean_np) / std_np
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.parametrize("img, mean, std, dim", [
    # NumPy image, scalar mean/std
    (np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32), 128, 64, -1),
    # NumPy image, vector mean/std
    (np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32),
     np.array([0, 128, 255]), np.array([1, 64, 255]), -1),
    # Torch tensor, scalar mean/std
    (torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32), 128, 64, -1),
    # Torch tensor, vector mean/std
    (torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32),
     torch.tensor([0, 128, 255]), torch.tensor([1, 64, 255]), -1),
])
def test_inv_normalize(img, mean, std, dim):
    result = inv_normalize(img, mean, std, dim=dim)

    # Convert to numpy for checking
    result_np = result.numpy() if isinstance(result, torch.Tensor) else result
    mean_np = mean.numpy() if isinstance(mean, torch.Tensor) else mean
    std_np = std.numpy() if isinstance(std, torch.Tensor) else std

    # Broadcast if needed
    if np.ndim(mean_np) == 1:
        shape = [1] * result_np.ndim
        shape[dim] = -1
        mean_np = mean_np.reshape(shape)
        std_np = std_np.reshape(shape)

    expected = (img.numpy() if isinstance(
        img, torch.Tensor) else img) * std_np + mean_np
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "data_np, expected_np, pattern",
    [
        (np.array([0, 128, 255], np.uint8), np.array([0.0, 128 / 255.0, 1.0], np.float32), "uint8->0_1"),
        (np.array([0, 128, 255], np.float16), np.array([0.0, 128 / 255.0, 1.0], np.float16), "uint8->0_1"),
        (np.array([0, 128, 255], np.long), np.array([0.0, 128 / 255.0, 1.0], np.float32), "uint8->0_1"),
        (np.array([0.0, 0.5, 1.0], np.float32), np.array([0, 128, 255], np.uint8), "0_1->uint8"),
        (np.array([0, 128, 255], np.uint8), np.array([-1.0, (128 / 255.0) * 2 - 1, 1.0], np.float32), "uint8->-1_1"),
        (np.array([-1.0, 0.0, 1.0], np.float32), np.array([0, 128, 255], np.uint8), "-1_1->uint8"),
        (np.array([0.0, 0.5, 1.0], np.float32), np.array([-1.0, 0.0, 1.0], np.float32), "0_1->-1_1"),
        (np.array([-1.0, 0.0, 1.0], np.float32), np.array([0.0, 0.5, 1.0], np.float32), "-1_1->0_1"),
        (np.array([0.0, 0.5, 1.0], np.float16), np.array([0, 128, 255], np.uint8), "0_1->uint8"),
        (np.array([0, 128, 255], np.uint8), np.array([0.0, 128 / 255.0, 1.0], np.float32), "uint8->0_1"),
        (np.array([0.1, 0.9], np.float32), np.array([0.1, 0.9], np.float32), "0_1->0_1"),
        (np.array([-1.0, 0.0, 1.0], np.float32), np.array([-1.0, 0.0, 1.0], np.float32), "-1_1->-1_1"),
    ]
)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_convert_image_video_range(data_np, expected_np, pattern, backend):
    # Prepare backend-specific data
    if backend == "numpy":
        data = data_np
        expected = expected_np
    else:
        data = torch.from_numpy(data_np)
        expected = torch.from_numpy(expected_np)

    out = convert_image_video_range(data, pattern)

    # Compare
    if isinstance(expected, np.ndarray):
        if expected.dtype == np.uint8:
            np.testing.assert_array_equal(out, expected)
        else:
            np.testing.assert_allclose(out, expected, atol=1e-3)
    else:  # torch
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "data,pattern",
    [
        pytest.param(np.array([1, 2, 3], dtype=np.uint8), "uint8->uint8", id="np uint8"),
        pytest.param(np.array([0, 128, 255], dtype=np.long), "uint8->uint8", id="np long"),
        pytest.param(np.array([0.0, 0.5, 1.0], dtype=np.float32), "0_1->0_1", id="np float32"),
        pytest.param(np.array([-1.0, 0.0, 1.0], dtype=np.float32), "-1_1->-1_1", id="np float32"),
        pytest.param(np.array([0.0, 0.5, 1.0], dtype=np.float16), "0_1->0_1", id="np float16"),
        pytest.param(torch.tensor([1, 2, 3], dtype=torch.uint8), "uint8->uint8", id="torch uint8"),
        pytest.param(torch.tensor([0, 128, 255], dtype=torch.long), "uint8->uint8", id="torch long"),
        pytest.param(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32), "0_1->0_1", id="torch float32"),
        pytest.param(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32), "-1_1->-1_1", id="torch float32"),
        pytest.param(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float16), "0_1->0_1", id="torch float16"),
        pytest.param(torch.tensor([0.0, 0.5, 1.0], dtype=torch.bfloat16), "0_1->0_1", id="torch bfloat16"),
    ]
)
def test_convert_image_video_range_identity_returns_original_object(data, pattern):
    out = convert_image_video_range(data, pattern)
    assert out is data  # Should return same object, not a copy


@pytest.mark.parametrize(
    "pattern",
    ["nonsense", "uint8->none", "uint8->0_2"]
)
def test_convert_image_video_range_invalid_pattern(pattern):
    with pytest.raises(ValueError):
        convert_image_video_range(np.zeros(1), pattern)

# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from pytest import fixture

from dl_utils import (
    get_video_duration,
    get_video_duration_batch,
    load_video,
    save_video,
)


def test_save_video(tmp_path):
    numpy_path = tmp_path / "test_save_video_numpy.mp4"
    torch_path = tmp_path / "test_save_video_torch.mp4"

    v = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    save_video(v, str(numpy_path), 30)
    v = torch.randint(0, 255, (10, 128, 128, 3), dtype=torch.uint8)
    save_video(v, str(torch_path), 30)

    assert numpy_path.exists()
    assert torch_path.exists()


def test_get_duration_info(tmp_path):
    v = np.random.randint(0, 255, (30, 128, 128, 3), dtype=np.uint8)
    video = str(tmp_path / "test_save_video_numpy.mp4")
    save_video(v, video, 30)

    fps, frame_count, duration = get_video_duration(video)
    batch = get_video_duration_batch([video, video])

    assert fps == 30
    assert frame_count == 30
    assert duration == 1.0
    assert batch == [(30, 30, 1.0), (30, 30, 1.0)]


@fixture
def mock_videocap():
    fake_frame = np.ones((7, 11, 3), dtype=np.uint8) * 255
    mock_instance = MagicMock()
    mock_instance.read.side_effect = [(True, fake_frame)] * 10 + [(False, None)]

    with patch("cv2.VideoCapture", return_value=mock_instance):
        yield


def test_load_video_mocked(mock_videocap):
    frames = load_video("dummy.mp4", max_frames=5)
    assert frames.shape == (5, 7, 11, 3)


def test_load_video_resize(mock_videocap):
    frames = load_video("dummy.mp4", resize=(3, 5))  # W, H
    assert frames.shape == (10, 5, 3, 3)  # F, H, W, C


def test_load_video_resize_int(mock_videocap):
    frames = load_video("dummy.mp4", resize=14)
    assert frames.shape == (10, 14, 22, 3)


def test_load_video_center_crop(mock_videocap):
    frames = load_video("dummy.mp4", center_crop=(3, 5))
    assert frames.shape == (10, 5, 3, 3)


def test_load_video_center_crop_int(mock_videocap):
    frames = load_video("dummy.mp4", center_crop=2)
    assert frames.shape == (10, 2, 2, 3)

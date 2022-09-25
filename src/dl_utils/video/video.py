# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:37
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : video.py

__all__ = ["get_duration_info"]

from typing import *

import os
from joblib import Parallel, delayed


def _get_single_video_duration_info(video_path) -> (float, float, int):
    """
    return video duration in seconds
    :param video_path: video path
    :return: video duration, fps, frame count
    """

    import cv2
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count / fps, fps, int(frame_count)


def get_duration_info(video_paths: Union[str, Iterable]) -> (float, float, int):
    """

    :param video_paths: video path or a list of video path
    :return: video duration, fps, frame count
    """
    if isinstance(video_paths, str):
        return _get_single_video_duration_info(video_paths)
    else:
        return Parallel(n_jobs=os.cpu_count())(
            delayed(_get_single_video_duration_info)(path) for path in video_paths
        )

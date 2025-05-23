# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:37
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : video.py

__all__ = ["get_duration_info", "convert_to_h265", "convert_to_h264"]

import os
import subprocess
from typing import *

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


def convert_to_h265(input_file: AnyStr, output_file: AnyStr,
                    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
                    keyint: int = None,
                    overwrite: bool = False,
                    verbose: bool = False) -> None:
    """
    convert video to h265 format using ffmpeg
    @param input_file: input path
    @param output_file: output path
    @param ffmpeg_exec:
    @param keyint:
    @param overwrite: overwrite the existing file
    @param verbose: show ffmpeg output
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # `-max_muxing_queue_size 9999` is for the problem reported in:
    # https://stackoverflow.com/questions/49686244/ffmpeg-too-many-packets-buffered-for-output-stream-01
    # <!> This may cause OOM error.
    if keyint is None:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx265", "-vtag", "hvc1",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    else:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx265", "-vtag", "hvc1", "-x265-params", f"keyint={keyint}",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(command,
                   stderr=subprocess.DEVNULL if not verbose else None,
                   stdout=subprocess.DEVNULL if not verbose else None)
    # TODO: return


def convert_to_h264(input_file: AnyStr, output_file: AnyStr,
                    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
                    keyint: int = None,
                    overwrite: bool = False,
                    verbose: bool = False) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if keyint is None:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx264",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    else:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx264", "-x264-params", f"keyint={keyint}",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(command,
                   stderr=subprocess.DEVNULL if not verbose else None,
                   stdout=subprocess.DEVNULL if not verbose else None)
    # TODO: return

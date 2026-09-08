# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:37
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : video.py

import os
import subprocess
from typing import *

import cv2
import numpy as np
from joblib import Parallel, delayed

from .. import make_parent_dirs
from ..type_hint import ArrayLike, PathLike
from .array import to_numpy


def save_video(
    frames: ArrayLike,
    save_path: PathLike,
    fps: Union[int, float] = 30,
    codec: str = "avc1",
):
    """
    Save video frames to a video file utilizing opencv.

    Args:
        frames: Video frames in shape (F, H, W, C). The pixel values should be in range [0, 255].
        save_path: Path to save video.
        fps: FPS of video, default 30.
        codec: Codec of video, default avc1.
    """
    frames = to_numpy(frames)

    height, width = frames.shape[1:3]

    make_parent_dirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Ensure the video writer is opened successfully
    if not out.isOpened():
        raise RuntimeError(
            "Failed to open video writer, consider setting the codec to 'XVID' for compatibility. "
            "Check the opencv error message above for more details."
        )

    for frame in frames:
        out.write(frame)
    out.release()


def load_video(
    video_path: PathLike,
    resize: Union[Tuple[int, int], int] = None,
    center_crop: Union[Tuple[int, int], int] = None,
    max_frames: int = None,
) -> np.ndarray:
    """
    Load a video file.

    Args:
        video_path: Path to the video file.
        resize: Resize frames to the specified size. If None, no resizing. Accepts (width, height) or int.
        center_crop: Center crop frames to the specified size. If None, no cropping. Accepts (width, height) or int.
        max_frames: Maximum number of frames to load. If None, load all frames.

    Returns:
        Frames as a NumPy array with shape (F, H, W, C). Pixel values are in [0, 255], color order is RGB.

    Note:
        - If the video is grayscale, the color channel will be replicated to 3.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture.")

    frames = []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        h, w, _ = frame.shape
        if resize is not None:
            if isinstance(resize, int):
                # Resize by short edge
                resize = (
                    (resize, int(resize * h / w))
                    if h > w
                    else (int(resize * w / h), resize)
                )
            elif isinstance(resize, tuple) or isinstance(resize, list):
                assert len(resize) == 2, "'resize' should be a tuple/list of two."
            else:
                raise ValueError(
                    f"Invalid resize value: {resize}, expected int or tuple/list of two."
                )

            frame = cv2.resize(frame, resize)

        # Center crop
        h, w, _ = frame.shape
        if center_crop is not None:
            if isinstance(center_crop, int):
                center_crop = (center_crop, center_crop)
            elif isinstance(center_crop, tuple) or isinstance(center_crop, list):
                assert len(center_crop) == 2, (
                    "'center_crop' should be a tuple/list of two."
                )
            else:
                raise ValueError(
                    f"Invalid center_crop value: {center_crop}, expected int or tuple/list of two."
                )

            assert h >= center_crop[1] and w >= center_crop[0], (
                f"Frame size ({h}, {w}) is smaller than center crop size ({center_crop[0]}, {center_crop[1]})."
            )

            frame = frame[
                h // 2 - center_crop[1] // 2 : h // 2
                + (center_crop[1] - center_crop[1] // 2),
                w // 2 - center_crop[0] // 2 : w // 2
                + (center_crop[0] - center_crop[0] // 2),
            ]

        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()

    return np.stack(frames)  # (f,h,w,c)


def get_video_fps(video_path: PathLike) -> float:
    """
    Retrieve the FPS of a video.

    Args:
        video_path: Path to the video file.

    Returns:
        The FPS of the video.
    """
    import cv2

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def get_video_frame_count(video_path: PathLike) -> int:
    """
    Retrieve the total number of frames in a video.

    Args:
        video_path: Path to the video file.

    Returns:
        The number of frames in the video.
    """
    import cv2

    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(frame_count)


def get_video_duration(video_path: PathLike) -> Tuple[float, int, float]:
    """
    Retrieve the FPS, frame count, and duration (in seconds) of a video.

    Args:
        video_path: Path to the video file.
    Returns:
        A tuple containing FPS, frame count, and duration in seconds.
    """
    import cv2

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, frame_count, frame_count / fps


def get_video_duration_batch(video_paths: List[PathLike]) -> List[float]:
    """
    Get duration of videos in batch.

    Args:
        video_paths: List of paths to videos.

    Returns:
        A list of tuples, each containing FPS, frame count, and duration in seconds.
    """
    return Parallel(n_jobs=os.cpu_count())(
        delayed(get_video_duration)(p) for p in video_paths
    )


def convert_to_h265(
    input_file: AnyStr,
    output_file: AnyStr,
    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
    keyint: int = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
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
        command = [
            ffmpeg_exec,
            "-i",
            f"{input_file}",
            "-max_muxing_queue_size",
            "9999",
            "-c:v",
            "libx265",
            "-vtag",
            "hvc1",
            "-c:a",
            "copy",
            "-movflags",
            "faststart",
            f"{output_file}",
        ]
    else:
        command = [
            ffmpeg_exec,
            "-i",
            f"{input_file}",
            "-max_muxing_queue_size",
            "9999",
            "-c:v",
            "libx265",
            "-vtag",
            "hvc1",
            "-x265-params",
            f"keyint={keyint}",
            "-c:a",
            "copy",
            "-movflags",
            "faststart",
            f"{output_file}",
        ]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(
        command,
        stderr=subprocess.DEVNULL if not verbose else None,
        stdout=subprocess.DEVNULL if not verbose else None,
    )
    # TODO: return


def convert_to_h264(
    input_file: AnyStr,
    output_file: AnyStr,
    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
    keyint: int = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if keyint is None:
        command = [
            ffmpeg_exec,
            "-i",
            f"{input_file}",
            "-max_muxing_queue_size",
            "9999",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            "-movflags",
            "faststart",
            f"{output_file}",
        ]
    else:
        command = [
            ffmpeg_exec,
            "-i",
            f"{input_file}",
            "-max_muxing_queue_size",
            "9999",
            "-c:v",
            "libx264",
            "-x264-params",
            f"keyint={keyint}",
            "-c:a",
            "copy",
            "-movflags",
            "faststart",
            f"{output_file}",
        ]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(
        command,
        stderr=subprocess.DEVNULL if not verbose else None,
        stdout=subprocess.DEVNULL if not verbose else None,
    )
    # TODO: return


__all__ = [
    "save_video",
    "load_video",
    "get_video_fps",
    "get_video_frame_count",
    "get_video_duration",
    "get_video_duration_batch",
    "convert_to_h265",
    "convert_to_h264",
]

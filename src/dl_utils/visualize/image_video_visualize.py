# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 13:55
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : image_video_visualize.py

__all__ = ["visualize_image", "visualize_video", "inv_normalize"]

import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms.functional import normalize


def _convert_to_numpy_array(data):
    # convert any type to numpy array
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, Image.Image):
        data = np.array(data)
    else:
        raise ValueError("Data type is not supported to convert to numpy array: {}".format(type(data)))
    return data


def _convert_to_torch_tensor(data):
    # convert any type tot torch tensor
    if isinstance(data, torch.Tensor):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise ValueError("Data type is not supported to convert to torch.Tensor: {}".format(type(data)))
    return data.detach().cpu()


def visualize_image(image, channel_first=False, backend="matplotlib", **backend_args):
    # (C, H, W) or (H, W, C)
    image = _convert_to_numpy_array(image)
    assert len(image.shape) == 2 or len(image.shape) == 3, f"Image shape is invalid: {image.shape}"

    if len(image.shape) == 3 and channel_first:
        image = image.transpose(1, 2, 0)

    _visualize_with_backend(image, backend, **backend_args)


def visualize_video(video, channel_first=False, backend="matplotlib", **backend_args):
    # (N, C, H, W) or (N, H, W, C)
    video = _convert_to_numpy_array(video)
    for frame in video:
        visualize_image(frame, channel_first=channel_first, backend=backend, **backend_args)


def _visualize_with_backend(content, backend, **backend_args):
    backend_func = globals()[f"_backend_{backend}"]
    backend_func(content, **backend_args)


def _backend_matplotlib(image, dpi=1024):
    sizes = np.shape(image)
    fig = plt.figure(dpi=dpi)
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    plt.show()


def _backend_opencv(image, display="localhost:10.0"):
    import cv2
    os.environ["DISPLAY"] = display
    cv2.imshow("visualize", image)
    cv2.waitKey(1)


def _backend_save_image(image, path=None):
    if path is None:
        raise ValueError("path should not be None, add it to  **backend_args")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image)


def inv_normalize(image_or_video, mean, std):
    # convert to torch.Tensor
    data = _convert_to_torch_tensor(image_or_video)
    data = normalize(data, mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
    return data.numpy()

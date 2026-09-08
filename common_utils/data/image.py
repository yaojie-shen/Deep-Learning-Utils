# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 22:33
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : image.py


from io import BytesIO

import numpy as np
from PIL import Image

__all__ = ["byte_imread", "byte_imwrite"]


def byte_imread(data: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


def byte_imwrite(image: np.ndarray, format="PNG", **kwargs) -> bytes:
    image = Image.fromarray(image)
    with BytesIO() as f:
        image.save(f, format=format, **kwargs)
        data = f.getvalue()
    return data

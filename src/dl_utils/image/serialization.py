# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 22:33
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : serialization.py


from PIL import Image
from io import BytesIO

__all__ = ['byte_imread', "byte_imwrite"]


def byte_imread(data):
    return Image.open(BytesIO(data))


def byte_imwrite(image, quality=100, subsampling=0):
    image = Image.fromarray(image)
    with BytesIO() as f:
        image.save(f, format="JPEG", quality=quality, subsampling=subsampling)
        data = f.getvalue()
    return data

# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 22:41
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_image.py

import unittest

import numpy as np

from dl_utils.data.image import *
from dl_utils.visualize import visualize_image


class TestImage(unittest.TestCase):

    def test_byte_imread_imwrite(self):
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        data = byte_imwrite(image)
        image_recover = byte_imread(data)
        visualize_image(image_recover)


if __name__ == '__main__':
    unittest.main()

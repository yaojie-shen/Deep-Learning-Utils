# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:00
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_visualize.py

import unittest
import numpy as np
from dl_utils.visualize import *


class TestPlot(unittest.TestCase):

    def test_plot_distribution(self):
        # generate random data
        data = np.random.default_rng(1).standard_normal(1000)

        plot_distribution(data, remove_outlier=True)


if __name__ == '__main__':
    unittest.main()

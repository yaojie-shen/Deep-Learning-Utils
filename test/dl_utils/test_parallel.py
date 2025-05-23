# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_parallel.py

import time
import unittest

import joblib

from dl_utils import tqdm_joblib


class TestParallel(unittest.TestCase):
    def test_tqdm_joblib(self):
        with tqdm_joblib(total=10, disable=False):
            joblib.Parallel(n_jobs=2)(joblib.delayed(time.sleep)(1) for _ in range(10))

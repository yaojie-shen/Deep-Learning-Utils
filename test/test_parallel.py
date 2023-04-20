# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_parallel.py

import unittest

from dl_utils.parallel import tqdm_joblib

import joblib
import time


class TestParallel(unittest.TestCase):
    def test_tqdm_joblib(self):
        with tqdm_joblib(total=10, disable=False):
            joblib.Parallel(n_jobs=2)([joblib.delayed(time.sleep)(1) for _ in range(100)])

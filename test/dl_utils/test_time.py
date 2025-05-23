# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 19:29
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_time.py

import time
import unittest

from dl_utils import *


class TestTime(unittest.TestCase):
    def test_get_timestamp(self):
        print(get_timestamp())

    def test_timer(self):
        timer = Timer()

        time.sleep(0.5)
        timer("S1")
        time.sleep(0.1)
        timer("S2")

        timer.print()
        print(timer)


if __name__ == '__main__':
    unittest.main()

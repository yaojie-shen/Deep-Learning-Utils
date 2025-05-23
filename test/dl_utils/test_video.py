# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

import os
import unittest

from dl_utils import get_duration_info


class TestVideo(unittest.TestCase):
    # the folder contains test videos
    video_root_dir = ""

    def test_get_duration_info(self):
        videos = [os.path.join(self.video_root_dir, f) for f in os.listdir(self.video_root_dir)]

        # single
        print("single:")
        print(get_duration_info(videos[0]))
        print("batch:")
        print(get_duration_info(videos[:10]))


if __name__ == '__main__':
    unittest.main()

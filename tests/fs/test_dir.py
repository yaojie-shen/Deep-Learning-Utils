# -*- coding: utf-8 -*-
# @Time    : 6/24/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_dir.py

from common_utils import make_parent_dirs


def test_make_parent_dirs_creates_nested_parent(tmp_path):
    path = tmp_path / "a" / "b" / "file.txt"

    make_parent_dirs(path)

    assert path.parent.is_dir()
    assert not path.exists()

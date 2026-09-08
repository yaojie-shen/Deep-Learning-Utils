# -*- coding: utf-8 -*-
# @Time    : 5/23/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_list_files.py

import os

import pytest

from common_utils import list_files, list_files_multithread


@pytest.fixture
def file_tree(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested").mkdir()
    (tmp_path / "a" / "one.txt").write_text("1", encoding="utf-8")
    (tmp_path / "a" / "nested" / "two.txt").write_text("2", encoding="utf-8")
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "depth,expected",
    [
        (
            None,
            {
                "root.txt",
                os.path.join("a", "one.txt"),
                os.path.join("a", "nested", "two.txt"),
            },
        ),
        (1, {"root.txt"}),
        (2, {"root.txt", os.path.join("a", "one.txt")}),
    ],
)
def test_list_files_respects_depth(file_tree, depth, expected):
    files = {
        os.path.relpath(path, file_tree)
        for path in list_files(str(file_tree), depth=depth)
    }

    assert files == expected


def test_list_files_multithread_matches_single_thread(file_tree):
    files = set(list_files(str(file_tree)))
    files_mt = set(list_files_multithread(str(file_tree), n_jobs=2))

    assert files_mt == files

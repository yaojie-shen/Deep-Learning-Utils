# -*- coding: utf-8 -*-
# @Time    : 6/24/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_chunk_utils.py

import pytest

from common_utils import chunk, sort_chunk


def test_chunk_by_number_balances_remainder():
    assert chunk(list(range(10)), n_chunks=3) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_chunk_by_size_and_idx():
    data = ["a", "b", "c", "d", "e"]

    assert chunk(data, chunk_size=2) == [["a", "b"], ["c", "d"], ["e"]]
    assert chunk(data, chunk_size=2, idx=1) == ["c", "d"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_chunks": 2, "chunk_size": 2},
        {"n_chunks": 0},
        {"chunk_size": 0},
    ],
)
def test_chunk_rejects_invalid_split_arguments(kwargs):
    with pytest.raises(ValueError):
        chunk([1, 2, 3], **kwargs)


def test_sort_chunk_uses_key_and_reverse():
    data = ["bbb", "a", "cc", "dddd"]

    assert sort_chunk(data, key=len, reverse=True, n_chunks=2) == [
        ["dddd", "bbb"],
        ["cc", "a"],
    ]

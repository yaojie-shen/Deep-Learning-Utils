# -*- coding: utf-8 -*-
# @Time    : 12/5/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_decorators.py

import inspect
from unittest.mock import Mock, patch

import pytest

from common_utils import barrier_on_entry, flex_kwargs, log_on_entry


def test_log_on_entry_logs_and_calls_function():
    mock_print = Mock()

    @log_on_entry(print_fn=mock_print)
    def my_func(a, b):
        return a + b

    result = my_func(2, 3)
    assert result == 5

    mock_print.assert_called_once_with("my_func")


def test_barrier_on_entry_direct():
    """Test usage: @barrier_on_entry"""
    mock_barrier = Mock()

    with patch("common_utils.distributed.basic.barrier_if_distributed", mock_barrier):

        @barrier_on_entry
        def my_func(x):
            """My fn"""
            return x + 1

        result = my_func(10)

    assert result == 11
    mock_barrier.assert_called_once()


def test_flex_kwargs():
    def fn(x, y):
        return x+y

    print(inspect.signature(fn).parameters.keys())

    assert flex_kwargs(fn)(x=1, y=2, z=3) == 3
    assert flex_kwargs(fn)(1, y=2, z=3) == 3
    assert flex_kwargs(fn)(1, 2) == 3
    with pytest.raises(TypeError):
        flex_kwargs(fn)()
    with pytest.raises(TypeError):
        flex_kwargs(fn)(x=1)

    @flex_kwargs
    def fn2(x, y):
        return x+y

    assert fn2(x=1, y=2, z=3) == 3

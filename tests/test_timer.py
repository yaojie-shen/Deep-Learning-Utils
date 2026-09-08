# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 19:29
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_timer.py

import time

from pytest import mark, param

from common_utils import get_timestamp, get_readable_timestamp, ExecutionTimer


def test_get_timestamp():
    print(get_timestamp())


def test_get_readable_timestamp():
    print(get_readable_timestamp())


@mark.parametrize("precision", [
    param(0, id="precision=0"),
    param(1, id="precision=1"),
    param(2, id="precision=2"),
])
def test_timer(precision):
    timer = ExecutionTimer(log=True, precision=precision)

    print()

    timer.start_stage("Stage 1")
    time.sleep(0.1)
    timer.end_stage("Stage 1")
    timer.start_stage()
    time.sleep(0.2)
    timer.end_stage("Stage 2")
    time.sleep(0.2)
    timer.end_stage("Stage 2")

    timer.start_stage("Stage 3")
    time.sleep(0.3)
    timer.start_stage()
    time.sleep(0.4)
    timer.end_stage("Stage 4")

    print()
    timer.print_table()
    print()
    print(timer)


def test_timer_with_name():
    timer = ExecutionTimer(log=True, name="Test Timer")
    print()
    timer.start_stage("Stage 1")
    time.sleep(0.1)
    timer.start_stage("Stage 2")
    time.sleep(0.2)

    timer.print_table()

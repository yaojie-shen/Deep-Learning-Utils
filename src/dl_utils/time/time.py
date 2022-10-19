# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:41
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : _time.py


import datetime
import torch
import time
from tabulate import tabulate
from collections import defaultdict

__all__ = ["get_timestamp", "Timer"]


def get_timestamp():
    return "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.datetime.now())


class Timer:

    def __init__(self, synchronize=False, history_size=1000, precision=3):
        self._precision = precision
        self._stage_index = 0
        self._time_info = {}
        self._time_history = defaultdict(list)
        self._history_size = history_size
        if synchronize:
            assert torch.cuda.is_available(), "cuda is not available for synchronize"
        self._synchronize = synchronize
        self._time = self._get_time()

    def _get_time(self):
        return round(time.time() * 1000, self._precision)

    def __call__(self, stage_name=None, reset=True):
        if self._synchronize:
            torch.cuda.synchronize(torch.cuda.current_device())

        current_time = self._get_time()
        duration = (current_time - self._time)
        if reset:
            self._time = current_time

        if stage_name is None:
            self._time_info[self._stage_index] = duration
        else:
            self._time_info[stage_name] = duration
            self._time_history[stage_name] = self._time_history[stage_name][-self._history_size:]
            self._time_history[stage_name].append(duration)

        return duration

    def reset(self):
        if self._synchronize:
            torch.cuda.synchronize(torch.cuda.current_device())
        self._time = time.time()

    def __str__(self):
        return str(self.get_info())

    def get_info(self):
        info = {
            "current": {k: round(v, self._precision) for k, v in self._time_info.items()},
            "average": {k: round(sum(v) / len(v), self._precision) for k, v in self._time_history.items()}
        }
        return info

    def print(self):
        data = [[k, round(sum(v) / len(v), self._precision)] for k, v in self._time_history.items()]
        print(tabulate(data, headers=["Stage", "Time (ms)"], tablefmt="simple"))

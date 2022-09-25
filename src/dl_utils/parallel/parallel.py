# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:06
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : parallel.py

"""
Provide a IterableParallel class to run in parallel.
The API is similar to joblib, but the parallel running is based on torch dataloader.
The class return an iterator in order to solve the **inefficient memory** when the output (return data) is very large.
"""

__all__ = ["delayed", "IterableParallel"]

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import *


class _ThreadDataset(Dataset):

    def __init__(self, callers: Iterable[Tuple[Callable, List, Dict]]):
        self.caller_list = list(callers)

    def __len__(self):
        return len(self.caller_list)

    def __getitem__(self, index):
        callable_func, args, kwargs = self.caller_list[index]
        return callable_func(*args, **kwargs)


# noinspection PyPep8Naming
class delayed:
    def __init__(self, callable_func: Callable):
        self.callable_func = callable_func

    def __call__(self, *args, **kwargs):
        return self.callable_func, args, kwargs


class IterableParallel:
    def __init__(self, n_jobs=1, prefetch_factor=2, batch_size=1, timeout=0,
                 verbose=False, verbose_bar_desc=None):
        self.n_jobs = n_jobs
        self.prefetch_factor = prefetch_factor
        self.batch_size = batch_size
        self.timeout = timeout
        self.verbose = verbose
        self.verbose_bar_desc = verbose_bar_desc

    @staticmethod
    def _dummy_collate_fn(x):
        return x

    def __call__(self, iterable: Iterable) -> Iterable:
        dataset = _ThreadDataset(iterable)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_jobs,
            collate_fn=self._dummy_collate_fn,
            prefetch_factor=self.prefetch_factor,
            timeout=self.timeout
        )

        bar = tqdm(desc=self.verbose_bar_desc, disable=not self.verbose, total=len(dataset))
        for processing_output in loader:
            for element in processing_output:
                bar.update(1)
                yield element

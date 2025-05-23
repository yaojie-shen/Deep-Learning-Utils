# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:06
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : parallel.py


__all__ = ["delayed", "IterableParallel", "tqdm_joblib"]

import contextlib
from typing import *

import joblib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
    """
    Run in parallel.
    The API is similar to joblib, but the parallel running is based on torch dataloader.
    The class return an iterator in order to solve the **inefficient memory** when the output (return data) is very large.
    """

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


@contextlib.contextmanager
def tqdm_joblib(iterable=None, desc=None, total=None, leave=True, file=None,
                ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
                ascii=None, disable=False, unit='it', unit_scale=False,
                dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
                position=None, postfix=None, unit_divisor=1000, write_bytes=None,
                lock_args=None, nrows=None, colour=None, delay=0, gui=False,
                **kwargs):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    Copied from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution.
    """

    tqdm_object = tqdm(iterable=iterable, desc=desc, total=total, leave=leave, file=file,
                       ncols=ncols, mininterval=mininterval, maxinterval=maxinterval, miniters=miniters,
                       ascii=ascii, disable=disable, unit=unit, unit_scale=unit_scale,
                       dynamic_ncols=dynamic_ncols, smoothing=smoothing, bar_format=bar_format, initial=initial,
                       position=position, postfix=postfix, unit_divisor=unit_divisor, write_bytes=write_bytes,
                       lock_args=lock_args, nrows=nrows, colour=colour, delay=delay, gui=gui,
                       **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

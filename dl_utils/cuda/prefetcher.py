# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : prefetcher.py


__all__ = ["CudaPreFetcher"]

from typing import Union, List, Any, AnyStr

import torch
from torch.utils import data


class CudaPreFetcher:
    def __init__(self, data_loader: data.DataLoader):
        self.dl = data_loader
        self.dataset = data_loader.dataset if hasattr(data_loader, "dataset") else None
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.batch = None

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.cuda(self.batch)

    @staticmethod
    def cuda(x: Union[list, tuple, dict, torch.Tensor]):
        if isinstance(x, list) or isinstance(x, tuple):
            return [CudaPreFetcher.cuda(i) for i in x]
        elif isinstance(x, dict):
            return {k: CudaPreFetcher.cuda(v) for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        else:
            return x

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __len__(self):
        return len(self.dl)

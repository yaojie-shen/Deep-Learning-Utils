# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : gather.py

import itertools
from typing import Any, List

import torch.distributed as dist


def gather_objects(list_object: List[Any]) -> List[Any]:
    """
    gather a list of something from multiple GPU.
    """
    assert type(list_object) == list, "This function only receive a list as input."
    gathered_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_objects, list_object)
    return list(itertools.chain(*gathered_objects))


__all__ = ["gather_objects"]

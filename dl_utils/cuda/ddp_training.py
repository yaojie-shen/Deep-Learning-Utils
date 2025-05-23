# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : ddp_training.py


__all__ = ["gather_object_multiple_gpu"]

import hashlib
import itertools
import os
import pickle
import time
from typing import List, Any, AnyStr

import torch
import torch.distributed as dist


def gather_object_multiple_gpu(list_object: List[Any], backend: AnyStr = "nccl",
                               shared_folder=None, retry: int = 600, sleep: float = 0.1):
    """
    gather a list of something from multiple GPU
    """
    assert type(list_object) == list, "`list_object` only receive list."
    assert backend in ["nccl", "filesystem"]
    if backend == "nccl":
        gathered_objects = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_objects, list_object)
        return list(itertools.chain(*gathered_objects))
    else:
        assert shared_folder is not None, "`share_folder` should be set if backend is `filesystem`"
        os.makedirs(shared_folder, exist_ok=True)
        uuid = torch.randint(2 ** (8 * 4), 2 ** (8 * 4 + 1), size=(1,), dtype=torch.long).cuda()
        dist.all_reduce(uuid)
        uuid = hex(uuid.cpu().item())[-8:]
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.pkl"), "wb") as f:
            data = pickle.dumps(list_object)
            f.write(data)
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.md5"), "wb") as f:
            checksum = hashlib.md5(data).hexdigest()
            pickle.dump(checksum, f)
        gathered_list = []
        dist.barrier()
        for rank in range(dist.get_world_size()):
            data_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.pkl")
            checksum_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.md5")
            data = None
            error = None
            for _ in range(retry):
                time.sleep(sleep)
                try:
                    if not os.path.exists(data_filename):
                        continue
                    if not os.path.exists(checksum_filename):
                        continue
                    raw_data = open(data_filename, "rb").read()
                    checksum = pickle.load(open(checksum_filename, "rb"))
                    assert checksum == hashlib.md5(raw_data).hexdigest()
                    data = pickle.loads(raw_data)
                    break
                except Exception as e:
                    error = e
            assert data is not None, f"Gather from filesystem failed after retry for {retry} times, last error: {error}"
            gathered_list.extend(data)
        dist.barrier()
        return gathered_list

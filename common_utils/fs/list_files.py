# -*- coding: utf-8 -*-
# @Time    : 5/23/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : list_files.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union


def list_files(path: str, depth: Union[int, None] = None) -> List[str]:
    """
    List all files in a folder recursively.

    Args:
        path: Root path to start the search.
        depth: Maximum depth to search.
            If None, there is no depth limit.
            If 0 or less, stop searching deeper.

    Returns: A List of file paths found under the given path.
    """
    if os.path.isdir(path):
        if depth is not None and depth <= 0:
            return []
        next_depth = depth - 1 if depth is not None else None

        files = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            files.extend(list_files(full_path, depth=next_depth))
        return files
    else:
        return [path]


def list_files_multithread(directory, n_jobs=16, depth: Union[int, None] = None):
    """
    List all files in a directory recursively using multiple threads.
    Useful for list files on NFS.

    Args:
        directory: The directory to search.
        n_jobs: Number of parallel jobs (threads) to use.
        depth: Maximum recursion depth. If None, no depth limit.

    Returns: List of all file paths found under the directory.
    """
    entries = os.listdir(directory)
    next_depth = depth - 1 if depth is not None else None
    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                list_files, os.path.join(directory, entry), depth=next_depth
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            results.extend(future.result())
    return results


__all__ = [
    "list_files",
    "list_files_multithread",
]

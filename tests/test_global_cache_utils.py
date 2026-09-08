# -*- coding: utf-8 -*-
# @Time    : 9/12/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_global_cache_utils.py

import gc
import multiprocessing
import random
import time

import pytest

from dl_utils import GlobalCache


@pytest.fixture(scope="function")
def cache():
    cache = GlobalCache(max_size=2)
    cache.clear()
    cache["a"] = 1
    cache["b"] = 2
    return cache


def test_global_cache(cache):
    cache2 = GlobalCache()
    assert cache2.get("a") == 1
    assert cache2.get("b") == 2

    del cache, cache2
    gc.collect()

    cache3 = GlobalCache()
    assert cache3.get("a") == 1
    assert cache3.get("b") == 2


def worker(key, value, return_dict):
    cache = GlobalCache()
    time.sleep(random.random() * 0.1)
    cache[key] = value
    # Return the value from this process
    return_dict[key] = cache.get(key)


def test_global_cache_multiprocessing():
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(100):
        p = multiprocessing.Process(target=worker, args=(f"key{i}", i, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Ensure values set in one process are not seen by other processes
    # The return_dict contains values from each process only
    for i in range(100):
        assert return_dict[f"key{i}"] == i


def test_global_cache_in(cache):
    assert "a" in cache
    assert "b" in cache
    assert "c" not in cache


def test_global_cache_len(cache):
    assert len(cache) == 2
    cache["c"] = 3
    assert len(cache) == 2  # max size is 2


def test_global_cache_items(cache):
    assert list(cache.items()) == [("a", 1), ("b", 2)]


def test_global_cache_del(cache):
    del cache["a"]
    assert "a" not in cache
    assert "b" in cache


def test_global_cache_iter(cache):
    assert list(iter(cache)) == ["a", "b"]


def test_global_cache_get_with_fn(cache):
    cache.clear()
    assert "a" not in cache
    cache.get("a", lambda x: x + 1, x=1)
    assert cache.get("a") == 2


def test_global_cache_get_missing_without_fn_raises_key_error(cache):
    cache.clear()

    with pytest.raises(KeyError):
        cache.get("missing")


def test_global_cache_get_updates_lru_order(cache):
    assert cache.get("a") == 1
    cache["c"] = 3

    assert "b" not in cache
    assert list(cache.cache.items()) == [("a", 1), ("c", 3)]


def test_global_cache_persistence(cache):
    dump = cache.to_bytes()
    cache.clear()
    assert "a" not in cache
    assert "b" not in cache
    cache = GlobalCache.from_bytes(dump)
    assert cache.get("a") == 1
    assert cache.get("b") == 2

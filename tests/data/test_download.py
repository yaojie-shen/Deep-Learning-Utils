# -*- coding: utf-8 -*-
# @Time    : 1/13/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_download.py

import hashlib
import multiprocessing as mp
import os

import pytest

from common_utils import download


def test_download_reuses_existing_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first payload")
    target = tmp_path / "payload.bin"

    downloaded_path = download(source.as_uri(), filepath=str(target))

    assert downloaded_path == str(target)
    assert target.read_bytes() == b"first payload"

    source.write_bytes(b"changed payload")
    downloaded_path = download(source.as_uri(), filepath=str(target))

    assert downloaded_path == str(target)
    assert target.read_bytes() == b"first payload"


def test_download_with_sha256(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    target = tmp_path / "payload.bin"
    expected_sha256 = hashlib.sha256(b"payload").hexdigest()

    downloaded_path = download(
        source.as_uri(), filepath=str(target), expected_sha256=expected_sha256
    )

    assert downloaded_path == str(target)
    assert target.read_bytes() == b"payload"


def test_download_removes_tmp_file_when_sha256_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    target = tmp_path / "payload.bin"

    with pytest.raises(RuntimeError):
        download(source.as_uri(), filepath=str(target), expected_sha256="wrong")

    assert not target.exists()
    assert not (tmp_path / "payload.bin.tmp").exists()


def _worker(url, path, queue):
    try:
        queue.put(download(url, filepath=path))
    except Exception as e:
        queue.put(e)


def test_parallel_download_multiprocessing(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    target_path = str(tmp_path / "payload.bin")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(source.as_uri(), target_path, queue))
        for _ in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    results = [queue.get() for _ in processes]

    assert all(isinstance(result, str) for result in results)
    assert all(result == target_path for result in results)
    assert os.path.isfile(target_path)
    assert os.path.getsize(target_path) > 0
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]

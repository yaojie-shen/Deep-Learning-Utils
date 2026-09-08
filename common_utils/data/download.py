# -*- coding: utf-8 -*-
# @Time    : 1/13/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : download.py

import hashlib
import os
import urllib.request
import warnings
from typing import Optional

from filelock import FileLock
from tqdm import tqdm

__all__ = ["download"]


# Modified from: https://github.com/openai/CLIP/blob/main/clip/clip.py
def download(
    url: str,
    filepath: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    cache_dir: str = "~/.cache/common_utils",
):
    """
    Download file from URL to the given path in a multi-process safe way.

    Internal logic:
    - Return early if file already downloaded
    - Verify using SHA256 if provided
    - File lock to avoid concurrent downloads
    - Temporary file + atomic rename to avoid partial files

    Args:
        url: Download URL.
        filepath: Path to save the file. If not provided, the file is saved under `cache_dir` with filename
            `os.path.basename(url)`.
        expected_sha256: Expected SHA256 checksum.
        cache_dir: Cache directory if filepath is not specified. By default, it is `~/.cache/common_utils`.

    Returns:
        Path to the downloaded file.
    """
    # If `filepath` is not specified, download into `cache_dir` using the URL basename as filename.
    if filepath is None:
        filename = os.path.basename(url)
        filepath = os.path.join(os.path.expanduser(cache_dir), filename)

    filepath = os.path.expanduser(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    lock_path = filepath + ".lock"
    tmp_path = filepath + ".tmp"

    # File-level lock: only one process can download at a time
    with FileLock(lock_path):
        # If file already exists and checksum is correct, reuse it
        if os.path.isfile(filepath):
            if expected_sha256 is None:
                return filepath

            with open(filepath, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()

            if sha256 == expected_sha256:
                return filepath
            else:
                warnings.warn(
                    f"{filepath} exists but checksum mismatch; re-downloading."
                )

        # Download to temporary file
        with urllib.request.urlopen(url) as source, open(tmp_path, "wb") as output:
            total = source.info().get("Content-Length")
            total = int(total) if total is not None else None

            with tqdm(
                total=total,
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                disable=total is None,
            ) as pbar:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    pbar.update(len(buffer))

        # Verify checksum after download
        if expected_sha256 is not None:
            with open(tmp_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            if sha256 != expected_sha256:
                os.remove(tmp_path)
                raise RuntimeError(
                    "Downloaded file checksum does not match expected SHA256"
                )

        # Atomic replace: safe even with concurrent readers
        os.replace(tmp_path, filepath)

    return filepath

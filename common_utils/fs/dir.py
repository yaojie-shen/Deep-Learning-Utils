# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : dir.py

from pathlib import Path

from ..type_hint import PathLike


def make_parent_dirs(path: PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


__all__ = ["make_parent_dirs"]

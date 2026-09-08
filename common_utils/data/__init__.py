# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : __init__.py.py

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule, define_import_structure

if TYPE_CHECKING:
    from .array import *
    from .download import *
    from .image import *
    from .lmdb import *
    from .normalize import *
    from .sample import *
    from .save_and_load import *
    from .video import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_import_structure(_file), module_spec=__spec__
    )

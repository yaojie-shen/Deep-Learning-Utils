# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 18:52
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : __init__.py

from typing import TYPE_CHECKING

from .import_utils import _LazyModule, define_import_structure

if TYPE_CHECKING:
    from .chunk_utils import *
    from .data import *
    from .decorators import *
    from .distributed import *
    from .env_utils import *
    from .fs import *
    from .global_cache_utils import *
    from .id_utils import *
    from .inference import *
    from .inspect_data_utils import *
    from .mask_utils import *
    from .memory_utils import *
    from .prefetcher import *
    from .timer import *
    from .type_hint import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_import_structure(_file), module_spec=__spec__
    )

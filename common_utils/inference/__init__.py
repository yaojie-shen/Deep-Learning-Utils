# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : __init__.py

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule, define_import_structure

if TYPE_CHECKING:
    from .llm_utils import *
    from .ollama_utils import *
    from .qps_control import *
    from .ray_inference_utils import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_import_structure(_file), module_spec=__spec__
    )

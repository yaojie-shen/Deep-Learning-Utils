# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : decorators.py

import inspect
from functools import wraps
from typing import Any, Callable, Optional


def log_on_entry(
    fn: Callable | None = None, print_fn: Optional[Callable[[str], Any]] = None
) -> Callable:
    """
    Functions with this decorator will log the function name at entry.
    When using multiple decorators, this must be applied innermost to properly capture the name.
    """
    if print_fn is None:
        print("print_fn is none")
        print_fn = lambda x: print(f"Entering {x}")

    if fn is None:
        return lambda _fn: log_on_entry(_fn, print_fn=print_fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        print_fn(str(fn.__name__))
        return fn(*args, **kwargs)

    return wrapper


def barrier_on_entry(fn: Callable) -> Callable:
    """
    Functions with this decorator will start executing when all ranks are ready to enter.
    """
    from .distributed.basic import barrier_if_distributed

    @wraps(fn)
    def wrapper(*args, **kwargs):
        barrier_if_distributed()
        return fn(*args, **kwargs)

    return wrapper


def flex_kwargs(fn: Callable) -> Callable:
    """
    Decorator that allows a function to ignore extra keyword arguments.
    Only the parameters that the function actually accepts will be passed in.
    """
    sig = inspect.signature(fn)
    accepted = sig.parameters.keys()

    @wraps(fn)
    def wrapper(*args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return fn(*args, **filtered)

    return wrapper


__all__ = ["log_on_entry", "barrier_on_entry", "flex_kwargs"]

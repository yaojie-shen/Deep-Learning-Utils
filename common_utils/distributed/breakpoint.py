# -*- coding: utf-8 -*-
# @Time    : 7/21/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : breakpoint.py

import sys

from .basic import barrier_if_distributed, get_local_rank, get_world_size


def _require_ipython():
    """Try to install IPython if failed to import it."""
    try:
        import IPython
    except ImportError:
        import pip

        if hasattr(pip, "main"):
            pip.main(["install", "IPython"])
        else:
            pip._internal.main(["install", "IPython"])


def _my_embed(*, stack_depth=2, header="", compile_flags=None, **kwargs):
    """
    This is a modified version of IPython.terminal.embed.embed(), add `stack_depth` to arguments.
    """
    # Install IPython if failed to import it.
    _require_ipython()

    from IPython.core.interactiveshell import InteractiveShell
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.ipapp import load_default_config

    config = kwargs.get("config")
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs["config"] = config
    using = kwargs.get("using", "sync")
    colors = kwargs.pop("colors", "nocolor")
    if using:
        kwargs["config"].update(
            {
                "TerminalInteractiveShell": {
                    "loop_runner": using,
                    "colors": colors,
                    "autoawait": using != "sync",
                }
            }
        )
    # save ps1/ps2 if defined
    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass
    # save previous instance
    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
    frame = sys._getframe(1)
    shell = InteractiveShellEmbed.instance(
        _init_location_id="%s:%s" % (frame.f_code.co_filename, frame.f_lineno), **kwargs
    )
    shell(
        header=header,
        stack_depth=stack_depth,
        compile_flags=compile_flags,
        _call_location_id="%s:%s" % (frame.f_code.co_filename, frame.f_lineno),
    )
    InteractiveShellEmbed.clear_instance()
    # restore previous instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
        for subclass in cls._walk_mro():
            subclass._instance = saved_shell_instance
    if ps1 is not None:
        sys.ps1 = ps1
        sys.ps2 = ps2


def dist_breakpoint(rank: int = 0):
    """
    Breakpoint for distributed training.
    Enter the breakpoint only if the current rank is `rank`, and block all other processes using distributed barrier.
    """
    assert 0 <= rank < get_world_size(), (
        f"Invalid rank {rank}, world size: {get_world_size()}."
    )
    if get_local_rank() == rank:
        _my_embed(stack_depth=3)
    barrier_if_distributed()


__all__ = ["dist_breakpoint"]

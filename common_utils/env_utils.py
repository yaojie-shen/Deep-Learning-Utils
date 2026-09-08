# -*- coding: utf-8 -*-
# @Time    : 3/6/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : env_utils.py

import os
from pathlib import Path


def _normalize_newlines(value: str, allow_multiline: bool) -> str:
    """Normalize or strip newlines from a string.

    - If allow_multiline is False, only the first line is returned, with all line breaks removed.
    - If allow_multiline is True, preserve multiple lines but normalize to "\n" line endings.
    """
    if allow_multiline:
        # Normalize Windows/Mac newlines to \n for consistency
        return value.replace("\r\n", "\n").replace("\r", "\n")
    else:
        # Remove all line breaks entirely to return a single line string
        return value.splitlines()[0]


def get_env(
    name: str, allow_multiline: bool = False, cwd: str | Path | None = None
) -> str | list[str]:
    """Get a secret or config value from environment or a dotfile in the working directory.

    Lookup order (later overwrites earlier if both exist):

    1) A file named ``{name}`` under the current working directory (or ``cwd`` if provided).
    2) A file named ``.{name}`` under the current working directory (or ``cwd`` if provided).
    3) Environment variable ``name``.

    By default, the returned string contains no line breaks. Set ``allow_multiline=True`` to preserve
    multiple lines (line endings normalized to ``\\n``).

    Args:
        name: Environment variable name, and the filename stem for the ``.{name}`` file.
        allow_multiline: Whether to preserve multiple lines in the returned string.
        cwd: Optional directory to resolve the ``.{name}`` file from. Defaults to ``Path.cwd()``.

    Returns:
        The resolved secret/config value as a string.

    Raises:
        KeyError: If the value is not found in either the environment or the ``.{name}`` file.
    """

    workdir = Path(cwd) if cwd is not None else Path.cwd()

    # Check files in priority order: {name}, then .{name}
    for filename in (name, f".{name}"):
        path = workdir / filename
        if path.is_file():
            return _normalize_newlines(
                path.read_text(encoding="utf-8"),
                allow_multiline=allow_multiline,
            )

    # Finally check environment variable
    chosen = os.environ.get(name)

    if chosen is None:
        raise KeyError(
            f"'{name}' not found in environment or files '{workdir / name}' or '{workdir / f'.{name}'}'. "
            f"Set env var {name} or create a file '{name}' or '.{name}' in {workdir}."
        )

    return _normalize_newlines(chosen, allow_multiline=allow_multiline)


__all__ = [
    "get_env",
]

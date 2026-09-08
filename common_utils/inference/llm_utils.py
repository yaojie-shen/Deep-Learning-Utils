# -*- coding: utf-8 -*-
# @Time    : 2/6/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : llm_utils.py

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, os.PathLike]


def _normalize_extensions(extensions: Sequence[str]) -> tuple:
    return tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)


def _resolve_versioned_path(
    resource_dir: PathLike,
    resource_name: str,
    *,
    version: Optional[str] = None,
    extensions: Sequence[str],
    resource_type: str = "resource",
) -> Path:
    resource_dir = Path(resource_dir).expanduser()
    resource_name_path = Path(resource_name)

    if resource_name_path.suffix:
        candidates = [resource_dir / resource_name_path]
    else:
        normalized_extensions = _normalize_extensions(extensions)
        candidates = []

        if version is not None:
            version = str(version)
            for ext in normalized_extensions:
                candidates.extend(
                    [
                        resource_dir / f"{resource_name}_{version}{ext}",
                        resource_dir / f"{resource_name}.{version}{ext}",
                        resource_dir / resource_name / f"{version}{ext}",
                    ]
                )
        else:
            candidates.extend(
                resource_dir / f"{resource_name}{ext}" for ext in normalized_extensions
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find {resource_type} file. Tried: {candidate_text}")


def load_prompt_schema(
    prompt_dir: PathLike,
    prompt_name: str,
    version: Optional[str] = None,
    *,
    extensions: Sequence[str] = (".schema.json", ".json"),
    encoding: str = "utf-8",
) -> Any:
    """
    Load the JSON schema associated with a prompt.

    Args:
        prompt_dir: Directory that stores prompt and schema files.
        prompt_name: Prompt/schema name under ``prompt_dir``. If it has no
            suffix, ``extensions`` are tried in order. When ``version`` is set,
            common versioned names are tried, such as ``name_v1.schema.json``,
            ``name.v1.schema.json`` and ``name/v1.schema.json``.
        version: Optional schema version saved as a separate file.
        extensions: Candidate schema file extensions. Defaults to
            ``.schema.json`` and ``.json``.
        encoding: File encoding used when reading schema files.

    Returns:
        Parsed JSON schema object.
    """
    path = _resolve_versioned_path(
        prompt_dir,
        prompt_name,
        version=version,
        extensions=extensions,
        resource_type="prompt schema",
    )
    return json.loads(path.read_text(encoding=encoding))


def _load_prompt_schema_for_path(
    prompt_path: Path,
    *,
    extensions: Sequence[str] = (".schema.json", ".json"),
    encoding: str = "utf-8",
) -> Any:
    return load_prompt_schema(
        prompt_path.parent,
        prompt_path.stem,
        extensions=extensions,
        encoding=encoding,
    )


def load_prompt(
    prompt: Optional[PathLike] = None,
    *,
    prompt_dir: Optional[PathLike] = None,
    prompt_name: Optional[str] = None,
    version: Optional[str] = None,
    extensions: Sequence[str] = (".txt", ".md"),
    load_schema: bool = False,
    schema_extensions: Sequence[str] = (".schema.json", ".json"),
    encoding: str = "utf-8",
) -> Union[str, Tuple[str, Any]]:
    """
    Load prompt text from a file path or a prompt directory.

    Args:
        prompt: Prompt file path.
        prompt_dir: Directory that stores prompt files.
        prompt_name: Prompt name under ``prompt_dir``. If it has no suffix,
            ``extensions`` are tried in order. When ``version`` is set, common
            versioned names are tried, such as ``name_v1.txt``, ``name.v1.txt``
            and ``name/v1.txt``.
        version: Optional prompt version saved as a separate file.
        extensions: Candidate prompt file extensions. Defaults to ``.txt`` and
            ``.md``.
        load_schema: If ``True``, also load the prompt's matching JSON schema
            with :func:`load_prompt_schema` and return ``(prompt_text, schema)``.
        schema_extensions: Candidate schema file extensions used when
            ``load_schema`` is ``True``. Defaults to ``.schema.json`` and
            ``.json``.
        encoding: File encoding used when reading prompt files.

    Returns:
        Loaded prompt text, or ``(prompt_text, schema)`` when ``load_schema`` is
        ``True``.
    """
    if prompt_dir is not None or prompt_name is not None:
        if prompt_dir is None or prompt_name is None:
            raise ValueError("prompt_dir and prompt_name must be provided together.")
        path = _resolve_versioned_path(
            prompt_dir,
            prompt_name,
            version=version,
            extensions=extensions,
            resource_type="prompt",
        )
        prompt_text = path.read_text(encoding=encoding)
        if load_schema:
            if Path(prompt_name).suffix:
                schema = _load_prompt_schema_for_path(
                    path, extensions=schema_extensions, encoding=encoding
                )
            else:
                schema = load_prompt_schema(
                    prompt_dir,
                    prompt_name,
                    version=version,
                    extensions=schema_extensions,
                    encoding=encoding,
                )
            return prompt_text, schema
        return prompt_text

    if prompt is None:
        raise ValueError(
            "Either prompt or both prompt_dir and prompt_name must be provided."
        )

    path = Path(prompt).expanduser()
    prompt_text = path.read_text(encoding=encoding)
    if load_schema:
        schema = _load_prompt_schema_for_path(
            path, extensions=schema_extensions, encoding=encoding
        )
        return prompt_text, schema
    return prompt_text


def _format_bracket_prompt(prompt: str, variables: Mapping[str, Any]) -> str:
    pattern = re.compile(r"(?<!\[)\[([A-Za-z_]\w*)\](?!\])")

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(key)
        return str(variables[key])

    return pattern.sub(replace, prompt).replace("[[", "[").replace("]]", "]")


def format_prompt(
    prompt: str,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    style: Optional[str] = "format",
    **kwargs: Any,
) -> str:
    """
    Fill placeholders in a prompt string.

    Args:
        prompt: Prompt text to format.
        variables: Placeholder values.
        style: Placeholder style. ``format``/``brace``/``python`` uses Python
            ``str.format`` syntax like ``{name}`` and escapes literal braces as
            ``{{text}}``. ``bracket``/``square`` uses ``[name]`` and escapes
            literal brackets as ``[[text]]``. ``none`` disables formatting.
        **kwargs: Extra placeholder values. Values here override ``variables``.

    Returns:
        Formatted prompt text.
    """
    data = dict(variables or {})
    data.update(kwargs)

    if style is None or style == "none":
        return prompt

    if style in {"format", "brace", "python"}:
        return prompt.format_map(data)

    if style in {"bracket", "square"}:
        return _format_bracket_prompt(prompt, data)

    raise ValueError(
        "style must be one of: 'format', 'brace', 'python', 'bracket', 'square', 'none'."
    )


def render_prompt(
    prompt: Optional[PathLike] = None,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    prompt_dir: Optional[PathLike] = None,
    prompt_name: Optional[str] = None,
    version: Optional[str] = None,
    extensions: Sequence[str] = (".txt", ".md"),
    load_schema: bool = False,
    schema_extensions: Sequence[str] = (".schema.json", ".json"),
    encoding: str = "utf-8",
    style: Optional[str] = "format",
    **kwargs: Any,
) -> Union[str, Tuple[str, Any]]:
    """
    Load a prompt from path and fill placeholders in one call.

    Args:
        prompt: Prompt file path.
        variables: Placeholder values.
        prompt_dir: Directory that stores prompt files.
        prompt_name: Prompt name under ``prompt_dir``.
        version: Optional prompt version saved as a separate file.
        extensions: Candidate prompt file extensions.
        load_schema: If ``True``, also load the prompt's matching JSON schema
            and return ``(rendered_prompt, schema)``.
        schema_extensions: Candidate schema file extensions used when
            ``load_schema`` is ``True``.
        encoding: File encoding used when reading prompt files.
        style: Placeholder style passed to :func:`format_prompt`.
        **kwargs: Extra placeholder values. Values here override ``variables``.

    Returns:
        Rendered prompt text, or ``(rendered_prompt, schema)`` when
        ``load_schema`` is ``True``.
    """
    loaded_prompt = load_prompt(
        prompt,
        prompt_dir=prompt_dir,
        prompt_name=prompt_name,
        version=version,
        extensions=extensions,
        load_schema=load_schema,
        schema_extensions=schema_extensions,
        encoding=encoding,
    )

    if load_schema:
        prompt_text, schema = loaded_prompt
        return format_prompt(prompt_text, variables, style=style, **kwargs), schema

    prompt_text = loaded_prompt
    return format_prompt(prompt_text, variables, style=style, **kwargs)


def extract_json(llm_output: str) -> Any:
    """
    Extract and parse JSON from an LLM output string.
    The output may contain text before or after the JSON, including markdown code fences. Returns the parsed Python object.

    Raises:
        ValueError: if no valid JSON can be found or parsed.
    """
    # 1. Try to extract JSON inside Markdown code fences first
    fence_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\}|$begin:math:display$\.\*\?$end:math:display$)\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    fence_match = fence_pattern.search(llm_output)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to more general extraction

    # 2. Fallback: find the first valid JSON object or array by scanning
    start_indices = [i for i, ch in enumerate(llm_output) if ch in "{["]

    for start in start_indices:
        for end in range(len(llm_output), start, -1):
            if llm_output[end - 1] not in "}]":
                continue
            candidate = llm_output[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found in LLM output.")


__all__ = [
    "extract_json",
    "format_prompt",
    "load_prompt",
    "load_prompt_schema",
    "render_prompt",
]

"""Utilities for parsing named processing steps."""

from collections.abc import Mapping


StepInput = str | list[str] | tuple[str, ...]


def _normalize_steps(
    value: StepInput,
    *,
    argument_name: str,
    separator: str,
) -> list[str]:
    if isinstance(value, str):
        items = value.split(separator)
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise TypeError(
            f"{argument_name} must be a string, list, or tuple, "
            f"got {type(value).__name__}"
        )

    normalized = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(
                f"Every item in {argument_name} must be a string, "
                f"got {type(item).__name__}"
            )

        item = item.strip()
        if not item:
            raise ValueError(f"{argument_name} must not contain empty step names")
        normalized.append(item)

    if not normalized:
        raise ValueError(f"{argument_name} must contain at least one step")

    return normalized


def parse_steps(
    steps: StepInput,
    *,
    all_steps: StepInput | None = None,
    aliases: Mapping[str, str] | None = None,
    separator: str = ",",
) -> list[str]:
    """Parse requested processing steps into canonical step names.

    String inputs are split using ``separator``. Lists and tuples are treated as
    already separated, so their individual items are never split again. Leading
    and trailing whitespace is removed from every step name.

    The reserved step name ``"all"`` expands to ``all_steps`` and must be used
    on its own. ``all_steps`` is otherwise optional: when supplied, it also acts
    as the set of valid canonical step names. Aliases are replaced with their
    canonical names before validation. Duplicate results are removed while
    preserving the order of their first occurrence.

    Args:
        steps: Requested steps as a separated string, list, or tuple.
        all_steps: Optional canonical step names. This is required when
            ``steps`` is ``"all"`` and enables unknown-step validation when
            provided for ordinary selections.
        aliases: Optional mapping from external aliases to canonical step names.
            When ``all_steps`` is provided, every alias target must be present
            in it.
        separator: String used to split string inputs. Defaults to ``,``.

    Returns:
        Canonical step names in execution order.

    Raises:
        TypeError: If an input has an unsupported type or contains non-string
            items.
        ValueError: If an input is empty, ``all`` is combined with another
            step, ``all_steps`` is missing for ``all``, an alias conflicts with
            a canonical name, or a selected step is unknown.

    Examples:
        Parse a comma-separated command-line value and resolve aliases:

        >>> from dl_utils import parse_steps
        >>> parse_steps(
        ...     "prepare, infer, export-json",
        ...     all_steps=["prepare", "inference", "export"],
        ...     aliases={"infer": "inference", "export-json": "export"},
        ... )
        ['prepare', 'inference', 'export']

        Expand ``all`` in the order defined by ``all_steps``:

        >>> parse_steps("all", all_steps=("prepare", "inference", "export"))
        ['prepare', 'inference', 'export']

        The result can be mapped directly to processing functions:

        >>> handlers = {
        ...     "prepare": lambda: "prepared",
        ...     "inference": lambda: "inferred",
        ... }
        >>> selected = parse_steps(["prepare", "infer"], aliases={"infer": "inference"})
        >>> [handlers[name]() for name in selected]
        ['prepared', 'inferred']
    """
    if not isinstance(separator, str):
        raise TypeError(f"separator must be a string, got {type(separator).__name__}")
    if not separator:
        raise ValueError("separator must not be empty")

    requested = _normalize_steps(
        steps,
        argument_name="steps",
        separator=separator,
    )

    canonical_steps = None
    canonical_set = None
    if all_steps is not None:
        canonical_steps = _normalize_steps(
            all_steps,
            argument_name="all_steps",
            separator=separator,
        )
        # Remove duplicates while preserving their first-seen order.
        canonical_steps = list(dict.fromkeys(canonical_steps))
        canonical_set = set(canonical_steps)
        if "all" in canonical_set:
            raise ValueError("'all' is reserved and cannot be a canonical step name")

    if aliases is None:
        normalized_aliases = {}
    elif not isinstance(aliases, Mapping):
        raise TypeError(f"aliases must be a mapping, got {type(aliases).__name__}")
    else:
        normalized_aliases = {}
        for alias, canonical in aliases.items():
            if not isinstance(alias, str) or not isinstance(canonical, str):
                raise TypeError("Alias names and targets must be strings")

            alias = alias.strip()
            canonical = canonical.strip()
            if not alias or not canonical:
                raise ValueError("Alias names and targets must not be empty")
            if alias == "all" or canonical == "all":
                raise ValueError(
                    "'all' is reserved and cannot be used as an alias or alias target"
                )
            if canonical_set is not None and canonical not in canonical_set:
                raise ValueError(
                    f"Alias {alias!r} targets unknown canonical step {canonical!r}"
                )
            if (
                canonical_set is not None
                and alias in canonical_set
                and alias != canonical
            ):
                raise ValueError(
                    f"Alias {alias!r} conflicts with an existing canonical step"
                )

            normalized_aliases[alias] = canonical

    if "all" in requested:
        if any(step != "all" for step in requested):
            raise ValueError("'all' must be used on its own")
        if canonical_steps is None:
            raise ValueError("all_steps must be provided when steps is 'all'")
        return canonical_steps.copy()

    resolved = []
    unknown = []
    for step in requested:
        if canonical_set is not None and step in canonical_set:
            canonical = step
        else:
            canonical = normalized_aliases.get(step, step)

        if canonical_set is not None and canonical not in canonical_set:
            unknown.append(step)
        else:
            resolved.append(canonical)

    if unknown:
        unknown = list(dict.fromkeys(unknown))
        names = ", ".join(repr(step) for step in unknown)
        raise ValueError(f"Unknown step name(s): {names}")

    return list(dict.fromkeys(resolved))


__all__ = ["parse_steps"]

# -*- coding: utf-8 -*-
# @Time    : 1/28/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : id_utils.py
import hashlib
import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable


def list_ids(
    roots: str | list[str],
    include: str | re.Pattern | list[re.Pattern] = "auto",
    exclude: str | re.Pattern | list[re.Pattern] | None = None,
    matching: str | re.Pattern = "auto",
    simple: bool = True,
    return_filepath: bool = False,
    return_dict: bool = False,
) -> list[str] | tuple[list[str], list[str]] | dict[str, str]:
    """
    Extract sample IDs from file or folder names under given root paths.

    Args:
        roots: Root path(s) to search for files or folders.
        include: Patterns to include files/folders in directory. If "auto", include all files if there are more files than folders, else include all folders.
        exclude: Patterns to exclude files/folders in directory.
        matching: Patterns to extract IDs from file/folder names. If "auto", use common patterns: for files use the full name without extension, for folders use the full folder name.
        simple: If True, use `os.listdir` and treat entries with a suffix as files (heuristic). This avoids filesystem stat calls and is faster on large NFS directories.
        return_filepath: Whether to return the full file/folder paths along with IDs.
        return_dict: Whether to return a dictionary mapping IDs to file/folder paths. Only effective when return_filepath is True.

    Returns: A list of extracted IDs, or a tuple of (IDs, file/folder paths) if return_filepath is True.
    """

    def as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def compile_patterns(patterns):
        compiled = []
        for pattern_like in as_list(patterns):
            if isinstance(pattern_like, re.Pattern):
                compiled.append(pattern_like)
            else:
                compiled.append(re.compile(pattern_like))
        return compiled

    def matches_any(name: str, patterns: Iterable[re.Pattern]) -> bool:
        return any(pattern.search(name) for pattern in patterns)

    roots = [Path(r) for r in as_list(roots)]
    exclude_patterns = compile_patterns(exclude)

    pattern = None
    if matching != "auto":
        pattern = matching if isinstance(matching, re.Pattern) else re.compile(matching)

    ids: list[str] = []
    paths: list[str] | None = [] if (return_filepath or return_dict) else None

    for root in roots:
        if not root.exists():
            continue

        entries: list[tuple[str, str, bool, bool]] = []  # (name, path, is_file, is_dir)
        if simple:
            # Heuristic mode: classify by whether the name has a suffix.
            try:
                for name in os.listdir(root):
                    is_file = bool(Path(name).suffix)
                    is_dir = not is_file
                    entries.append((name, str(root / name), is_file, is_dir))
            except OSError:
                continue
        else:
            # Accurate mode: scandir + is_file/is_dir (may stat). Still much faster than Path.is_file per entry.
            try:
                with os.scandir(root) as it:
                    for de in it:
                        try:
                            is_file = de.is_file()
                        except OSError:
                            is_file = False
                        try:
                            is_dir = de.is_dir()
                        except OSError:
                            is_dir = False
                        entries.append((de.name, de.path, is_file, is_dir))
            except OSError:
                continue

        if not entries:
            continue

        files = [e for e in entries if e[2]]
        dirs = [e for e in entries if e[3]]

        # --- auto include logic ---
        if include == "auto":
            candidates = files if len(files) >= len(dirs) else dirs
            include_patterns = []
        else:
            candidates = entries
            include_patterns = compile_patterns(include)

        for name, path_str, is_file, is_dir in candidates:
            # include filter
            if include_patterns and not matches_any(name, include_patterns):
                continue

            # exclude filter
            if exclude_patterns and matches_any(name, exclude_patterns):
                continue

            # --- ID extraction ---
            if matching == "auto":
                if is_file:
                    sample_id = Path(name).stem
                else:
                    sample_id = name
            else:
                m = pattern.search(name)  # type: ignore[union-attr]
                if not m:
                    continue
                sample_id = m.group(1) if m.groups() else m.group(0)

            ids.append(sample_id)
            if paths is not None:
                p = Path(path_str)
                # In simple mode we avoid resolve() to prevent extra filesystem access on NFS.
                paths.append(str(p.absolute() if simple else p.resolve()))

    if return_dict:
        return {id_: path for id_, path in zip(ids, paths)}  # type: ignore[arg-type]

    return (ids, paths) if return_filepath else ids


def generate_id(seed: str | None = None) -> str:
    """Generate a sample ID.

    Two modes:
    - `seed is None`: generate a random ID (UUIDv4).
    - `seed is not None`: generate a deterministic ID from the input string (hash).

    Args:
        seed: Seed string used for deterministic ID generation. If None, return a random ID.

    Returns:
        A 32-character lowercase hex string.
    """
    if seed is None:
        # UUIDv4: 122 bits of randomness; collision probability is negligible for practical use.
        return uuid.uuid4().hex

    # Deterministic ID from input string.
    h = hashlib.blake2b(seed.encode("utf-8"), digest_size=16)
    return h.hexdigest()


def index_by_id(
    entries: Iterable[Any] | None,
    key: Callable[[Any], Any] | None = None,
    ignore_duplicates: bool = False,
) -> dict[str, Any]:
    """Build a dict index from entries.

    Args:
        entries: Entries to index.
        key: Function used to extract a key from each entry. If None, try common
            id-like keys from dict entries.
        ignore_duplicates: Whether to ignore duplicated keys. If False, raise a
            ValueError when duplicated keys are found. If True, skip duplicate
            checks and let later entries overwrite earlier ones.

    Returns:
        A dictionary mapping extracted keys to original entries.
    """
    default_id_keys = ("id", "uuid", "key", "name")

    def get_default_id(entry: Any) -> str:
        if not isinstance(entry, dict):
            raise ValueError(
                "Cannot infer id from non-dict entry. Please provide `key` explicitly."
            )

        for field in default_id_keys:
            value = entry.get(field)
            if value is not None:
                return str(value)

        raise ValueError(
            f"Cannot infer id from dict entry. Tried keys {default_id_keys}. Please provide `key` explicitly."
        )

    index: dict[str, Any] = {}

    for entry in entries or []:
        value = key(entry) if key is not None else get_default_id(entry)

        if value is None:
            continue

        value = str(value)
        if not ignore_duplicates and value in index:
            raise ValueError(f"Duplicated key found: {value}")

        index[value] = entry

    return index


class IdSampleManager:
    """Manage a unique ID pool, named selections, and non-overlapping sampling.

    This class only operates on unique IDs so it can be reused with arbitrary data
    structures outside the manager. Sampling results are recorded as named
    selections, and consumed selections are automatically excluded from later
    sampling operations.

    Args:
        ids: Unique IDs representing the full pool.
        records: Existing named records to restore. Each record should contain
            `ids`, and can optionally include `kind`, `consumed`, and `metadata`.
    """

    def __init__(
        self,
        ids: Iterable[Any],
        records: dict[str, dict[str, Any]] | None = None,
    ):
        self._all_ids = set(self._normalize_ids(ids, field_name="ids"))
        self._records: dict[str, dict[str, Any]] = {}

        for name, record in (records or {}).items():
            self.add_record(
                name=name,
                ids=record.get("ids", []),
                kind=str(record.get("kind", "manual")),
                consumed=bool(record.get("consumed", False)),
                metadata=record.get("metadata"),
            )

    @staticmethod
    def _normalize_ids(ids: Iterable[Any] | None, field_name: str) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in ids or []:
            id_ = str(value)
            if id_ in seen:
                raise ValueError(f"Duplicated ID found in {field_name}: {id_}")
            normalized.append(id_)
            seen.add(id_)

        return normalized

    def _ensure_name_available(self, name: str, overwrite: bool = False):
        if not name:
            raise ValueError("Record name must be a non-empty string.")
        if not overwrite and name in self._records:
            raise ValueError(f"Record already exists: {name}")

    def _ensure_known_ids(self, ids: Iterable[str], field_name: str):
        unknown_ids = [id_ for id_ in ids if id_ not in self._all_ids]
        if unknown_ids:
            raise ValueError(
                f"Found IDs outside the managed pool in {field_name}: {unknown_ids[:5]}"
            )

    def _serialize_ids(self, ids: Iterable[str]) -> list[str]:
        return sorted(set(ids))

    def _collect_ids(self, names: Iterable[str]) -> set[str]:
        collected: set[str] = set()
        for name in names:
            collected.update(self.get_record_ids(name))
        return collected

    def _store_record(
        self,
        name: str | None,
        ids: Iterable[str],
        *,
        kind: str,
        consumed: bool,
        metadata: dict[str, Any] | None,
        overwrite: bool = False,
    ) -> list[str]:
        record_ids = set(ids)
        if name is None:
            return self._serialize_ids(record_ids)

        self._ensure_name_available(name, overwrite=overwrite)
        self._records[name] = {
            "ids": set(record_ids),
            "kind": kind,
            "consumed": consumed,
            "metadata": dict(metadata or {}),
        }
        return self._serialize_ids(record_ids)

    @property
    def all_ids(self) -> list[str]:
        """Return all managed IDs."""
        return self._serialize_ids(self._all_ids)

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        """Return a copy of all named records."""
        return {name: self.get_record(name) for name in self._records}

    def __len__(self) -> int:
        return len(self._all_ids)

    def has_record(self, name: str) -> bool:
        """Return whether a record with the given name exists."""
        return name in self._records

    def list_records(
        self,
        kind: str | None = None,
        consumed: bool | None = None,
    ) -> list[str]:
        """List record names filtered by kind and/or consumed flag."""
        names: list[str] = []
        for name, record in self._records.items():
            if kind is not None and record["kind"] != kind:
                continue
            if consumed is not None and record["consumed"] != consumed:
                continue
            names.append(name)
        return names

    def get_record(self, name: str) -> dict[str, Any]:
        """Return a copy of a named record."""
        if name not in self._records:
            raise KeyError(f"Unknown record: {name}")

        record = self._records[name]
        return {
            "ids": self._serialize_ids(record["ids"]),
            "kind": record["kind"],
            "consumed": record["consumed"],
            "metadata": dict(record.get("metadata") or {}),
        }

    def get_record_ids(self, name: str) -> list[str]:
        """Return the IDs stored in a named record."""
        return self.get_record(name)["ids"]

    def find_records_by_id(
        self,
        id_: Any,
        *,
        kind: str | None = None,
        consumed: bool | None = None,
    ) -> list[str]:
        """Find all record names containing the given ID."""
        id_ = str(id_)
        results: list[str] = []
        for name, record in self._records.items():
            if id_ not in record["ids"]:
                continue
            if kind is not None and record["kind"] != kind:
                continue
            if consumed is not None and record["consumed"] != consumed:
                continue
            results.append(name)
        return results

    def add_record(
        self,
        name: str,
        ids: Iterable[Any],
        *,
        kind: str = "manual",
        consumed: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Add a named ID record.

        Args:
            name: Record name.
            ids: IDs to register. IDs must belong to the managed pool.
            kind: Record type, e.g. `sample`, `manual`, or `derived`.
            consumed: Whether these IDs should be excluded from future sampling.
            metadata: Optional metadata stored with the record.
            overwrite: Whether to overwrite an existing record with the same name.

        Returns:
            The normalized IDs stored in the record.
        """
        normalized_ids = self._normalize_ids(ids, field_name=f"record `{name}`")
        self._ensure_known_ids(normalized_ids, field_name=f"record `{name}`")
        return self._store_record(
            name,
            normalized_ids,
            kind=kind,
            consumed=consumed,
            metadata=metadata,
            overwrite=overwrite,
        )

    def remove_record(self, name: str) -> dict[str, Any]:
        """Remove and return a named record."""
        record = self.get_record(name)
        del self._records[name]
        return record

    def get_selected_ids(self, consumed_only: bool = False) -> list[str]:
        """Return the union of selected IDs across records."""
        selected: set[str] = set()
        for record in self._records.values():
            if consumed_only and not record["consumed"]:
                continue
            selected.update(record["ids"])
        return self._serialize_ids(selected)

    def get_remaining_ids(self) -> list[str]:
        """Return IDs that have not been consumed by sampling records."""
        consumed_ids = set(self.get_selected_ids(consumed_only=True))
        return self._serialize_ids(self._all_ids - consumed_ids)

    def sample_record(
        self,
        n: int,
        name: str,
        *,
        seed: int | None = None,
        include_ids: Iterable[Any] | None = None,
        exclude_ids: Iterable[Any] | None = None,
        allow_partial: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Sample IDs without repeating previously consumed IDs.

        Args:
            n: Number of IDs to sample.
            name: Record name used to store this sampling result.
            seed: Optional random seed for reproducibility.
            include_ids: Optional subset of pool IDs to sample from.
            exclude_ids: Optional extra IDs to exclude.
            allow_partial: If True, return all remaining candidates when `n`
                exceeds the available count.
            metadata: Optional metadata stored with the sampling record.
            overwrite: Whether to overwrite an existing record with the same name.

        Returns:
            The sampled IDs.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        candidates = self.get_remaining_ids()

        if include_ids is not None:
            included = self._normalize_ids(include_ids, field_name="include_ids")
            self._ensure_known_ids(included, field_name="include_ids")
            included_set = set(included)
            candidates = [id_ for id_ in candidates if id_ in included_set]

        if exclude_ids is not None:
            excluded = self._normalize_ids(exclude_ids, field_name="exclude_ids")
            self._ensure_known_ids(excluded, field_name="exclude_ids")
            excluded_set = set(excluded)
            candidates = [id_ for id_ in candidates if id_ not in excluded_set]

        if not allow_partial and n > len(candidates):
            raise ValueError(
                f"Requested {n} IDs but only {len(candidates)} are available."
            )

        if allow_partial:
            n = min(n, len(candidates))

        sampled_ids = random.Random(seed).sample(candidates, n)

        return self._store_record(
            name,
            sampled_ids,
            kind="sample",
            consumed=True,
            metadata=metadata,
            overwrite=overwrite,
        )

    def union(
        self,
        *names: str,
        name: str | None = None,
        consumed: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Return the union of multiple records; store it only when `name` is provided."""
        ids = self._collect_ids(names)
        return self._store_record(
            name,
            ids,
            kind="derived",
            consumed=consumed,
            metadata=metadata,
            overwrite=overwrite,
        )

    def intersection(
        self,
        *names: str,
        name: str | None = None,
        consumed: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Return the intersection of multiple records; store it only when `name` is provided."""
        if not names:
            raise ValueError("intersection requires at least one record name")

        common_ids = set(self.get_record_ids(names[0]))
        for record_name in names[1:]:
            common_ids &= set(self.get_record_ids(record_name))

        return self._store_record(
            name,
            common_ids,
            kind="derived",
            consumed=consumed,
            metadata=metadata,
            overwrite=overwrite,
        )

    def difference(
        self,
        base_name: str,
        *other_names: str,
        name: str | None = None,
        consumed: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Return IDs in `base_name` but not in the other records; store only when `name` is provided."""
        remaining = set(self.get_record_ids(base_name))
        for record_name in other_names:
            remaining -= set(self.get_record_ids(record_name))

        return self._store_record(
            name,
            remaining,
            kind="derived",
            consumed=consumed,
            metadata=metadata,
            overwrite=overwrite,
        )

    def complement(
        self,
        *names: str,
        name: str | None = None,
        consumed: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Return IDs in the full pool excluding the given records; store only when `name` is provided."""
        ids = self._all_ids - self._collect_ids(names)
        return self._store_record(
            name,
            ids,
            kind="derived",
            consumed=consumed,
            metadata=metadata,
            overwrite=overwrite,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manager into a JSON-friendly dictionary."""
        return {
            "all_ids": self.all_ids,
            "records": {
                name: {
                    "ids": self._serialize_ids(record["ids"]),
                    "kind": record["kind"],
                    "consumed": record["consumed"],
                    "metadata": dict(record.get("metadata") or {}),
                }
                for name, record in self._records.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdSampleManager":
        """Restore a manager from a dictionary returned by :meth:`to_dict`."""
        return cls(ids=data.get("all_ids", []), records=data.get("records", {}))

    def save_json(self, path: str | Path, indent: int = 2):
        """Save manager state as JSON."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=indent))

    @classmethod
    def load_json(cls, path: str | Path) -> "IdSampleManager":
        """Load manager state from a JSON file."""
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))


__all__ = [
    "list_ids",
    "generate_id",
    "index_by_id",
    "IdSampleManager",
]

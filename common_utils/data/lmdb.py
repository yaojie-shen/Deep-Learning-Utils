# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : lmdb.py

__all__ = ["JsonLmdb"]

import json
from typing import Any

from lmdbm import Lmdb


class JsonLmdb(Lmdb):
    """Persistent string-to-JSON mapping backed by LMDB.

    ``JsonLmdb`` provides the dictionary-style interface of ``lmdbm.Lmdb``
    while automatically converting string keys to UTF-8 bytes and values to
    JSON bytes. It is useful for metadata, indexes, inference results, and other
    JSON-compatible records that need fast lookup by key.

    Values must be serializable by ``json.dumps``. Common supported types
    include dictionaries, lists, strings, numbers, booleans, and ``None``.
    NumPy arrays, PyTorch tensors, arbitrary Python objects, and raw binary data
    must be converted before storage.

    Examples:
        Create a database and write individual or batched records:

        >>> from common_utils import JsonLmdb
        >>> with JsonLmdb.open("metadata.lmdb", "c") as db:
        ...     db["sample-0001"] = {"path": "images/0001.jpg", "label": 3}
        ...     db.update({
        ...         "sample-0002": {"path": "images/0002.jpg", "label": 7},
        ...         "sample-0003": {"path": "images/0003.jpg", "label": 3},
        ...     })

        Open the database in read-only mode and access records like a mapping:

        >>> with JsonLmdb.open("metadata.lmdb", "r") as db:
        ...     record = db["sample-0001"]
        ...     keys = list(db.keys())
        >>> record["label"]
        3
        >>> "sample-0001" in keys
        True

    Notes:
        ``update()`` is preferable to repeated item assignment for bulk writes
        because :mod:`lmdbm` performs the batch in a single transaction.

        LMDB supports many concurrent readers but only one writer at a time.
        When multiple processes may write, disable automatic map growth and set
        a sufficiently large map size explicitly, for example
        ``JsonLmdb.open(path, "c", map_size=2**40, autogrow=False)``. See the
        `lmdbm documentation <https://pypi.org/project/lmdbm/>`_ for the full
        opening options and concurrency warning.

        This class does not compress values. Compression can be added by
        subclassing and overriding ``_pre_value`` and ``_post_value``.
    """

    def _pre_key(self, value: str) -> bytes:
        """Encode a string key before writing it to LMDB."""
        return value.encode("utf-8")

    def _post_key(self, value: bytes) -> str:
        """Decode a key read from LMDB into a string."""
        return value.decode("utf-8")

    def _pre_value(self, value: Any) -> bytes:
        """Serialize a JSON-compatible value before writing it to LMDB."""
        return json.dumps(value).encode("utf-8")

    def _post_value(self, value: bytes) -> Any:
        """Deserialize a JSON value read from LMDB."""
        return json.loads(value.decode("utf-8"))

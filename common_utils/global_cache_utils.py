# -*- coding: utf-8 -*-
# @Time    : 9/12/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : global_cache_utils.py
import pickle
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Callable, Optional


class GlobalCache(MutableMapping):
    """Create a singleton key-value cache.

    Args:
        max_size: Maximum size of the cache. Defaults to None.

    Examples:

        Set and retrieve a cached value:

        >>> cache = GlobalCache()
        >>> cache["a"] = 1
        >>> cache2 = GlobalCache()
        >>> cache2["a"]
        1

        Retrieve a value with a function and arguments, automatically caching the result:

        >>> cache = GlobalCache()
        >>> cache.get("b", fn=lambda x: x+1, x=1)
        2
        >>> cache["b"]
        2
    """

    _instance = None  # singleton instance

    def __new__(cls, max_size=None):
        """Return the singleton instance"""
        if cls._instance is None:
            cls._instance = super(GlobalCache, cls).__new__(cls)
            cls._instance.max_size = max_size
            cls._instance.cache = OrderedDict()
        return cls._instance

    def __getitem__(self, key):
        """Get cached value; if exists, update its usage order"""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value  # Move to the end (most recently used)
            return value
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """Set cached value, supports LRU eviction"""
        if key in self.cache:
            self.cache.pop(key)
        elif self.max_size and len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Evict the oldest item
        self.cache[key] = value

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        # NOTE: Convert to list first to avoid 'OrderedDict mutated during iteration' error
        return iter(list(self.cache))

    def get(self, key, fn: Optional[Callable] = None, **kwargs) -> Any:
        """
        Retrieve a value from the cache, or compute and store it if missing.
        This method will call the provided function `fn` with `kwargs` to compute the value

        Args:
            key: The key to look up in the cache.
            fn (Optional[Callable]): A function to compute the value if the key is missing.
            **kwargs: Keyword arguments to pass to `fn`.

        Returns:
            The cached or newly computed value.

        Raises:
            KeyError: If the key is missing and no `fn` is provided.
        """
        if key in self:
            return self[key]

        if fn is None:
            raise KeyError(key)

        value = fn(**kwargs)
        self[key] = value
        return value

    # ---------------- Persistence ----------------
    def save(self, file_path: str):
        """Save only cache content to file.

        Args:
            file_path: Path to the cache file using pickle.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.cache, f)

    @classmethod
    def load(
        cls, file_path: str, max_size: Optional[int] = None, update: bool = True
    ) -> "GlobalCache":
        """Load cache from file and optionally set new max_size, return a GlobalCache instance.

        Args:
            file_path: Path to the cache file.
            max_size: Maximum size of the cache.
            update: If True (default), merge the loaded cache into the existing cache.
                If False, replace the current cache entirely with the loaded data.
        """
        with open(file_path, "rb") as f:
            cache_data = pickle.load(f)
        instance = cls(max_size=max_size)

        if update:
            instance.cache.update(cache_data)
        else:
            instance.cache = cache_data
        return instance

    def to_bytes(self) -> bytes:
        """Serialize cache content to bytes using pickle.

        Returns:
            Serialized cache content in bytes.
        """
        return pickle.dumps(self.cache)

    @classmethod
    def from_bytes(
        cls, data: bytes, max_size: Optional[int] = None, update: bool = True
    ) -> "GlobalCache":
        """Deserialize bytes and optionally set new max_size, return a GlobalCache instance.

        Args:
            data: Serialized cache content.
            max_size: Maximum size of the cache.
            update: If True (default), merge the loaded cache into the existing cache.
                If False, replace the current cache entirely with the loaded data.
        """
        cache_data = pickle.loads(data)
        instance = cls(max_size=max_size)

        if update:
            instance.cache.update(cache_data)
        else:
            instance.cache = cache_data
        return instance


__all__ = ["GlobalCache"]

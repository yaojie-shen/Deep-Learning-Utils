# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : chunk_utils.py

from typing import Any, Callable, List, Optional, Union


def chunk(
    data: List[Any],
    n_chunks: Optional[int] = None,
    chunk_size: Optional[int] = None,
    idx: Optional[int] = None,
) -> Union[List[List[Any]], List[Any]]:
    """
    Split a list into multiple smaller chunks.

    You can specify either:
      - `n_chunks`: the number of chunks to create, which will divide the list into approximately equal parts, or
      - `chunk_size`: the number of elements each chunk should have.

    Args:
        data: The list to chunk.
        n_chunks: The number of chunks to split the list into.
        chunk_size: The size of each chunk.
        idx: The index of the chunk to return. If None, return all chunks.

    Returns:
        A list of chunks, or a single chunk if idx is specified.
    """
    if n_chunks is not None and chunk_size is not None:
        raise ValueError("Only one of n_chunks or chunk_size can be set")
    if n_chunks is None and chunk_size is None:
        raise ValueError("One of n_chunks or chunk_size must be set")

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    elif n_chunks is not None:
        if n_chunks <= 0:
            raise ValueError("Number of chunks must be greater than 0")

        chunk_size = len(data) // n_chunks
        remainder = len(data) % n_chunks

        chunks = []
        start = 0
        for i in range(n_chunks):
            # Calculate the end index for the current chunk
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
    # integrity check
    assert sum(len(c) for c in chunks) == len(data), (
        "The total length of chunks is not equal to the length of the input data."
    )

    if idx is not None:
        return chunks[idx]
    else:
        return chunks


def sort_chunk(
    data: List[Any],
    key: Callable[[Any], Any] = lambda x: x,
    reverse: bool = False,
    *args,
    **kwargs,
) -> Union[List[List[Any]], List[Any]]:
    """
    Sort a list and then split it into chunks.
    Useful in distributed processing where input data may be unordered and must be sorted before splitting into chunks.

    Args:
        data: The list to sort and chunk.
        key: A function to extract the sort key for each element.
        reverse: Whether to sort in descending order.
        args, kwargs: Arguments passed to the ``chunk`` function
                         (e.g., n_chunks or chunk_size).

    Returns:
        A list of chunks of the sorted data.
    """
    data = sorted(data, key=key, reverse=reverse)
    return chunk(data, *args, **kwargs)


__all__ = ["chunk", "sort_chunk"]

# -*- coding: utf-8 -*-
# @Time    : 1/20/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_save_and_load.py
import json
import warnings
from pathlib import Path

from tqdm import tqdm

from common_utils import (
    JsonlHelper,
    concurrent_file_loader,
    iter_files,
    iter_jsonl,
    load_bytes,
    load_files,
    load_json,
    load_jsonl,
    load_pickle,
    load_text,
    save_bytes,
    save_json,
    save_jsonl,
    save_pickle,
    save_text,
)


def test_save_and_load_text(tmp_path):
    f = tmp_path / "sub" / "hello.txt"

    save_text("hello world", str(f))

    assert f.exists()
    assert load_text(str(f)) == "hello world"

    save_text("hello again\n\n\n", str(f))
    assert load_text(str(f)) == "hello again\n\n\n"


def test_save_and_load_json_with_bytes(tmp_path):
    f = tmp_path / "data.json"

    data = {"num": 1, "lst": [1, 2, 3], "b": b"bytes"}
    save_json(data, str(f))
    loaded = load_json(str(f))

    assert loaded["num"] == 1
    assert loaded["lst"] == [1, 2, 3]
    # bytes should be converted to a utf-8 string by the custom encoder
    assert isinstance(loaded["b"], str)
    assert loaded["b"] == "bytes"


def test_concurrent_file_loader_large_scale(tmp_path):
    num_files = 150

    file_paths = []
    for i in range(num_files):
        path = tmp_path / f"file_{i:03d}.json"
        save_json({"index": i, "content": f"data_{i}"}, path)
        file_paths.append(path)

    results = list(
        concurrent_file_loader(
            file_paths,
            loader=load_json,
            concurrency_limit=2,
            chunk_size=10,
        )
    )

    assert len(results) == num_files
    for i, data in enumerate(results):
        assert data["index"] == i
        assert data["content"] == f"data_{i}"


def test_iter_and_load_files_from_file_list(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_json([{"id": 1}, {"id": 2}], first)
    save_json({"id": 3}, second)

    assert list(
        iter_files([second, first], pattern="*.json", loader=load_json, n_jobs=2)
    ) == [
        {"id": 3},
        [{"id": 1}, {"id": 2}],
    ]
    assert load_files(
        [first, second], pattern="*.json", flatten=True, loader=load_json, n_jobs=2
    ) == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]


def test_iter_and_load_files_file_list_without_pattern(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_json({"id": 1}, first)
    save_json({"id": 2}, second)

    assert list(iter_files([second, first], loader=load_json, n_jobs=2)) == [
        {"id": 2},
        {"id": 1},
    ]
    assert load_files([first, second], loader=load_json, n_jobs=2) == [
        {"id": 1},
        {"id": 2},
    ]


def test_iter_and_load_files_default_to_bytes_loader(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    save_bytes(b"first", first)
    save_bytes(b"second", second)

    assert list(iter_files([second, first], n_jobs=2)) == [b"second", b"first"]
    assert load_files([first, second], n_jobs=2) == [b"first", b"second"]


def test_load_files_directory_requires_pattern(tmp_path):
    save_json({"id": 1}, tmp_path / "data.json")

    try:
        load_files(tmp_path, n_jobs=2)
        assert False, "Expected ValueError when loading a directory without pattern"
    except ValueError as exc:
        assert "pattern" in str(exc)


def test_load_files_from_directory_sorted_and_recursive_glob(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    save_json({"name": "b"}, tmp_path / "b.json")
    save_json({"name": "a"}, tmp_path / "a.json")
    save_json({"name": "nested"}, nested / "c.json")
    save_text("ignore", tmp_path / "ignore.txt")

    assert load_files(tmp_path, pattern="*.json", loader=load_json, n_jobs=2) == [
        {"name": "a"},
        {"name": "b"},
    ]
    assert load_files(tmp_path, pattern="**/*.json", loader=load_json, n_jobs=2) == [
        {"name": "a"},
        {"name": "b"},
        {"name": "nested"},
    ]


def test_load_files_accepts_loader_function(tmp_path):
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")

    assert load_files(tmp_path, pattern="*.txt", loader=load_text, n_jobs=2) == [
        "A",
        "B",
    ]


def test_iter_files_accepts_custom_loader_and_kwargs(tmp_path):
    first = tmp_path / "1.txt"
    second = tmp_path / "2.txt"
    first.write_text("hello", encoding="utf-8")
    second.write_text("world", encoding="utf-8")

    def load_with_prefix(path, prefix=""):
        return f"{prefix}{Path(path).read_text(encoding='utf-8')}"

    assert list(
        iter_files(
            [first, second],
            pattern="*.txt",
            loader=load_with_prefix,
            load_kwargs={"prefix": "say-"},
            n_jobs=2,
        )
    ) == ["say-hello", "say-world"]


def test_save_json_pretty(tmp_path):
    f = tmp_path / "pretty.json"

    data = {"z": 2, "a": 1}
    save_json(data, str(f), save_pretty=True)

    content = f.read_text()

    # pretty option should produce newlines and indentation
    assert "\n" in content
    assert "    " in content or "\t" in content


def test_save_and_load_jsonl_with_bytes(tmp_path):
    f = tmp_path / "data.jsonl"

    data = [{"num": 1, "b": b"bytes"}, {"num": 2, "lst": [1, 2, 3]}]
    save_jsonl(data, str(f))
    loaded = load_jsonl(str(f))

    assert loaded[0]["num"] == 1
    assert loaded[0]["b"] == "bytes"
    assert loaded[1]["num"] == 2
    assert loaded[1]["lst"] == [1, 2, 3]


def test_save_jsonl_accepts_generator(tmp_path):
    f = tmp_path / "data.jsonl"

    save_jsonl(({"i": i} for i in range(3)), str(f))

    assert load_jsonl(str(f)) == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_save_jsonl_writes_json_array_for_json_suffix(tmp_path):
    f = tmp_path / "data.json"

    save_jsonl(({"i": i, "b": b"bytes"} for i in range(3)), str(f))

    content = f.read_text()
    assert content.startswith("[\n")
    assert not content.rstrip().endswith(",")
    assert json.loads(content) == [
        {"i": 0, "b": "bytes"},
        {"i": 1, "b": "bytes"},
        {"i": 2, "b": "bytes"},
    ]


def test_load_jsonl_max_samples(tmp_path):
    f = tmp_path / "data.jsonl"

    data = [{"i": i} for i in range(5)]
    save_jsonl(data, str(f))

    loaded = load_jsonl(str(f), max_samples=2)

    assert len(loaded) == 2
    assert loaded == data[:2]


def test_save_jsonl_rejects_newlines(tmp_path):
    f = tmp_path / "data.jsonl"

    try:
        save_jsonl([{"a": 1}], str(f), indent=2)
        assert False, "Expected ValueError due to newline in JSONL line"
    except ValueError as exc:
        assert "indent" in str(exc)


def test_iter_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"

    f.write_text('{"a": 1}\n\n{"a": 2}\n')

    it = iter_jsonl(str(f))
    assert list(it) == [{"a": 1}, {"a": 2}]

    tqdm_items = list(tqdm(iter_jsonl(str(f))))
    assert tqdm_items == [{"a": 1}, {"a": 2}]


def test_jsonl_helper_direct_usage(tmp_path):
    f = tmp_path / "data.jsonl"

    f.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3}\n')

    records = JsonlHelper(str(f), max_samples=2)
    assert len(records) == 2
    assert records[0] == {"a": 1}
    assert list(records) == [{"a": 1}, {"a": 2}]

    assert records._load_by_offset(records._offsets[0]) == {"a": 1}
    assert records._load_by_offset(records._offsets[:2]) == [
        {"a": 1},
        {"a": 2},
    ]


def test_jsonl_helper_rejects_invalid_max_samples(tmp_path):
    f = tmp_path / "data.jsonl"
    save_jsonl(({"i": i} for i in range(2)), f)

    try:
        JsonlHelper(str(f), max_samples=-1)
        assert False, "Expected ValueError for negative max_samples"
    except ValueError as exc:
        assert "max_samples" in str(exc)

    try:
        JsonlHelper(str(f), max_samples=1.5)
        assert False, "Expected TypeError for non-integer max_samples"
    except TypeError as exc:
        assert "max_samples" in str(exc)

    try:
        JsonlHelper(str(f), max_samples=True)
        assert False, "Expected TypeError for bool max_samples"
    except TypeError as exc:
        assert "max_samples" in str(exc)


def test_jsonl_helper_supports_negative_index_and_slice(tmp_path):
    f = tmp_path / "data.jsonl"

    save_jsonl(({"i": i} for i in range(5)), f)

    records = JsonlHelper(str(f))
    assert records[-1] == {"i": 4}
    assert records[-2] == {"i": 3}
    assert records[:2] == [{"i": 0}, {"i": 1}]
    assert records[1:4] == [{"i": 1}, {"i": 2}, {"i": 3}]
    assert records[-3:] == [{"i": 2}, {"i": 3}, {"i": 4}]
    assert records[::2] == [{"i": 0}, {"i": 2}, {"i": 4}]
    assert records[::-1] == [
        {"i": 4},
        {"i": 3},
        {"i": 2},
        {"i": 1},
        {"i": 0},
    ]

    try:
        _ = records[-6]
        assert False, "Expected IndexError for negative index out of range"
    except IndexError:
        pass


def test_jsonl_helper_slice_respects_max_samples(tmp_path):
    f = tmp_path / "data.jsonl"

    save_jsonl(({"i": i} for i in range(5)), f)

    records = JsonlHelper(str(f), max_samples=3)
    assert len(records) == 3
    assert len(records._offsets) == 3
    assert records[-1] == {"i": 2}
    assert records[:] == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert records[-2:] == [{"i": 1}, {"i": 2}]


def test_jsonl_helper_cache_index_with_max_samples_warns_and_caches_full_index(tmp_path):
    f = tmp_path / "data.jsonl"
    index_file = tmp_path / ".data.jsonl.index"

    save_jsonl(({"i": i} for i in range(5)), f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = JsonlHelper(str(f), max_samples=2, cache_index=True)

    assert len(caught) == 1
    assert "builds and caches the full JSONL offset index" in str(caught[0].message)
    assert len(records) == 2
    assert records[-1] == {"i": 1}

    index_data = json.loads(index_file.read_text())
    assert len(index_data["offsets"]) == 5


def test_jsonl_helper_cache_index(tmp_path):
    f = tmp_path / "data.jsonl"
    index_file = tmp_path / ".data.jsonl.index"

    f.write_text('{"a": 1}\n\n{"a": 2}\n')

    records = JsonlHelper(str(f), cache_index=True)
    assert len(records) == 2
    assert index_file.exists()

    index_data = json.loads(index_file.read_text())
    assert index_data["version"] == 1
    assert index_data["file"]["name"] == "data.jsonl"
    assert index_data["file"]["size"] == f.stat().st_size
    assert index_data["file"]["mtime_ns"] == f.stat().st_mtime_ns
    assert len(index_data["offsets"]) == 2

    # A matching cache should be reused instead of rebuilding the index.
    index_data["offsets"] = [index_data["offsets"][0]]
    index_file.write_text(json.dumps(index_data))
    cached_records = JsonlHelper(str(f), cache_index=True)
    assert len(cached_records) == 1

    # A cache with an unsupported version should be treated as stale.
    index_data["version"] = 0
    index_file.write_text(json.dumps(index_data))
    rebuilt_records = JsonlHelper(str(f), cache_index=True)
    assert len(rebuilt_records) == 2
    rebuilt_index_data = json.loads(index_file.read_text())
    assert rebuilt_index_data["version"] == 1
    assert len(rebuilt_index_data["offsets"]) == 2


def test_jsonl_helper_rebuilds_stale_cache_index(tmp_path):
    f = tmp_path / "data.jsonl"
    index_file = tmp_path / ".data.jsonl.index"

    f.write_text('{"a": 1}\n{"a": 2}\n')
    assert len(JsonlHelper(str(f), cache_index=True)) == 2

    old_index_data = json.loads(index_file.read_text())
    assert len(old_index_data["offsets"]) == 2

    f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    records = JsonlHelper(str(f), cache_index=True)
    assert len(records) == 3
    assert records[2] == {"a": 3}

    new_index_data = json.loads(index_file.read_text())
    assert new_index_data["file"]["size"] == f.stat().st_size
    assert new_index_data["file"]["mtime_ns"] == f.stat().st_mtime_ns
    assert len(new_index_data["offsets"]) == 3


def test_iter_jsonl_max_samples_affects_len_and_iteration(tmp_path):
    f = tmp_path / "data.jsonl"

    f.write_text("""{"a": 1}
{"a": 2}
{"a": 3}
""")

    it = iter_jsonl(str(f), max_samples=2)
    assert list(it) == [{"a": 1}, {"a": 2}]


def test_save_and_load_pickle(tmp_path):
    f = tmp_path / "obj.pkl"

    obj = {"k": (1, 2), "s": "x"}
    save_pickle(obj, str(f))
    loaded = load_pickle(str(f))

    assert loaded == obj


def test_save_and_load_bytes(tmp_path):
    f = tmp_path / "data.bin"

    data = b"\x00\x01\x02\x03\x04"
    save_bytes(data, str(f))
    loaded = load_bytes(str(f))

    assert loaded == data

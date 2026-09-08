# -*- coding: utf-8 -*-
# @Time    : 1/28/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_id_utils.py

import re
from pathlib import Path

import pytest

from common_utils import IdSampleManager, generate_id, index_by_id, list_ids


def test_list_ids_with_tmp_path(tmp_path):
    """
    Directory layout:

    temp_dir/
        sample_001.txt
        sample_002.txt
        ignore.log
        run_A/
        run_B/
    """
    # --- create files ---
    (tmp_path / "sample_001.txt").write_text("data")
    (tmp_path / "sample_002.txt").write_text("data")
    (tmp_path / "ignore.log").write_text("data")

    # --- create directories ---
    (tmp_path / "run_A").mkdir()
    (tmp_path / "run_B").mkdir()

    # --- default behavior (auto include -> files, auto matching -> stem) ---
    ids = list_ids(str(tmp_path))
    assert sorted(ids) == ["ignore", "sample_001", "sample_002"]

    # --- include only .txt files ---
    ids = list_ids(
        str(tmp_path),
        include=r"\.txt$",
    )
    assert sorted(ids) == ["sample_001", "sample_002"]

    # --- extract numeric IDs via regex ---
    ids = list_ids(
        str(tmp_path),
        include=r"\.txt$",
        matching=re.compile(r"_(\d+)\.txt$"),
    )
    assert sorted(ids) == ["001", "002"]

    # --- exclude pattern ---
    ids = list_ids(
        str(tmp_path),
        exclude=r"ignore",
    )
    assert "ignore" not in ids


def test_list_ids_directories_only_with_tmp_path(tmp_path):
    # more dirs than files
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    (tmp_path / "C").mkdir()
    (tmp_path / "file.txt").write_text("x")

    ids = list_ids(str(tmp_path))
    assert sorted(ids) == ["A", "B", "C"]


def test_list_ids_return_filepaths_with_tmp_path(tmp_path):
    (tmp_path / "x_01.txt").write_text("x")
    (tmp_path / "x_02.txt").write_text("x")

    ids, paths = list_ids(
        str(tmp_path),
        matching=r"x_(\d+)",
        return_filepath=True,
    )

    assert sorted(ids) == ["01", "02"]
    assert len(ids) == len(paths)
    assert all(p.endswith(".txt") for p in paths)

    # --- return dict ---
    id2path = list_ids(
        str(tmp_path),
        matching=r"x_(\d+)",
        return_filepath=True,
        return_dict=True,
    )
    print(id2path)

    assert sorted(id2path.keys()) == ["01", "02"]
    assert len(id2path) == len(ids)
    assert all(p.endswith(".txt") for p in id2path.values())


def test_generate_id_random_mode():
    # --- random mode ---
    ids = [generate_id() for _ in range(100)]
    print(f"Random ids {ids}")
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{32}", x) for x in ids)


def test_generate_id_deterministic_mode():
    # --- deterministic mode ---
    a1 = generate_id("hello")
    a2 = generate_id("hello")
    b = generate_id("world")

    print(f"{a1}=={a2} {a1}!={b}")
    assert a1 == a2
    assert a1 != b
    assert re.fullmatch(r"[0-9a-f]{32}", a1)


def test_index_by_id_raises_on_duplicated_keys_by_default():
    entries = [
        {"id": 1, "value": "old"},
        {"uuid": "u-2", "value": "second"},
        {"id": 1, "value": "new"},
    ]

    with pytest.raises(ValueError, match="Duplicated key found: 1"):
        index_by_id(entries)


def test_index_by_id_can_ignore_duplicated_keys_and_keep_last_entry():
    entries = [
        {"id": 1, "value": "old"},
        {"uuid": "u-2", "value": "second"},
        {"id": 1, "value": "new"},
    ]

    index = index_by_id(entries, ignore_duplicates=True)

    assert sorted(index.keys()) == ["1", "u-2"]
    assert index["1"] == {"id": 1, "value": "new"}
    assert index["u-2"] == {"uuid": "u-2", "value": "second"}


def test_index_by_id_supports_arbitrary_entries_with_key_function():
    entries = ["alpha.txt", "beta.txt"]

    index = index_by_id(entries, key=lambda x: Path(x).stem)

    assert index == {
        "alpha": "alpha.txt",
        "beta": "beta.txt",
    }


def test_index_by_id_skips_none_key_values():
    entries = [
        {"id": "keep"},
        {"id": None},
        "skip-me",
    ]

    index = index_by_id(entries, key=lambda x: x.get("id") if isinstance(x, dict) else None)

    assert index == {"keep": {"id": "keep"}}


@pytest.mark.parametrize(
    "entry",
    [
        "plain-string",
        {"value": 123},
    ],
)
def test_index_by_id_raises_when_key_cannot_be_inferred(entry):
    with pytest.raises(ValueError, match="provide `key` explicitly"):
        index_by_id([entry])


def test_id_sample_manager_samples_without_repeating_consumed_ids():
    manager = IdSampleManager(["a", "b", "c", "d", "e", "f"])

    first = manager.sample_record(2, "first", seed=0)
    second = manager.sample_record(2, "second", seed=1)

    assert len(first) == 2
    assert len(second) == 2
    assert set(first).isdisjoint(second)
    assert set(manager.get_selected_ids(consumed_only=True)) == set(first) | set(second)
    assert set(manager.get_remaining_ids()) == {"a", "b", "c", "d", "e", "f"} - set(first) - set(second)
    assert manager.list_records(consumed=True) == ["first", "second"]


def test_id_sample_manager_raises_or_returns_partial_when_pool_is_insufficient():
    manager = IdSampleManager(["a", "b", "c"])
    manager.sample_record(2, "first", seed=0)

    with pytest.raises(ValueError, match="only 1 are available"):
        manager.sample_record(2, "second", seed=1)

    remaining = manager.sample_record(2, "second", seed=1, allow_partial=True)
    assert len(remaining) == 1
    assert remaining == manager.get_record_ids("second")
    assert manager.get_remaining_ids() == []


def test_id_sample_manager_supports_named_set_operations_and_record_queries():
    manager = IdSampleManager(["a", "b", "c", "d", "e"])
    manager.add_record("left", ["a", "b", "c"], consumed=True)
    manager.add_record("right", ["b", "c", "d"], consumed=False)
    original_names = manager.list_records()

    assert manager.union("left", "right") == ["a", "b", "c", "d"]
    assert manager.intersection("left", "right") == ["b", "c"]
    assert manager.difference("left", "right") == ["a"]
    assert manager.complement("left") == ["d", "e"]
    assert manager.list_records() == original_names

    combined = manager.union("left", "right", name="combined")
    assert combined == ["a", "b", "c", "d"]
    assert manager.get_record("combined") == {
        "ids": ["a", "b", "c", "d"],
        "kind": "derived",
        "consumed": False,
        "metadata": {},
    }


def test_id_sample_manager_can_find_records_by_id_efficiently():
    manager = IdSampleManager(["a", "b", "c", "d", "e"])
    manager.add_record("sample_a", ["a", "b"], kind="sample", consumed=True)
    manager.add_record("manual_b", ["b", "c"], kind="manual", consumed=False)
    manager.add_record("derived_b", ["b", "d"], kind="derived", consumed=False)

    assert manager.find_records_by_id("b") == ["sample_a", "manual_b", "derived_b"]
    assert manager.find_records_by_id("b", consumed=True) == ["sample_a"]
    assert manager.find_records_by_id("b", kind="manual") == ["manual_b"]
    assert manager.find_records_by_id("missing") == []


def test_id_sample_manager_can_save_and_load_json_state(tmp_path):
    path = tmp_path / "state.json"

    manager = IdSampleManager(["a", "b", "c", "d"])
    manager.sample_record(2, "seeded", seed=123, metadata={"note": "demo"})
    manager.add_record("manual", ["d"], consumed=False, metadata={"source": "external"})
    manager.save_json(path)

    loaded = IdSampleManager.load_json(path)

    assert loaded.to_dict() == manager.to_dict()
    assert loaded.get_remaining_ids() == manager.get_remaining_ids()
    assert loaded.get_record("manual") == {
        "ids": ["d"],
        "kind": "manual",
        "consumed": False,
        "metadata": {"source": "external"},
    }


def test_id_sample_manager_rejects_duplicate_or_unknown_ids():
    with pytest.raises(ValueError, match="Duplicated ID found in ids"):
        IdSampleManager(["a", "a"])

    manager = IdSampleManager(["a", "b"])
    with pytest.raises(ValueError, match="outside the managed pool"):
        manager.add_record("bad", ["a", "c"])

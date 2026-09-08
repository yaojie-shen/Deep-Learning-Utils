# -*- coding: utf-8 -*-
# @Time    : 3/6/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_env_utils.py

from pathlib import Path

import pytest

from common_utils.env_utils import get_env


def test_env_used_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "line1\r\nline2\rline3\n")

    assert get_env("API_TOKEN", cwd=tmp_path) == "line1"
    assert get_env("API_TOKEN", allow_multiline=True, cwd=tmp_path) == "line1\nline2\nline3\n"


def test_dotfile_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MYSECRET", "env-value")

    (tmp_path / ".MYSECRET").write_text("file\nvalue\r\nline2", encoding="utf-8")

    assert get_env("MYSECRET", allow_multiline=True, cwd=tmp_path) == "file\nvalue\nline2"
    assert get_env("MYSECRET", allow_multiline=False, cwd=tmp_path) == "file"


def test_plain_file_has_highest_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN", "env")

    (tmp_path / ".TOKEN").write_text("dot", encoding="utf-8")
    (tmp_path / "TOKEN").write_text("plain", encoding="utf-8")

    assert get_env("TOKEN", cwd=tmp_path) == "plain"


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NOT_EXISTS_KEY", raising=False)

    with pytest.raises(KeyError):
        get_env("NOT_EXISTS_KEY", cwd=tmp_path)

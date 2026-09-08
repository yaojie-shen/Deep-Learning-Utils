# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_ollama_utils.py

import pytest

ollama = pytest.importorskip("ollama")

from common_utils.inference import ollama_utils
from common_utils.inference.ollama_utils import OllamaModel, ensure_ollama_running


def test_ensure_ollama_running_returns_when_server_is_already_running(monkeypatch):
    monkeypatch.setattr(ollama_utils, "is_ollama_running", lambda: True)
    monkeypatch.setattr(
        ollama_utils,
        "start_ollama_background",
        lambda *args: pytest.fail("server should not be started"),
    )

    ensure_ollama_running(max_retries=1, wait_seconds=0)


def test_ensure_ollama_running_retries_and_raises(monkeypatch):
    starts = []

    class Process:
        def poll(self):
            return 1

    def start(*args):
        starts.append(args)
        return Process()

    monkeypatch.setattr(ollama_utils, "is_ollama_running", lambda: False)
    monkeypatch.setattr(ollama_utils, "start_ollama_background", start)

    with pytest.raises(RuntimeError):
        ensure_ollama_running("--host", "127.0.0.1", max_retries=2, wait_seconds=0)

    assert starts == [("--host", "127.0.0.1"), ("--host", "127.0.0.1")]


def test_ollama_model_chat_calls_ollama_client(monkeypatch):
    calls = []

    def chat(model, messages):
        calls.append((model, messages))
        return {"message": {"content": "2"}}

    monkeypatch.setattr(
        ollama_utils, "ensure_ollama_running", lambda: calls.append("ensure")
    )
    monkeypatch.setattr(
        ollama_utils, "pull", lambda model_name: calls.append(("pull", model_name))
    )
    monkeypatch.setattr(ollama_utils, "chat", chat)

    model = OllamaModel(model_name="test-model")
    response = model.chat(messages=[{"role": "user", "content": "1+1=?"}])

    assert response == {"message": {"content": "2"}}
    assert calls == [
        "ensure",
        ("pull", "test-model"),
        ("test-model", [{"role": "user", "content": "Hello!"}]),
        "ensure",
        ("test-model", [{"role": "user", "content": "1+1=?"}]),
    ]

# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : ollama_utils.py

import socket
import subprocess
import time

from ollama import AsyncClient, ChatResponse, chat, pull


def is_ollama_running(host="127.0.0.1", port=11434) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def start_ollama_background(*args):
    """
    Start `ollama serve` detached from the parent process.
    """
    return subprocess.Popen(
        ["ollama", "serve", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # fully detached (Linux)
    )


def ensure_ollama_running(
    *args,
    max_retries: int = 16,
    wait_seconds: int = 1,
):
    """
    Ensure that Ollama server is running, start it if not.

    Args:
        *args: Additional arguments to pass to `ollama serve`.
        max_retries: Maximum number of retries to check if Ollama is running.
        wait_seconds: Seconds to wait between retries.

    Returns:

    """
    if is_ollama_running():
        return

    last_process = None

    for attempt in range(1, max_retries + 1):
        last_process = start_ollama_background(*args)
        # Give it time to bind the port
        time.sleep(wait_seconds)

        if is_ollama_running():
            print("Ollama started successfully.")
            return

        # If the process died immediately, clean up
        if last_process.poll() is not None:
            print("Ollama process exited early, retrying...")
        else:
            print("Ollama process alive but port not open yet, retrying...")

    raise RuntimeError(f"Failed to start Ollama after {max_retries} attempts.")


class OllamaModel:
    def __init__(self, model_name: str) -> None:
        """1. Run ollama serve in subprocess
        2. Call ensure_ollama_running before each call"""
        self.model_name = model_name

        ensure_ollama_running()
        pull(model_name)

        # Health check
        response: ChatResponse = chat(
            model=self.model_name, messages=[{"role": "user", "content": "Hello!"}]
        )

    def chat(self, *args, **kwargs) -> ChatResponse:
        ensure_ollama_running()
        return chat(model=self.model_name, *args, **kwargs)

    async def async_chat(self, *args, **kwargs):
        return await AsyncClient().chat(model=self.model_name, *args, **kwargs)


# TODO: Process image
# TODO: Test cases

__all__ = [
    "is_ollama_running",
    "start_ollama_background",
    "ensure_ollama_running",
    "OllamaModel",
]

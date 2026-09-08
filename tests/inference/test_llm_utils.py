# -*- coding: utf-8 -*-
# @Time    : 2/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_llm_utils.py

import json

import pytest

from dl_utils import (
    extract_json,
    format_prompt,
    load_prompt,
    load_prompt_schema,
    render_prompt,
)


@pytest.mark.parametrize(
    "llm_output, expected",
    [
        # Plain JSON
        ('{"key": "value"}', {"key": "value"}),
        # JSON in markdown code fences
        ('```json\n{"key": 123}\n```', {"key": 123}),
        ("```json\n[1, 2, 3]\n```", [1, 2, 3]),
        # JSON with text before and after
        ('Here is the data:\n```json\n{"foo": "bar"}\n```\nThank you!', {"foo": "bar"}),
        # JSON array
        ("[1, 2, 3]", [1, 2, 3]),
        # JSON with extra text but valid extraction
        ('Output: {"a": true, "b": false} End', {"a": True, "b": False}),
    ],
)
def test_extract_json_valid(llm_output, expected):
    assert extract_json(llm_output) == expected


def test_extract_json_invalid():
    # Invalid JSON should raise ValueError
    with pytest.raises(ValueError):
        extract_json("No JSON here!")


def test_extract_json_partial_malformed():
    # Malformed JSON inside text, but valid part exists
    llm_output = 'Data: {"valid": 1, "invalid": } end'
    # Should raise ValueError because no fully valid JSON can be parsed
    with pytest.raises(ValueError):
        extract_json(llm_output)


def test_load_prompt_from_path(tmp_path):
    prompt_path = tmp_path / "qa_prompt.txt"
    prompt_path.write_text("Question: {question}", encoding="utf-8")

    assert load_prompt(prompt_path) == "Question: {question}"


def test_load_prompt_missing_path_raises(tmp_path):
    missing_prompt_path = tmp_path / "missing_prompt.txt"

    with pytest.raises(FileNotFoundError):
        load_prompt(str(missing_prompt_path))


def test_load_prompt_from_directory_with_version(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "classify_v1.txt").write_text("Classify: {text}", encoding="utf-8")
    (prompt_dir / "classify_v2.md").write_text(
        "Classify carefully: {text}", encoding="utf-8"
    )

    assert (
        load_prompt(prompt_dir=prompt_dir, prompt_name="classify", version="v2")
        == "Classify carefully: {text}"
    )


def test_load_prompt_from_directory_nested_version(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_version_dir = prompt_dir / "extract"
    prompt_version_dir.mkdir(parents=True)
    (prompt_version_dir / "v1.md").write_text(
        "Extract [field] from [text].", encoding="utf-8"
    )

    assert (
        load_prompt(prompt_dir=prompt_dir, prompt_name="extract", version="v1")
        == "Extract [field] from [text]."
    )


def test_load_prompt_schema_from_directory_with_version(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    schema = {
        "type": "object",
        "required": ["label"],
        "properties": {"label": {"type": "string"}},
    }
    (prompt_dir / "classify_v1.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    assert (
        load_prompt_schema(prompt_dir=prompt_dir, prompt_name="classify", version="v1")
        == schema
    )


def test_load_prompt_schema_from_nested_version(tmp_path):
    prompt_dir = tmp_path / "prompts"
    schema_dir = prompt_dir / "extract"
    schema_dir.mkdir(parents=True)
    schema = {"type": "array", "items": {"type": "string"}}
    (schema_dir / "v2.schema.json").write_text(json.dumps(schema), encoding="utf-8")

    assert (
        load_prompt_schema(prompt_dir=prompt_dir, prompt_name="extract", version="v2")
        == schema
    )


def test_load_prompt_with_schema_from_directory(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    (prompt_dir / "qa_v1.txt").write_text("Question: {question}", encoding="utf-8")
    (prompt_dir / "qa_v1.schema.json").write_text(json.dumps(schema), encoding="utf-8")

    prompt, loaded_schema = load_prompt(
        prompt_dir=prompt_dir,
        prompt_name="qa",
        version="v1",
        load_schema=True,
    )

    assert prompt == "Question: {question}"
    assert loaded_schema == schema


def test_load_prompt_with_schema_from_file_path(tmp_path):
    prompt_path = tmp_path / "qa.md"
    schema_path = tmp_path / "qa.schema.json"
    schema = {"type": "object", "required": ["answer"]}
    prompt_path.write_text("Question: {question}", encoding="utf-8")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    prompt, loaded_schema = load_prompt(prompt_path, load_schema=True)

    assert prompt == "Question: {question}"
    assert loaded_schema == schema


def test_format_prompt_with_python_format_style():
    prompt = "Hello {name}, keep literal braces like {{json}}."

    assert (
        format_prompt(prompt, {"name": "Alice"})
        == "Hello Alice, keep literal braces like {json}."
    )


def test_format_prompt_with_bracket_style():
    prompt = "Hello [name], keep literal brackets like [[text]]."

    assert (
        format_prompt(prompt, {"name": "Alice"}, style="bracket")
        == "Hello Alice, keep literal brackets like [text]."
    )


def test_render_prompt_from_file_path(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Translate {text} to {language}.", encoding="utf-8")

    assert (
        render_prompt(prompt_path, text="hello", language="Chinese")
        == "Translate hello to Chinese."
    )


def test_render_prompt_from_directory(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "summarize.txt").write_text("Summarize: {text}", encoding="utf-8")

    assert (
        render_prompt(prompt_dir=prompt_dir, prompt_name="summarize", text="long text")
        == "Summarize: long text"
    )


def test_render_prompt_with_schema_from_directory(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    schema = {"type": "object", "required": ["summary"]}
    (prompt_dir / "summarize_v1.txt").write_text("Summarize: {text}", encoding="utf-8")
    (prompt_dir / "summarize_v1.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    prompt, loaded_schema = render_prompt(
        prompt_dir=prompt_dir,
        prompt_name="summarize",
        version="v1",
        load_schema=True,
        text="long text",
    )

    assert prompt == "Summarize: long text"
    assert loaded_schema == schema


def test_load_prompt_missing_prompt_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompt(prompt_dir=tmp_path, prompt_name="missing")


def test_format_prompt_missing_placeholder():
    with pytest.raises(KeyError):
        format_prompt("Hello {name}", {})


def test_format_prompt_invalid_style():
    with pytest.raises(ValueError):
        format_prompt("Hello {name}", {"name": "Alice"}, style="unknown")

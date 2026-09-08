import pytest

from dl_utils import parse_steps


def test_parse_separated_string_strips_whitespace_and_deduplicates():
    assert parse_steps("prepare, inference,prepare") == ["prepare", "inference"]


@pytest.mark.parametrize(
    "steps",
    [
        ["prepare", "inference"],
        ("prepare", "inference"),
    ],
)
def test_list_and_tuple_inputs_are_already_separated(steps):
    assert parse_steps(steps) == ["prepare", "inference"]
    assert parse_steps(["prepare,inference"]) == ["prepare,inference"]


def test_custom_separator():
    assert parse_steps("prepare|inference|export", separator="|") == [
        "prepare",
        "inference",
        "export",
    ]


def test_aliases_resolve_to_canonical_names_before_deduplication():
    assert parse_steps(
        ["prep", "prepare", "infer"],
        all_steps=["prepare", "inference"],
        aliases={"prep": "prepare", "infer": "inference"},
    ) == ["prepare", "inference"]


def test_aliases_work_without_all_steps():
    assert parse_steps("prep,infer", aliases={"prep": "prepare", "infer": "inference"}) == [
        "prepare",
        "inference",
    ]


def test_all_expands_in_canonical_order():
    assert parse_steps("all", all_steps=("prepare", "inference", "export")) == [
        "prepare",
        "inference",
        "export",
    ]


def test_all_requires_all_steps_and_must_be_used_alone():
    with pytest.raises(ValueError, match="all_steps must be provided"):
        parse_steps("all")

    with pytest.raises(ValueError, match="must be used on its own"):
        parse_steps("all,prepare", all_steps=["prepare"])


def test_all_steps_enables_unknown_step_validation():
    with pytest.raises(ValueError, match="Unknown step name.*typo"):
        parse_steps("prepare,typo", all_steps=["prepare", "inference"])


def test_invalid_alias_configuration_is_rejected():
    with pytest.raises(ValueError, match="targets unknown canonical step"):
        parse_steps(
            "prepare",
            all_steps=["prepare"],
            aliases={"infer": "inference"},
        )

    with pytest.raises(ValueError, match="conflicts with an existing canonical step"):
        parse_steps(
            "prepare",
            all_steps=["prepare", "inference"],
            aliases={"prepare": "inference"},
        )


@pytest.mark.parametrize("steps", ["", "prepare,,inference", [], ()])
def test_empty_step_names_are_rejected(steps):
    with pytest.raises(ValueError):
        parse_steps(steps)


@pytest.mark.parametrize("steps", [None, {"prepare"}, ["prepare", 1]])
def test_invalid_step_types_are_rejected(steps):
    with pytest.raises(TypeError):
        parse_steps(steps)


def test_reserved_all_cannot_be_canonical_or_alias():
    with pytest.raises(ValueError, match="reserved"):
        parse_steps("all", all_steps=["all"])

    with pytest.raises(ValueError, match="reserved"):
        parse_steps("prepare", aliases={"all": "prepare"})

    with pytest.raises(ValueError, match="reserved"):
        parse_steps("everything", aliases={"everything": "all"})

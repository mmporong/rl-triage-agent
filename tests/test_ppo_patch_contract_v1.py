from __future__ import annotations

import textwrap

import pytest

from evals.ppo_patch_contract_v1 import (
    PatchContractError,
    apply_adapter,
    source_sha256,
    validate_proposal,
    validate_source,
)

VALID_SOURCE = textwrap.dedent(
    '''
    """A restricted adapter."""

    def build_algorithm_kwargs(config):
        """Forward configured PPO epochs."""
        kwargs = dict(config)
        epochs = config.get("num_learning_epochs", 1)
        kwargs["num_learning_epochs"] = epochs
        return kwargs
    '''
).lstrip()


def _source(body: str) -> str:
    return "def build_algorithm_kwargs(config):\n" + textwrap.indent(
        textwrap.dedent(body).strip() + "\n", "    "
    )


def _context() -> dict:
    current = _source(
        """
        kwargs = config.copy()
        return kwargs
        """
    )
    return {
        "source": current,
        "source_sha256": source_sha256(current),
        "facts": {"configured_epochs": 5, "observed_steps": 1},
        "case_id": "fixture-r1",
    }


def _proposal(**changes: object) -> dict:
    context = _context()
    proposal = {
        "action": "patch",
        "source_sha256": context["source_sha256"],
        "replacement_source": VALID_SOURCE,
        "evidence_ids": ["configured_epochs", "observed_steps"],
        "rationale": "설정된 epoch와 관측된 갱신 횟수가 다르다.",
    }
    proposal.update(changes)
    return proposal


def test_source_sha256_normalises_line_endings() -> None:
    assert source_sha256("a\r\nb\r") == source_sha256("a\nb\n")


def test_valid_adapter_copies_config_and_forwards_epochs() -> None:
    config = {"num_learning_epochs": 7, "learning_rate": 0.001}
    original = dict(config)

    validate_source(VALID_SOURCE)
    result = apply_adapter(VALID_SOURCE, config)

    assert result == config
    assert result is not config
    assert config == original


def test_config_copy_and_direct_lookup_are_allowed() -> None:
    source = _source(
        """
        output = config.copy()
        output["num_learning_epochs"] = config["num_learning_epochs"]
        return output
        """
    )
    assert apply_adapter(source, {"num_learning_epochs": 3}) == {
        "num_learning_epochs": 3
    }


@pytest.mark.parametrize(
    "source",
    [
        "import os\n" + VALID_SOURCE,
        _source("for item in config:\n    pass\nreturn dict(config)"),
        _source("items = [item for item in config]\nreturn dict(config)"),
        _source("maker = lambda: dict(config)\nreturn maker"),
        _source("def nested():\n    return dict(config)\nreturn nested"),
        "class Adapter:\n    pass\n" + VALID_SOURCE,
        _source("output = eval('dict(config)')\nreturn output"),
        _source("output = getattr(config, 'copy')()\nreturn output"),
        _source("output = config.pop('num_learning_epochs')\nreturn output"),
        _source("output = config.__class__(config)\nreturn output"),
        _source("output = config['__class__']\nreturn output"),
    ],
)
def test_forbidden_python_surface_is_rejected(source: str) -> None:
    with pytest.raises(PatchContractError):
        validate_source(source)


def test_hardcoded_epoch_is_safe_to_execute_but_not_judged_correct() -> None:
    source = _source(
        """
        output = dict(config)
        output["num_learning_epochs"] = 20
        return output
        """
    )
    validate_source(source)
    assert apply_adapter(source, {"num_learning_epochs": 3}) == {
        "num_learning_epochs": 20
    }


def test_dictionary_literal_can_expose_an_omitted_argument() -> None:
    source = _source(
        """
        output = {"learning_rate": config["learning_rate"]}
        return output
        """
    )
    validate_source(source)
    assert apply_adapter(source, {"num_learning_epochs": 3, "learning_rate": 0.1}) == {
        "learning_rate": 0.1
    }


def test_registered_development_variants_are_executable_not_prejudged() -> None:
    from bench.ppo_patch_cases_v1 import CONFIG, SOURCES

    results = {}
    for name, source in SOURCES.items():
        validate_source(source)
        results[name] = apply_adapter(source, dict(CONFIG))

    assert results["healthy"]["num_learning_epochs"] == 5
    assert "num_learning_epochs" not in results["omitted_argument"]
    assert results["literal_override"]["num_learning_epochs"] == 1
    assert results["late_override"]["num_learning_epochs"] == 1


def test_assignment_to_an_unregistered_field_is_rejected() -> None:
    source = _source(
        """
        output = dict(config)
        output["learning_rate"] = config["learning_rate"]
        return output
        """
    )
    with pytest.raises(PatchContractError):
        validate_source(source)


@pytest.mark.parametrize(
    "source",
    [
        _source("config = dict(config)\nreturn config"),
        _source("dict = config.copy()\nreturn dict"),
        _source("output = dict(config)\noutput = config.copy()\nreturn output"),
        _source("output = dict(config)\nconfig['num_learning_epochs'] = 2\nreturn output"),
        _source("output = dict(config)\nreturn config"),
        VALID_SOURCE + "\nEXTRA = 1\n",
        VALID_SOURCE + "\ndef another(config):\n    return dict(config)\n",
    ],
)
def test_argument_module_and_local_integrity_are_enforced(source: str) -> None:
    with pytest.raises(PatchContractError):
        validate_source(source)


@pytest.mark.parametrize(
    "signature",
    [
        "",
        "config, other",
        "config=None",
        "*args",
        "config, *, flag=False",
        "config: dict",
    ],
)
def test_entrypoint_signature_is_exact(signature: str) -> None:
    source = f"def build_algorithm_kwargs({signature}):\n    return {{}}\n"
    with pytest.raises(PatchContractError):
        validate_source(source)


def test_size_node_and_depth_limits_are_enforced() -> None:
    with pytest.raises(PatchContractError, match="8192"):
        validate_source(VALID_SOURCE + "#" * 8_192)

    many_assignments = "\n".join(f"value_{index} = {index}" for index in range(90))
    with pytest.raises(PatchContractError, match="node limit"):
        validate_source(_source(f"{many_assignments}\noutput = dict(config)\nreturn output"))

    deep_lookup = "config" + "['x']" * 25
    with pytest.raises(PatchContractError, match="depth limit"):
        validate_source(_source(f"value = {deep_lookup}\noutput = dict(config)\nreturn output"))


def test_apply_adapter_rejects_non_plain_config() -> None:
    class CustomDict(dict):
        pass

    with pytest.raises(PatchContractError, match="plain dictionary"):
        apply_adapter(VALID_SOURCE, CustomDict(num_learning_epochs=4))


def test_patch_proposal_is_bound_to_source_and_evidence() -> None:
    context = _context()
    validated = validate_proposal(_proposal(), context)
    assert validated == _proposal()
    assert validated is not _proposal()


def test_hold_proposal_preserves_source_by_omitting_replacement() -> None:
    context = _context()
    proposal = _proposal(action="hold", replacement_source=None)
    assert validate_proposal(proposal, context) == proposal


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"source_sha256": "0" * 64}, "source SHA"),
        ({"evidence_ids": []}, "at least one"),
        ({"evidence_ids": ["configured_epochs", "configured_epochs"]}, "duplicated"),
        ({"evidence_ids": ["missing"]}, "unknown evidence"),
        ({"rationale": ""}, "1 to 1000"),
        ({"rationale": "x" * 1_001}, "1 to 1000"),
        ({"action": "hold"}, "must not include"),
        ({"action": "skip"}, "patch or hold"),
    ],
)
def test_invalid_proposal_bindings_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(PatchContractError, match=message):
        validate_proposal(_proposal(**changes), _context())


def test_patch_must_change_source_and_still_pass_ast_contract() -> None:
    context = _context()
    with pytest.raises(PatchContractError, match="must differ"):
        validate_proposal(
            _proposal(replacement_source=context["source"]),
            context,
        )
    with pytest.raises(PatchContractError):
        validate_proposal(
            _proposal(replacement_source="import os\n" + VALID_SOURCE),
            context,
        )


def test_proposal_keys_are_exact_and_context_sha_is_recomputed() -> None:
    proposal = _proposal()
    proposal["extra"] = True
    with pytest.raises(PatchContractError, match="keys"):
        validate_proposal(proposal, _context())

    context = _context()
    context["source_sha256"] = "f" * 64
    with pytest.raises(PatchContractError, match="context source SHA"):
        validate_proposal(_proposal(), context)

"""CPU quality gate for restricted PPO kwargs adapter candidates.

The gate measures dictionary-forwarding behavior only.  It does not run PPO,
torch, an optimizer, a GPU policy, or a recovery evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evals import ppo_patch_contract_v1 as CONTRACT

SCHEMA: Final = "ppo_patch_gate_quality_v1"
EVALUATOR_SOURCE: Final = "evals/ppo_patch_gate_quality_v1.py"
CONTRACT_SOURCE: Final = "evals/ppo_patch_contract_v1.py"

SCOPE: Final = {
    "algorithm_dictionary_forwarding_only": True,
    "actual_ppo_optimizer_executed": False,
    "gpu_used": False,
    "policy_recovery_measured": False,
    "quality_measurement_is_policy_normalization": False,
    "reference_status_is_development_corpus_label": True,
    "current_source_is_not_external_fault_verification": True,
    "ast_rejected_execution_definition": (
        "source_execution_attempted after validate_source"
    ),
}


class QualityGateError(ValueError):
    """Raised when the quality-gate inputs are not evaluable."""


def _source(body: str, *, module_docstring: bool = False) -> str:
    function = "def build_algorithm_kwargs(config):\n" + textwrap.indent(
        textwrap.dedent(body).strip() + "\n", "    "
    )
    if module_docstring:
        return '"""Restricted forwarding candidate."""\n\n' + function
    return function


HEALTHY_REFERENCE_SOURCE: Final = _source(
    """
    forwarded = dict(config)
    return forwarded
    """
)

CONFIG_CORPUS: Final = (
    {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 0.0005,
        "schedule": "adaptive",
        "gamma": 0.99,
        "entropy_coef": 0.01,
    },
    {
        "num_learning_epochs": 2,
        "num_mini_batches": 8,
        "learning_rate": 0.0001,
        "schedule": "fixed",
        "clip_param": 0.17,
        "desired_kl": 0.008,
    },
    {
        "num_learning_epochs": 7,
        "num_mini_batches": 2,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "value_loss_coef": 0.8,
        "normalize_advantage_per_mini_batch": True,
    },
    {
        "num_mini_batches": 4,
        "learning_rate": 0.0005,
        "schedule": "adaptive",
        "gamma": 0.97,
        "max_grad_norm": 0.75,
    },
)


@dataclass(frozen=True)
class CorpusCase:
    """One explicit ground-truth row in the fixed development corpus."""

    case_id: str
    category: str
    source: str
    expected_admission: bool
    reference_status: str = "fault_confirmed"
    current_source: str = HEALTHY_REFERENCE_SOURCE


DEV_CASES: Final = (
    CorpusCase(
        "correct-dict-copy",
        "correct_copy",
        _source("forwarded = dict(config)\nreturn forwarded"),
        True,
    ),
    CorpusCase(
        "correct-copy-method",
        "correct_copy",
        _source("forwarded = config.copy()\nreturn forwarded"),
        True,
    ),
    CorpusCase(
        "correct-module-docstring",
        "correct_copy",
        _source("result = dict(config)\nreturn result", module_docstring=True),
        True,
    ),
    CorpusCase(
        "correct-function-docstring",
        "correct_copy",
        _source(
            '"""Preserve every configured algorithm kwarg."""\n'
            "algorithm_kwargs = config.copy()\nreturn algorithm_kwargs"
        ),
        True,
    ),
    CorpusCase(
        "wrong-hardcoded-epoch",
        "semantic_epoch_hardcode",
        _source(
            "forwarded = dict(config)\n"
            'forwarded["num_learning_epochs"] = 20\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "wrong-default-insertion",
        "semantic_missing_epoch_default",
        _source(
            "forwarded = dict(config)\n"
            'epoch = config.get("num_learning_epochs", 5)\n'
            'forwarded["num_learning_epochs"] = epoch\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "wrong-dropped-kwargs",
        "semantic_other_kwargs_dropped",
        _source(
            "forwarded = {"
            '"num_mini_batches": config["num_mini_batches"], '
            '"learning_rate": config["learning_rate"], '
            '"schedule": config["schedule"]}\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "wrong-other-kwarg-literal",
        "semantic_other_kwarg_changed",
        _source(
            'epoch = config.get("num_learning_epochs", 1)\n'
            "forwarded = {"
            '"num_learning_epochs": epoch, '
            '"num_mini_batches": config["num_mini_batches"], '
            '"learning_rate": 0.02, '
            '"schedule": config["schedule"]}\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "unsafe-import",
        "security_unsafe_import",
        "import os\n" + _source("forwarded = dict(config)\nreturn forwarded"),
        False,
    ),
    CorpusCase(
        "unsafe-file-call",
        "security_unsafe_file_call",
        _source('handle = open("artifact.txt", "w")\nreturn handle'),
        False,
    ),
    CorpusCase(
        "unsafe-dynamic-call",
        "security_unsafe_dynamic_call",
        _source('forwarded = eval("dict(config)")\nreturn forwarded'),
        False,
    ),
    CorpusCase(
        "unsafe-input-mutation",
        "contract_ineligible_input_mutation",
        _source(
            "forwarded = dict(config)\n"
            'config["num_learning_epochs"] = 5\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "unsafe-other-kwarg-write",
        "contract_ineligible_other_kwarg_write",
        _source(
            "forwarded = dict(config)\n"
            'forwarded["learning_rate"] = config["learning_rate"]\n'
            "return forwarded"
        ),
        False,
    ),
    CorpusCase(
        "healthy-reference-modification",
        "unnecessary_patch",
        _source("unchanged = config.copy()\nreturn unchanged"),
        False,
        reference_status="healthy",
    ),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lf_file_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(
        text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    ).hexdigest()


def _validated_configs(configs: Sequence[dict]) -> tuple[dict, ...]:
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise QualityGateError("configs must be a non-empty sequence of dictionaries")
    if not configs:
        raise QualityGateError("configs must not be empty")
    if any(type(config) is not dict for config in configs):
        raise QualityGateError("each config must be a plain dictionary")
    return tuple(configs)


def evaluate_candidate(
    case: CorpusCase,
    *,
    configs: Sequence[dict] = CONFIG_CORPUS,
) -> dict:
    """Evaluate one candidate after AST validation and healthy-reference gating."""

    checked_configs = _validated_configs(configs)
    candidate_sha = CONTRACT.source_sha256(case.source)
    current_sha = CONTRACT.source_sha256(case.current_source)
    result = {
        "case_id": case.case_id,
        "category": case.category,
        "expected_admission": case.expected_admission,
        "candidate_source_sha256": candidate_sha,
        "current_source_sha256": current_sha,
        "reference_status": case.reference_status,
        "ast_contract_valid": False,
        "source_execution_attempted": False,
        "configs_checked": 0,
        "admitted": False,
        "reason": "ast_contract_rejected",
    }

    try:
        CONTRACT.validate_source(case.source)
    except CONTRACT.PatchContractError:
        return result
    result["ast_contract_valid"] = True

    if case.reference_status == "healthy" and candidate_sha != current_sha:
        result["reason"] = "unnecessary_patch"
        return result
    if case.reference_status not in {"healthy", "fault_confirmed"}:
        result["reason"] = "reference_status_unsupported"
        return result

    result["source_execution_attempted"] = True
    for config in checked_configs:
        try:
            forwarded = CONTRACT.apply_adapter(case.source, dict(config))
        except CONTRACT.PatchContractError:
            result["reason"] = "adapter_execution_rejected"
            return result
        result["configs_checked"] += 1
        if forwarded != config:
            result["reason"] = "forwarding_mismatch"
            return result

    result["admitted"] = True
    result["reason"] = "forwarding_preserved"
    return result


def run_quality_evaluation(
    cases: Iterable[CorpusCase] = DEV_CASES,
    *,
    configs: Sequence[dict] = CONFIG_CORPUS,
) -> tuple[dict, list[dict]]:
    """Run the fixed corpus and report ground-truth classification errors."""

    case_list = list(cases)
    checked_configs = _validated_configs(configs)
    rows = [evaluate_candidate(case, configs=checked_configs) for case in case_list]
    false_accept_ids = [
        row["case_id"]
        for row in rows
        if row["admitted"] and not row["expected_admission"]
    ]
    false_reject_ids = [
        row["case_id"]
        for row in rows
        if not row["admitted"] and row["expected_admission"]
    ]
    security_execution_ids = [
        row["case_id"]
        for row in rows
        if row["category"].startswith("security_")
        and row["source_execution_attempted"]
    ]
    contract_ineligible_execution_ids = [
        row["case_id"]
        for row in rows
        if row["category"].startswith("contract_ineligible_")
        and row["source_execution_attempted"]
    ]
    ast_rejected_execution_ids = (
        security_execution_ids + contract_ineligible_execution_ids
    )
    summary = {
        "schema": SCHEMA,
        "scope": dict(SCOPE),
        "quality_status": (
            "passed"
            if not false_accept_ids
            and not false_reject_ids
            and not ast_rejected_execution_ids
            else "failed"
        ),
        "counts": {
            "total": len(rows),
            "expected_accept": sum(
                1 for case in case_list if case.expected_admission
            ),
            "expected_reject": sum(
                1 for case in case_list if not case.expected_admission
            ),
            "admitted": sum(1 for row in rows if row["admitted"]),
            "rejected": sum(1 for row in rows if not row["admitted"]),
            "false_accept": len(false_accept_ids),
            "false_reject": len(false_reject_ids),
            "ast_rejected_sources_executed": len(ast_rejected_execution_ids),
            "security_sources_executed": len(security_execution_ids),
            "contract_ineligible_sources_executed": len(
                contract_ineligible_execution_ids
            ),
        },
        "false_accept_ids": false_accept_ids,
        "false_reject_ids": false_reject_ids,
        "ast_rejected_execution_ids": ast_rejected_execution_ids,
        "security_execution_ids": security_execution_ids,
        "contract_ineligible_execution_ids": contract_ineligible_execution_ids,
        "unnecessary_patch_blocks": sum(
            1 for row in rows if row["reason"] == "unnecessary_patch"
        ),
    }
    return summary, rows


def build_public_outputs() -> dict[str, object]:
    """Build public, path-free reports without writing them."""

    summary, rows = run_quality_evaluation()
    corpus_binding = [
        {
            "case_id": case.case_id,
            "category": case.category,
            "expected_admission": case.expected_admission,
            "reference_status": case.reference_status,
            "candidate_source_sha256": CONTRACT.source_sha256(case.source),
            "current_source_sha256": CONTRACT.source_sha256(case.current_source),
        }
        for case in DEV_CASES
    ]
    config_binding = [dict(config) for config in CONFIG_CORPUS]
    root = Path(__file__).resolve().parents[1]
    provenance = {
        "schema": SCHEMA,
        "source_bindings": {
            EVALUATOR_SOURCE: _lf_file_sha256(root / EVALUATOR_SOURCE),
            CONTRACT_SOURCE: _lf_file_sha256(root / CONTRACT_SOURCE),
        },
        "corpus_sha256": _sha256_json(corpus_binding),
        "config_corpus_sha256": _sha256_json(config_binding),
        "case_count": len(corpus_binding),
        "config_count": len(config_binding),
        "ground_truth_is_explicit": True,
    }
    return {
        "summary.json": summary,
        "cases.json": {"schema": SCHEMA, "cases": rows},
        "provenance.json": provenance,
    }


def write_public_outputs(out_dir: Path) -> None:
    """Write reports to a new directory, refusing replacement or append."""

    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("--out must name a new directory")
    if not out_dir.parent.is_dir():
        raise FileNotFoundError("--out parent directory must already exist")
    payloads = build_public_outputs()
    out_dir.mkdir()
    for name, payload in payloads.items():
        with (out_dir / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    write_public_outputs(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

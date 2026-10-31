"""검증된 runtime 숫자 fact와 count 일치 파생 fact를 분리해 봉인한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import optimizer_runtime_facts_v1 as FACT

SCHEMA = "diagnostic_fact_ledger_v1"
PROBE = "P_recorded_optimizer_updates"
DERIVED_NAMES = ("step_count_matches_config", "minibatch_count_matches_config")
DERIVED_IDS = tuple(
    f"derived:{role}:{name}" for role in ("case", "reference") for name in DERIVED_NAMES
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda item: (
        _ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), "fact ledger JSON은 객체여야 한다")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), "fact ledger output은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def _checkpoint(path: Path) -> str:
    value = read(path)
    checkpoint = value.get("envelope", {}).get("source_binding", {}).get("checkpoint_sha256")
    require(isinstance(checkpoint, str) and len(checkpoint) == 64,
            "runtime artifact checkpoint binding이 없다")
    return checkpoint


def _checked(path: Path, role: str) -> tuple[dict[str, Any], dict[str, bool | int | None]]:
    path = Path(path).resolve()
    envelope, facts = FACT.checked(path, expected_checkpoint_sha256=_checkpoint(path))
    require(set(facts) == {fact_id for fact_id in FACT.FACT_IDS
                           if fact_id.startswith(f"runtime:{role}:")},
            f"{role} runtime fact role이 다르다")
    return envelope, facts


def _derived(role: str, facts: dict[str, bool | int | None]) -> dict[str, bool | None]:
    prefix = f"runtime:{role}:"
    available = facts[prefix + "available"]
    require(type(available) is bool, "runtime available fact 형식")
    if not available:
        return {f"derived:{role}:{name}": None for name in DERIVED_NAMES}
    step_expected = facts[prefix + "expected_optimizer_steps"]
    step_min, step_max = facts[prefix + "optimizer_steps_min"], facts[prefix + "optimizer_steps_max"]
    mb_expected = facts[prefix + "expected_minibatches"]
    mb_min, mb_max = facts[prefix + "minibatches_min"], facts[prefix + "minibatches_max"]
    require(all(type(item) is int for item in (
        step_expected, step_min, step_max, mb_expected, mb_min, mb_max)),
        "available runtime count fact는 정수여야 한다")
    return {
        f"derived:{role}:step_count_matches_config": step_min == step_max == step_expected,
        f"derived:{role}:minibatch_count_matches_config": mb_min == mb_max == mb_expected,
    }


def validate_proposal_fact_ids(
    facts: dict[str, bool | int | None], fact_ids: object,
) -> list[str]:
    """count 검사 제안이 expected와 observed 양쪽 등록 ID를 인용하는지 검증한다."""
    require(isinstance(fact_ids, list) and bool(fact_ids)
            and all(isinstance(item, str) for item in fact_ids)
            and len(set(fact_ids)) == len(fact_ids), "proposal fact_ids 형식")
    ids = cast(list[str], fact_ids)
    require(set(ids) <= set(facts), "proposal에 등록되지 않은 fact ID가 있다")
    checked_group = False
    groups = {
        "optimizer_steps": ("expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max"),
        "minibatches": ("expected_minibatches", "minibatches_min", "minibatches_max"),
    }
    for role in ("case", "reference"):
        for names in groups.values():
            required = {f"runtime:{role}:{name}" for name in names}
            touched = required & set(ids)
            if touched:
                require(touched == required,
                        f"{role} count 제안은 expected와 observed min/max를 함께 인용해야 한다")
                require(facts[f"runtime:{role}:available"] is True,
                        "unavailable role을 count 검사 근거로 쓸 수 없다")
                checked_group = True
    require(checked_group, "proposal은 하나 이상의 count expected+observed 근거쌍을 포함해야 한다")
    return list(ids)


def build(
    case_runtime_path: Path,
    reference_runtime_path: Path | None,
    *,
    proposal_fact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """raw 16 facts와 별도 derived 4 facts를 현재 원문에서 재계산한다."""
    case_path = Path(case_runtime_path).resolve()
    case_envelope, case_facts = _checked(case_path, "case")
    if reference_runtime_path is None:
        reference_path = None
        reference_envelope = None
        reference_facts = FACT.unavailable_facts("reference")
    else:
        reference_path = Path(reference_runtime_path).resolve()
        reference_envelope, reference_facts = _checked(reference_path, "reference")
    raw_facts = {**case_facts, **reference_facts}
    require(set(raw_facts) == set(FACT.FACT_IDS), "raw runtime fact는 정확히 16개여야 한다")
    derived = {**_derived("case", case_facts), **_derived("reference", reference_facts)}
    require(set(derived) == set(DERIVED_IDS), "derived runtime fact는 정확히 4개여야 한다")
    combined = {**raw_facts, **derived}
    proposal = None
    if proposal_fact_ids is not None:
        proposal = {"probe": PROBE,
                    "fact_ids": validate_proposal_fact_ids(combined, proposal_fact_ids)}
    return {
        "schema": SCHEMA,
        "model_view": {"facts": combined, "count_probe_proposal": proposal},
        "source_binding": {
            "case": deepcopy(case_envelope["source_binding"]),
            "reference": (deepcopy(reference_envelope["source_binding"])
                          if reference_envelope is not None else None),
        },
        "private_source_refs": {
            "case": {"path": str(case_path), "sha256": sha(case_path)},
            "reference": ({"path": str(reference_path), "sha256": sha(reference_path)}
                          if reference_path is not None else None),
        },
        "scope": {"derived_from_verified_counts": True, "bayesian_probability": False,
                  "new_observation": False, "recovery_claim": False},
    }


def checked(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = read(path)
    require(value.get("schema") == SCHEMA, "diagnostic fact ledger schema가 다르다")
    refs = value.get("private_source_refs")
    if not isinstance(refs, dict) or not isinstance(refs.get("case"), dict):
        raise TypeError("diagnostic fact source refs가 없다")
    case_ref = refs["case"]
    reference_ref = refs.get("reference")
    case_path = Path(case_ref["path"]).resolve()
    require(case_path.is_file() and sha(case_path) == case_ref.get("sha256"),
            "case runtime artifact ref가 바뀌었다")
    reference_path = None
    if reference_ref is not None:
        require(isinstance(reference_ref, dict), "reference runtime artifact ref 형식")
        reference_path = Path(reference_ref["path"]).resolve()
        require(reference_path.is_file() and sha(reference_path) == reference_ref.get("sha256"),
                "reference runtime artifact ref가 바뀌었다")
    proposal = value.get("model_view", {}).get("count_probe_proposal")
    proposal_ids = proposal.get("fact_ids") if isinstance(proposal, dict) else None
    expected = build(case_path, reference_path, proposal_fact_ids=proposal_ids)
    require(value == expected, "diagnostic fact ledger가 현재 검증 원문과 다르다")
    return deepcopy(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-runtime", type=Path, required=True)
    parser.add_argument("--reference-runtime", type=Path)
    parser.add_argument("--proposal-fact-id", action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.out, build(args.case_runtime, args.reference_runtime,
                              proposal_fact_ids=args.proposal_fact_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

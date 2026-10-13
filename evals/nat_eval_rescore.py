"""저장된 E4b NAT 결정 기록과 시연 단일 호출 기록을 NAT 1.9 `nat eval --skip_workflow`로 다시 채점한다.

모델·GPU·네트워크를 쓰지 않는다. 데이터셋의 generated_answer는 이미 저장된 출력(NAT 제출 파일, 단일 호출 응답)이고,
evaluator는 기존 함수(nat_decision_call.validate_submission, 버전별 비교 모듈의 single_call_decision,
correction_model_call.review_call_record)를 그대로 부른다. 집계는 버전별 비교 모듈의 agreement·adoption·meets_rule을 쓴다.

대조 범위는 판정 필드다: run별 상태·결정, counts, 단일 호출 유효 여부, 일치 수와 분모, 완료 여부, 채택 판정, v3의
결정 분포·단일 호출 유무, 시연의 진단 결정·교정 검증 결과. 비용·궤적(도구 순서, LLM 호출 수, 토큰, wall 시간, 점검 호출)과
원본 verify_record의 설정·프롬프트·사건 기록 결합은 다시 확인하지 않는다.

결합(packet·기록 파일 SHA, 기록 안의 packet 결합)이 어긋나면 판정으로 바꾸지 않고 ValueError로 멈춘다.
NAT run의 상태 순서(거절 기록 > 제출 재검증 > 미제출)는 evals/nat_decision_call.py call()의 판정 순서와 사전등록 상태 정의를
따른다. 인프라 오류는 저장 출력으로 다시 판정할 수 없어 기록값을 쓴다.

NAT는 `nat.components` 진입점으로 플러그인을 찾는다. 새 진입점을 넣으려면 재설치가 필요하므로, 이 실행기가 evaluator를
GlobalTypeRegistry.push() 안에서만 등록하고 같은 프로세스에서 NAT의 `eval` click 명령을 부른다.

  python evals/nat_eval_rescore.py score --label <이름>     # NAT 없이 같은 채점 함수로 대조
  python evals/nat_eval_rescore.py nat-eval --label <이름>  # nat eval --skip_workflow 실행 뒤 출력 파일로 대조
  python evals/nat_eval_rescore.py publish --label <이름> --tag nat_eval_rescore_v1  # 커밋된 코드로 공개 요약
  python evals/nat_eval_rescore.py replay evals/results/nat_eval_rescore_v1        # 공개 파일만으로 요약 대조
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from functools import cache
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import correction_model_call as CM  # noqa: E402
import e4b_compare as E1  # noqa: E402
import e4b_compare_v2 as E2  # noqa: E402
import e4b_compare_v3 as E3  # noqa: E402
import nat_decision_call as W1  # noqa: E402

PRIVATE = ROOT / "bench/private/nat_eval_rescore"
CONFIG = ROOT / "configs/nat_eval_rescore.yml"
EVALUATOR_TYPE = "rl_triage_contract_rescore"
EVALUATOR_NAME = "contract_rescore"
SCHEMA = "nat_eval_rescore_dataset_v1"
LABEL_RE = r"[a-z0-9][a-z0-9_-]{0,40}"
MODULES = {"v1": E1, "v2": E2, "v3": E3}
SUMMARIES = {"v1": "evals/results/e4b_e2e_reward_s2029_20261013/summary.json",
             "v2": "evals/results/e4b_nat_compare_v2/summary.json",
             "v3": "evals/results/e4b_nat_compare_v3/summary.json"}
V1_TAG = "e2e_reward_s2029_20261013"
DEMOS = {"demo_v1": "e2e_reward_s2029_20261013", "demo_v2": "e2e_reward_s2030_20261013"}
STATUSES = ("valid", "invalid", "infra_error", "missing")
VERDICT_KEYS = ("accepted", "decision", "agrees_with_rule", "reasons", "unmet_checklist")
RESULTS = ROOT / "evals/results"
PUBLIC_SCHEMA = "nat_eval_rescore_public_v1"
PUBLIC_TAG_RE = r"nat_eval_rescore_v[0-9]{1,3}"
BOUND_CODE = ("evals/nat_eval_rescore.py", "configs/nat_eval_rescore.yml", "evals/e4b_compare.py",
              "evals/e4b_compare_v2.py", "evals/e4b_compare_v3.py", "evals/nat_decision_call.py",
              "evals/correction_model_call.py", "src/rl_triage/active_diagnosis.py")
STORED_PUBLIC = (*SUMMARIES.values(), *(f"evals/results/{tag}/demo_result.json" for tag in DEMOS.values()))
NAT_PACKAGES = ("nvidia-nat", "nvidia-nat-core", "nvidia-nat-eval")
PUBLIC_WORDING = ("저장된 E4b NAT 결정 기록과 시연 단일 호출 기록을 NAT eval(--skip_workflow)과 기존 계약 검증기로 다시 채점해 "
                  "공개 요약과 같은 판정을 얻었다는 재현 기록이다. 결정의 정확도나 NAT와 단일 호출 사이의 우열을 보여 주지 않으며 "
                  "모델을 다시 부르지 않는다. 데이터셋과 응답 원문은 비공개이고 여기에는 그 SHA만 남긴다.")
PRIVATE_MARKERS = (re.compile(r"bench/private", re.I), re.compile(r"[A-Za-z]:[\\/]"), re.compile(r"/home/|/mnt/|/Users/"),
                   re.compile(r'"content"'))


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _check_sha(path: Path, expected: str, what: str) -> None:
    if sha_raw(path) != expected:
        raise ValueError(f"{what}가 결합된 SHA와 다르다: {rel(path)}")


def _check_call_packet(kind: str, record: dict, packet_path: Path, what: str) -> None:
    """단일 호출 기록이 그 packet으로 만든 호출인지 확인한다(비교 모듈 bind_inputs와 같은 기준)."""
    if kind == "diagnosis":
        bound = record.get("packet_sha256") == sha_raw(packet_path)
    else:
        bound = record.get("packet_digest") == CM.packet_digest(read(packet_path))
    if not bound:
        raise ValueError(f"{what}: 호출 기록의 packet 결합이 packet 파일과 다르다")


# --- 데이터셋 ---
def version_sets(version: str) -> dict[str, dict]:
    """버전별 세트 → {kinds, inputs}. 입력 결합은 v1은 inputs.json, v2·v3는 프로토콜의 inputs다."""
    if version == "v1":
        inputs = read(E1.folder_for(V1_TAG) / "inputs.json")
        kinds = [kind for kind in E1.KINDS if not inputs[kind].get("not_reached")]
        return {V1_TAG: {"kinds": kinds, "inputs": inputs}}
    module = MODULES[version]
    protocol = read(module.PROTOCOL)
    return {name: {"kinds": spec["kinds"] if isinstance(spec, dict) else list(module.KINDS),
                   "inputs": protocol["inputs"][name]} for name, spec in module.INPUT_SETS.items()}


def run_path(version: str, set_name: str, kind: str, run: int, attempt: int) -> Path:
    return MODULES[version].run_path(set_name, kind, run, attempt)


def _run_files(record_path: Path) -> tuple[Path, Path]:
    return record_path.with_suffix(".submission.json"), record_path.with_suffix(".submission.rejected.json")


def nat_run_entry(version: str, set_name: str, kind: str, run: int, inputs: dict) -> dict:
    """run 번호의 최종 시도를 비교 모듈과 같은 규칙으로 고른다(첫 시도가 인프라 오류일 때만 두 번째 시도)."""
    module = MODULES[version]
    attempts = [run_path(version, set_name, kind, run, a) for a in range(1, module.MAX_ATTEMPTS_PER_RUN + 1)]
    found = [(a, p) for a, p in enumerate(attempts, start=1) if p.is_file()]
    if [a for a, _ in found] != list(range(1, len(found) + 1)):
        raise ValueError(f"{version} {set_name} {kind} run{run}: 시도 번호가 이어지지 않는다")
    meta = {"type": "nat_run", "version": version, "set": set_name, "kind": kind, "run": run,
            "packet": inputs[kind]["packet"], "packet_sha256": inputs[kind]["packet_sha256"]}
    if not found:
        return {**meta, "attempt": None, "record": None, "generated": {"missing": True}}
    attempt, path = found[-1]
    record = read(path)
    if record.get("packet_sha256") != inputs[kind]["packet_sha256"]:
        raise ValueError(f"{path.name}: 기록의 packet SHA가 입력 결합과 다르다")
    submission, rejected = _run_files(path)
    if ("rejection_sha256" in record) != rejected.exists():  # 비교 모듈 verify_record와 같은 기준
        raise ValueError(f"{path.name}: 거절 기록 결합이 맞지 않는다")
    if record.get("submission_sha256") is not None and not submission.exists():
        raise ValueError(f"{path.name}: 결합된 제출 파일이 없다")
    meta.update(attempt=attempt, record=rel(path), record_sha256=sha_raw(path))
    if record["status"] == "infra_error" and attempt < module.MAX_ATTEMPTS_PER_RUN:
        return {**meta, "generated": {"missing": True, "pending": "infra_retry"}}
    if record.get("submission_sha256") is not None:
        # worker 재검증에서 떨어진 제출은 submission_sha256이 비어 있다. 그때는 아래 채점이 다시 무효를 내야 한다.
        _check_sha(submission, record["submission_sha256"], "제출 파일")
    if rejected.exists():
        _check_sha(rejected, record["rejection_sha256"], "거절 기록")
    generated = {"missing": False, "record_status": record["status"],
                 "submission": read(submission) if submission.exists() else None, "rejected": rejected.exists()}
    return {**meta, "generated": generated}


def single_call_entry(version: str, set_name: str, kind: str, inputs: dict) -> dict:
    path, packet = ROOT / inputs[kind]["single_call"], ROOT / inputs[kind]["packet"]
    _check_sha(path, inputs[kind]["single_call_sha256"], "단일 호출 기록")
    _check_sha(packet, inputs[kind]["packet_sha256"], "packet")
    record = read(path)
    _check_call_packet(kind, record, packet, f"{version} {set_name} {kind}")
    return {"type": "single_call", "version": version, "set": set_name, "kind": kind,
            "packet": inputs[kind]["packet"], "packet_sha256": inputs[kind]["packet_sha256"],
            "single_call": inputs[kind]["single_call"], "single_call_sha256": inputs[kind]["single_call_sha256"],
            "generated": {"content": record.get("content")}}


def demo_entries() -> list[dict]:
    """시연 v1·v2가 실제로 쓴 단일 호출 기록. 결과 파일의 response_id와 기록 안의 packet 결합으로 같은 호출인지 확인한다."""
    entries = []
    for demo, tag in DEMOS.items():
        model_dir = ROOT / "bench/private/e2e_reward_demo" / tag / "model"
        result = read(ROOT / "evals/results" / tag / "demo_result.json")
        for kind in ("diagnosis", "correction"):
            call = E1._latest_diagnosis_call(model_dir) if kind == "diagnosis" else model_dir / "correction.call1.json"
            packet = model_dir / f"{kind}.packet.json"
            record = read(call)
            if record.get("response_id") != result[kind]["call"]["response_id"]:
                raise ValueError(f"{demo} {kind}: 결과 파일의 호출과 기록 파일이 다르다")
            _check_call_packet(kind, record, packet, f"{demo} {kind}")
            entries.append({"type": "demo_call", "version": demo, "set": tag, "kind": kind,
                            "packet": rel(packet), "packet_sha256": sha_raw(packet),
                            "single_call": rel(call), "single_call_sha256": sha_raw(call),
                            "generated": {"content": record.get("content")}})
    return entries


def stored_kind(summary: dict, version: str, set_name: str, kind: str) -> dict:
    return summary["kinds"][kind] if version == "v1" else summary["sets"][set_name]["kinds"][kind]


def stored_results() -> tuple[dict, dict]:
    summaries = {version: read(ROOT / path) for version, path in SUMMARIES.items()}
    demos = {tag: read(ROOT / "evals/results" / tag / "demo_result.json") for tag in DEMOS.values()}
    return summaries, demos


def expected_for(entry: dict, summaries: dict, demos: dict):
    """같은 항목의 저장된 결과. 점수는 다시 채점한 값이 이것과 같은지다."""
    if entry["type"] == "demo_call":
        stored = demos[entry["set"]][entry["kind"]]
        if entry["kind"] == "diagnosis":
            return {"decision": stored["decision"]}
        return {"verdict": {k: stored["model_proposal"][k] for k in VERDICT_KEYS}}
    kind_summary = stored_kind(summaries[entry["version"]], entry["version"], entry["set"], entry["kind"])
    if entry["type"] == "single_call":
        return {"valid": kind_summary["single_call_valid"]}
    row = next(r for r in kind_summary["runs"] if r["run"] == entry["run"])
    return {"status": row["status"], "decision": row.get("decision")}


def build_dataset() -> list[dict]:
    """NAT eval json 데이터셋 행. 메타데이터는 문자열 열 하나(item)에 넣어 pandas 형 변환을 피한다."""
    entries = []
    for version, module in MODULES.items():
        for set_name, spec in version_sets(version).items():
            for kind in spec["kinds"]:
                entries += [nat_run_entry(version, set_name, kind, run, spec["inputs"])
                            for run in range(1, module.RUNS_PER_KIND + 1)]
                if spec["inputs"][kind].get("single_call"):
                    entries.append(single_call_entry(version, set_name, kind, spec["inputs"]))
    entries += demo_entries()
    summaries, demos = stored_results()
    rows = []
    for entry in entries:
        ident = "/".join(str(entry[k]) for k in ("version", "set", "kind")) + (
            f"/run{entry['run']}" if entry["type"] == "nat_run" else f"/{entry['type']}")
        meta = {k: v for k, v in entry.items() if k != "generated"}
        rows.append({"id": ident, "question": f"{entry['type']} {ident}: 저장된 출력의 계약 재채점",
                     "answer": dumps(expected_for(entry, summaries, demos)),
                     "generated_answer": dumps(entry["generated"]), "item": dumps({"schema": SCHEMA, **meta})})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("데이터셋 id가 겹친다")
    return rows


# --- 항목 채점(기존 함수만 부른다) ---
def _packet(meta: dict) -> dict:
    path = ROOT / meta["packet"]
    _check_sha(path, meta["packet_sha256"], "packet")
    return read(path)


def _bind_nat_run(meta: dict, generated: dict) -> None:
    """데이터셋의 NAT 출력이 결합된 기록 파일과 같은지 확인한다. 다르면 판정이 아니라 오류다."""
    if meta.get("record") is None:
        if not generated.get("missing"):
            raise ValueError(f"{meta['version']} {meta['set']} {meta['kind']} run{meta['run']}: 기록 없이 출력이 있다")
        return
    path = ROOT / meta["record"]
    _check_sha(path, meta["record_sha256"], "NAT 실행 기록")
    if generated.get("missing"):
        return
    submission, rejected = _run_files(path)
    if (read(path)["status"] != generated["record_status"] or rejected.exists() != generated["rejected"]
            or (read(submission) if submission.exists() else None) != generated["submission"]):
        raise ValueError(f"{meta['record']}: 데이터셋 출력이 기록 파일과 다르다")


def _bind_single_call(meta: dict, generated: dict) -> dict:
    path = ROOT / meta["single_call"]
    _check_sha(path, meta["single_call_sha256"], "단일 호출 기록")
    record = read(path)
    if record.get("content") != generated["content"]:
        raise ValueError(f"{meta['single_call']}: 데이터셋 응답이 기록 파일과 다르다")
    return record


def score_entry(meta: dict, generated: dict) -> dict:
    """저장된 출력 하나를 다시 채점한다. 돌려주는 값의 비교 부분(comparable)이 저장 결과와 같아야 한다."""
    kind = meta["kind"]
    if meta["type"] == "nat_run":
        _bind_nat_run(meta, generated)
        if generated.get("missing"):
            return {"comparable": {"status": "missing", "decision": None}}
        if generated["record_status"] == "infra_error":
            return {"comparable": {"status": "infra_error", "decision": None}, "status_source": "record"}
        if generated["rejected"] or generated["submission"] is None:
            return {"comparable": {"status": "invalid", "decision": None},
                    "status_source": "rejection_record" if generated["rejected"] else "no_submission"}
        packet = _packet(meta)  # 결합 불일치는 판정이 아니라 오류다. 검증기 예외 처리 밖에서 확인한다
        try:
            decision = W1.validate_submission(kind, packet, generated["submission"])
        except (ValueError, TypeError, KeyError) as error:
            return {"comparable": {"status": "invalid", "decision": None}, "status_source": "validator",
                    "validator_error": type(error).__name__}
        return {"comparable": {"status": "valid", "decision": decision}, "status_source": "validator"}
    record = _bind_single_call(meta, generated)
    packet = _packet(meta)
    inputs = {kind: {"packet": meta["packet"], "single_call": meta["single_call"]}}
    if meta["type"] == "single_call":
        single = MODULES[meta["version"]].single_call_decision(kind, inputs)
        return {"comparable": {"valid": single is not None}, "decision": single}
    if kind == "diagnosis":
        return {"comparable": {"decision": E1.single_call_decision(kind, inputs)}}
    review = CM.review_call_record(record, packet)
    return {"comparable": {"verdict": {k: review[k] for k in VERDICT_KEYS}}}


def score_row(row: dict) -> dict:
    meta = json.loads(row["item"])
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{row['id']}: 데이터셋 형식이 다르다")
    result = score_entry(meta, json.loads(row["generated_answer"]))
    expected = json.loads(row["answer"])
    return {"id": row["id"], "score": 1.0 if result["comparable"] == expected else 0.0, "rescored": result,
            "expected": expected}


def score_rows(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: score_row(row) for row in rows}


# --- 집계와 대조 ---
def aggregate(rows: list[dict], scored: dict[str, dict]) -> dict:
    """채점 결과로 저장 요약과 같은 항목을 다시 세고, 비교 모듈의 agreement·adoption·meets_rule을 그대로 쓴다."""
    groups: dict[tuple, dict] = {}
    for row in rows:
        meta = json.loads(row["item"])
        if meta["type"] == "demo_call":
            continue
        group = groups.setdefault((meta["version"], meta["set"], meta["kind"]), {"runs": [], "single": None,
                                                                             "has_single": False})
        result = scored[row["id"]]["rescored"]
        if meta["type"] == "single_call":
            group["has_single"], group["single"] = True, result["decision"]
        else:
            attempt = None if result["comparable"]["status"] == "missing" else meta["attempt"]
            group["runs"].append({"run": meta["run"], "attempt": attempt, **result["comparable"]})
    out: dict = {}
    for (version, set_name, kind), group in sorted(groups.items()):
        module, runs = MODULES[version], sorted(group["runs"], key=lambda r: r["run"])
        valid = [r for r in runs if r["status"] == "valid"]
        counts = {status: sum(1 for r in runs if r["status"] == status) for status in STATUSES}
        single = group["single"]
        matches = [module.agreement(kind, r["decision"], single) for r in valid]
        summary = {"counts": counts, "non_infra_runs": counts["valid"] + counts["invalid"],
                   "single_call_valid": single is not None,
                   "agreement_with_single_call": None if single is None or not matches
                   else {key: sum(1 for m in matches if m[key]) for key in matches[0]},
                   "agreement_denominator": len(valid) if single is not None else 0,
                   "complete": counts["missing"] == 0,
                   "runs": [{"run": r["run"], "attempt": r["attempt"], "status": r["status"], "decision": r["decision"]}
                            for r in runs]}
        if version == "v3":
            labels = [r["decision"]["action" if kind == "diagnosis" else "decision"] for r in valid]
            summary["single_call_available"] = group["has_single"]
            summary["decisions"] = {label: labels.count(label) for label in sorted(set(labels))}
        out.setdefault(version, {}).setdefault(set_name, {})[kind] = summary
    adoption = {"v1": E1.adoption(out["v1"][V1_TAG])["decision"]}
    meets = {}
    for version in ("v2", "v3"):
        module = MODULES[version]
        complete = all(s["complete"] for kinds in out[version].values() for s in kinds.values())
        adoption[version] = None if not complete else (
            "adopt" if module.meets_rule(out[version][module.ADOPTION_SET]) else "do_not_adopt")
        meets[version] = {name: module.meets_rule(kinds) if complete else None for name, kinds in out[version].items()}
    return {"sets": out, "adoption": adoption, "meets_numeric_rule": meets}


COMPARED = ("counts", "non_infra_runs", "single_call_valid", "agreement_with_single_call", "agreement_denominator",
            "complete")
V3_COMPARED = ("single_call_available", "decisions")


def _stored_kinds(summary: dict, version: str) -> dict[str, dict]:
    if version == "v1":  # 재채점은 not_reached 종류를 만들지 않는다
        return {V1_TAG: {k: v for k, v in summary["kinds"].items() if v.get("status") != "not_reached"}}
    return {name: entry["kinds"] for name, entry in summary["sets"].items()}


def compare(aggregated: dict, scored: dict[str, dict], summaries: dict | None = None) -> dict:
    """다시 센 판정 필드와 저장된 summary.json·demo_result.json의 차이. 비어 있어야 통과다."""
    summaries = summaries or stored_results()[0]
    differences = []
    for version, sets in aggregated["sets"].items():
        stored_sets = _stored_kinds(summaries[version], version)
        if set(sets) != set(stored_sets):
            differences.append(f"{version}/set coverage")
        for set_name in set(sets) & set(stored_sets):
            mine_kinds, stored_kinds = sets[set_name], stored_sets[set_name]
            if set(mine_kinds) != set(stored_kinds):
                differences.append(f"{version}/{set_name}/kind coverage")
            for kind in set(mine_kinds) & set(stored_kinds):
                mine, stored = mine_kinds[kind], stored_kinds[kind]
                keys = COMPARED + (V3_COMPARED if version == "v3" else ())
                differences += [f"{version}/{set_name}/{kind}/{key}" for key in keys if mine[key] != stored.get(key)]
                stored_runs = [{k: r.get(k) for k in ("run", "attempt", "status", "decision")} for r in stored["runs"]]
                if mine["runs"] != stored_runs:
                    differences.append(f"{version}/{set_name}/{kind}/runs")
    for version, decision in aggregated["adoption"].items():
        if decision != summaries[version]["adoption"]["decision"]:
            differences.append(f"{version}/adoption")
    for version, sets in aggregated["meets_numeric_rule"].items():
        differences += [f"{version}/{name}/meets_numeric_rule" for name, value in sets.items()
                        if value != summaries[version]["sets"].get(name, {}).get("meets_numeric_rule")]
    mismatched = sorted(ident for ident, item in scored.items() if item["score"] != 1.0)
    nat_runs = [item for item in scored.values() if "status" in item["expected"]]
    return {"differences": sorted(differences), "mismatched_items": mismatched, "items": len(scored),
            "nat_run_items": len(nat_runs),
            "infra_error_items_taken_from_record": sum(1 for item in nat_runs
                                                         if item["rescored"].get("status_source") == "record"),
            "demo_items": sum(1 for ident in scored if ident.startswith("demo_")),
            "pass": not differences and not mismatched}


# --- NAT 등록과 실행 ---
@cache
def _evaluator_parts():
    """evaluator 설정 형식과 build 함수. 정의는 한 번만 하고, 등록은 nat_evaluator_registered()가 매번 한다."""
    from nat.builder.evaluator import EvaluatorInfo
    from nat.data_models.evaluator import EvaluatorBaseConfig
    from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
    from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

    class ContractRescoreConfig(EvaluatorBaseConfig, name=EVALUATOR_TYPE):
        """저장된 E4b·시연 출력을 기존 계약 검증기로 다시 채점한다. LLM을 쓰지 않는다."""

    class ContractRescoreEvaluator(BaseEvaluator):
        async def evaluate_item(self, item):
            row = {"id": item.id, "item": item.full_dataset_entry["item"], "answer": item.expected_output_obj,
                   "generated_answer": item.output_obj}
            result = score_row(row)
            return EvalOutputItem(id=item.id, score=result["score"],
                                  reasoning={"rescored": result["rescored"], "expected": result["expected"]})

    async def build(config: ContractRescoreConfig, builder):  # noqa: ARG001
        yield EvaluatorInfo(config=config, evaluate_fn=ContractRescoreEvaluator(max_concurrency=4).evaluate,
                            description="E4b·시연 저장 출력의 결정적 계약 재채점")

    return ContractRescoreConfig, build


def _refresh_config_schema() -> None:
    """레지스트리 상태대로 NAT Config 검증 스키마를 다시 만든다.

    NAT 1.9.0에서는 등록 변경 훅이 돈 뒤에도 Config.model_validate가 새 evaluator 태그를 거부했다. EvalConfig 주석은
    갱신됐는데 Config는 다시 만들어지지 않았다(이 저장소에서 재현, NAT 쪽 원인은 확인하지 않음). 그래서 직접 다시 만든다.
    """
    from nat.data_models.config import Config
    Config.rebuild_annotations()
    Config.model_rebuild(force=True)


@contextmanager
def nat_evaluator_registered():
    """evaluator를 GlobalTypeRegistry.push() 안에서만 등록한다. 나오면 레지스트리와 Config 스키마가 원래대로 돌아간다.

    push 안에서 플러그인이 처음 등록되면 나올 때 함께 사라지므로, 플러그인 탐색과 NAT eval 모듈 import를 먼저 끝낸다.
    """
    from nat.cli.register_workflow import register_evaluator
    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.runtime.loader import PluginTypes, discover_and_register_plugins
    import nat.plugins.eval.cli.evaluate  # noqa: F401  (평가 런타임 모듈을 push 밖에서 먼저 불러 둔다)

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    config_type, build = _evaluator_parts()
    try:
        with GlobalTypeRegistry.push():
            register_evaluator(config_type=config_type)(build)
            _refresh_config_schema()
            yield
    finally:
        _refresh_config_schema()


def run_nat_eval(dataset: Path, output_dir: Path, config: Path = CONFIG) -> Path:
    """NAT의 eval click 명령을 --skip_workflow로 부른다. 출력 파일 경로를 돌려준다."""
    from nat.plugins.eval.cli.evaluate import eval_command
    with nat_evaluator_registered():
        eval_command.main(["--config_file", str(config), "--dataset", str(dataset), "--skip_workflow",
                           "--override", "eval.general.output.dir", str(output_dir)], standalone_mode=False)
    output = output_dir / f"{EVALUATOR_NAME}_output.json"
    if not output.is_file():
        raise ValueError("NAT eval이 evaluator 출력 파일을 만들지 않았다")
    return output


def scored_from_nat_output(output: Path) -> dict[str, dict]:
    """NAT 출력 파일 → score_rows와 같은 모양. NAT가 항목 예외를 0점으로 바꾼 경우는 error로 남긴다."""
    data = read(output)
    scored = {}
    for item in data["eval_output_items"]:
        reasoning = item.get("reasoning") or {}
        entry = {"id": item["id"], "score": item["score"], "rescored": reasoning.get("rescored"),
                 "expected": reasoning.get("expected") or {}}
        if item.get("error") or "error" in reasoning:
            entry["error"] = item.get("error") or reasoning.get("error")
        scored[item["id"]] = entry
    return scored


def require_no_evaluator_errors(scored: dict[str, dict]) -> None:
    """NAT BaseEvaluator는 항목 예외를 0점으로 바꾼다. 결합 오류가 판정처럼 보이지 않게 집계 전에 거부한다."""
    errors = sorted(ident for ident, item in scored.items() if item.get("error"))
    if errors:
        raise ValueError(f"evaluator 오류 항목 {len(errors)}개: {errors[:5]}")


# --- 공개 요약 ---
def public_view(summaries: dict, demos: dict) -> dict:
    """공개 summary.json·demo_result.json에서 다시 읽을 수 있는 판정 필드. 비공개 경로·응답 원문이 없다."""
    view = {}
    for version in MODULES:
        kinds = _stored_kinds(summaries[version], version)
        view[version] = {"adoption": summaries[version]["adoption"]["decision"],
                         "sets": {name: {kind: {key: entry.get(key) for key in COMPARED if key != "complete"}
                                         for kind, entry in kinds[name].items()} for name in kinds}}
    view["demos"] = {tag: {"diagnosis": {k: demos[tag]["diagnosis"]["decision"][k] for k in ("action", "probe")},
                           "correction": {k: demos[tag]["correction"]["model_proposal"][k] for k in VERDICT_KEYS}}
                     for tag in DEMOS.values()}
    return view


def _check_public(value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False)
    found = [pattern.pattern for pattern in PRIVATE_MARKERS if pattern.search(text)]
    if found:
        raise ValueError(f"공개 요약에 비공개 표식이 있다: {found}")


def public_summary(result: dict, rows: list[dict], dataset: Path, commit: str) -> dict:
    """통과한 NAT eval 재채점 결과로 공개 요약을 만든다. 통과하지 않았으면 공개하지 않는다."""
    if not result.get("pass") or result.get("path") != "nat_eval_skip_workflow":
        raise ValueError("NAT eval 재채점이 저장 요약과 같지 않아 공개하지 않는다")
    summaries, demos = stored_results()
    summary = {"schema": PUBLIC_SCHEMA, "source_commit_sha": commit,
               "code_sha256_lf": {p: sha_lf(ROOT / p) for p in sorted(BOUND_CODE)},
               "stored_results_sha256_lf": {p: sha_lf(ROOT / p) for p in sorted(STORED_PUBLIC)},
               "nat_versions": {name: package_version(name) for name in NAT_PACKAGES},
               "execution": "NAT eval click 명령을 --skip_workflow로 같은 프로세스에서 호출(evaluator는 push 안에서만 등록)",
               "evaluator_type": EVALUATOR_TYPE, "model_calls": 0, "dataset_sha256": sha_raw(dataset),
               "items": result["items"], "item_types": dict(sorted(Counter(json.loads(r["item"])["type"] for r in rows).items())),
               "average_score": result["average_score"], "differences": result["differences"],
               "mismatched_items": result["mismatched_items"],
               "infra_error_items_taken_from_record": result["infra_error_items_taken_from_record"],
               "reproduced": public_view(summaries, demos), "pass": True, "wording": PUBLIC_WORDING}
    _check_public(summary)
    return summary


def write_public(folder: Path, summary: dict) -> None:
    raw = (json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    manifest = {"schema": PUBLIC_SCHEMA, "files": {"summary.json": hashlib.sha256(raw).hexdigest()}}
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "summary.json").open("xb") as stream:
        stream.write(raw)
    with (folder / "artifact_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")


def replay_public(folder: Path) -> dict:
    """공개 파일만으로 요약을 대조한다: 묶음 해시, 저장 결과 SHA, 저장 결과에서 다시 읽은 판정 필드. 비공개 기록은 쓰지 않는다."""
    folder = Path(folder)
    manifest = read(folder / "artifact_manifest.json")
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if (manifest.get("schema") != PUBLIC_SCHEMA or files != {"summary.json", "artifact_manifest.json"}
            or manifest.get("files") != {"summary.json": sha_raw(folder / "summary.json")}):
        raise ValueError("공개 재채점 묶음의 파일 결합이 다르다")
    summary = read(folder / "summary.json")
    if (summary.get("schema") != PUBLIC_SCHEMA or summary.get("pass") is not True or summary.get("differences")
            or summary.get("mismatched_items") or summary.get("model_calls") != 0):
        raise ValueError("공개 재채점 요약이 통과 기록이 아니다")
    if summary.get("stored_results_sha256_lf") != {p: sha_lf(ROOT / p) for p in sorted(STORED_PUBLIC)}:
        raise ValueError("재채점 뒤 공개 저장 결과가 바뀌었다")
    if summary.get("reproduced") != public_view(*stored_results()):
        raise ValueError("공개 요약의 판정 필드가 공개 저장 결과와 다르다")
    _check_public(summary)
    return {"items": summary["items"], "item_types": summary["item_types"], "average_score": summary["average_score"],
            "nat_versions": summary["nat_versions"], "dataset_sha256": summary["dataset_sha256"],
            "adoption": {v: summary["reproduced"][v]["adoption"] for v in MODULES}, "source_commit_sha": summary["source_commit_sha"]}


def command_publish(label: str, tag: str, verify_committed=None, results: Path = RESULTS) -> dict:
    """커밋된 코드와 공개 저장 결과로만 공개 요약을 만든다. 같은 tag 폴더가 있으면 덮어쓰지 않는다."""
    if not isinstance(tag, str) or not re.fullmatch(PUBLIC_TAG_RE, tag):
        raise ValueError("tag는 nat_eval_rescore_v<번호>")
    folder = results / tag
    if folder.exists():
        raise FileExistsError(f"{tag} 공개 폴더가 이미 있다")
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    try:
        commit = verify_committed(ROOT, set(BOUND_CODE) | set(STORED_PUBLIC))
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 공개 저장 결과가 커밋되지 않았다") from error
    result = command_nat_eval(label)
    rows = read(label_dir(label) / "dataset.json")
    summary = public_summary(result, rows, label_dir(label) / "dataset.json", commit)
    write_public(folder, summary)
    replay_public(folder)
    return {"status": "published", "folder": rel(folder), "items": summary["items"], "pass": True}


def label_dir(label: str) -> Path:
    if not isinstance(label, str) or not re.fullmatch(LABEL_RE, label):
        raise ValueError("label은 소문자·숫자·_- 41자 이내")
    return PRIVATE / label


def _prepare(label: str) -> tuple[Path, list[dict], Path]:
    folder = label_dir(label)
    folder.mkdir(parents=True, exist_ok=False)
    rows = build_dataset()
    dataset = folder / "dataset.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return folder, rows, dataset


def command_score(label: str) -> dict:
    folder, rows, _ = _prepare(label)
    scored = score_rows(rows)
    result = {"path": "score_rows", **compare(aggregate(rows, scored), scored)}
    (folder / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def command_nat_eval(label: str) -> dict:
    folder, rows, dataset = _prepare(label)
    output = run_nat_eval(dataset, folder / "nat_output")
    scored = scored_from_nat_output(output)
    if set(scored) != {row["id"] for row in rows}:
        raise ValueError("NAT 출력 항목이 데이터셋과 다르다")
    require_no_evaluator_errors(scored)
    result = {"path": "nat_eval_skip_workflow", "evaluator_output": rel(output),
              "average_score": read(output)["average_score"], **compare(aggregate(rows, scored), scored)}
    (folder / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("score", "nat-eval", "publish", "replay"))
    parser.add_argument("folder", nargs="?", type=Path, help="replay할 공개 폴더")
    parser.add_argument("--label")
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    if args.command != "replay" and not args.label:
        parser.error(f"{args.command}에는 --label이 필요하다")
    if (args.command == "replay") != (args.folder is not None):
        parser.error("공개 폴더는 replay에만, replay에는 반드시 준다")
    commands = {"score": lambda: command_score(args.label), "nat-eval": lambda: command_nat_eval(args.label),
                "publish": lambda: command_publish(args.label, args.tag),
                "replay": lambda: {**replay_public(args.folder), "pass": True}}
    try:
        result = commands[args.command]()
    except (ValueError, KeyError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

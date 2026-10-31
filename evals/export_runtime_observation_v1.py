"""runtime observation 개발 비교를 경로 안전한 공개 결과로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import immutable_git_proof_cache_v1 as GIT_CACHE
import runtime_observation_continuation_v1 as CONT
import runtime_observation_development_v1 as DEV
import runtime_observation_recovery_v1 as R
import runtime_observation_view_v1 as RV

SCHEMA = "runtime_observation_development_export_v1"
PROVENANCE_SCHEMA = "runtime_observation_development_provenance_v1"
PRIVATE_SCHEMA = "runtime_observation_development_export_metadata_v1"
CONDITIONS = ("telemetry_only", "telemetry_and_runtime")
EXPECTED_CHOICES = {
    "telemetry_only": {
        "probe": "P_episode",
        "fact_ids": ["telemetry:case:Episode_Termination/time_out",
                     "telemetry:reference:Episode_Termination/time_out"],
    },
    "telemetry_and_runtime": {
        "probe": "P_optimizer_updates",
        "fact_ids": ["runtime:case:updates_observed",
                     "runtime:case:expected_optimizer_steps",
                     "runtime:case:expected_minibatches"],
    },
}
EXPECTED_RUNTIME_FACTS = {
    "runtime:case:available": True,
    "runtime:case:updates_observed": 3,
    "runtime:case:expected_optimizer_steps": 20,
    "runtime:case:optimizer_steps_min": 4,
    "runtime:case:optimizer_steps_max": 4,
    "runtime:case:expected_minibatches": 20,
    "runtime:case:minibatches_min": 4,
    "runtime:case:minibatches_max": 4,
    "runtime:reference:available": False,
    "runtime:reference:updates_observed": None,
    "runtime:reference:expected_optimizer_steps": None,
    "runtime:reference:optimizer_steps_min": None,
    "runtime:reference:optimizer_steps_max": None,
    "runtime:reference:expected_minibatches": None,
    "runtime:reference:minibatches_min": None,
    "runtime:reference:minibatches_max": None,
}
WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s\"']+")
PRIVATE_MARKERS = ("bench/private", "bench\\private", "/home/", "\\users\\")
SECRET = re.compile(r"(?i)(?:api[_-]?key|authorization|bearer|token)\s*[:=]\s*[\"']?[A-Za-z0-9_-]{16,}")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _number(value: object, name: str, *, integer: bool = False) -> int | float:
    if integer:
        if type(value) is not int:
            raise TypeError(f"{name} 숫자 형식이 다르다")
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} 숫자 형식이 다르다")
    number = cast(int | float, value)
    require(math.isfinite(number) and number >= 0, f"{name}은 유한한 0 이상 수여야 한다")
    return number


def _safe_text(raw: bytes, name: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"공개 파일은 UTF-8이어야 한다: {name}") from error
    lowered = text.lower()
    require(WINDOWS_PATH.search(text) is None, f"공개 파일에 Windows 절대경로가 있다: {name}")
    require(not any(marker in lowered for marker in PRIVATE_MARKERS),
            f"공개 파일에 private 경로가 있다: {name}")
    require(SECRET.search(text) is None, f"공개 파일에 credential 패턴이 있다: {name}")


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"원문 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def _logical_ref(path: str, digest: str) -> dict[str, str]:
    require("\\" not in path and not Path(path).is_absolute() and ".." not in Path(path).parts,
            "공개 ref는 논리 상대경로여야 한다")
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "공개 ref SHA 형식")
    return {"path": path, "sha256": digest}


def _registered_protocol_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"bench/protocols/{Path(path).name}"


def _api_sources(nim_dir: Path, condition: str) -> dict[str, Path]:
    folders = {
        "request": nim_dir / "requests",
        "response": nim_dir / "responses",
        "dispatch": nim_dir / "dispatches",
        "response_sha": nim_dir / "response-sha",
    }
    found: dict[str, Path] = {}
    for kind, folder in folders.items():
        matches = list(folder.glob(f"{condition}-*.{'sha256' if kind == 'response_sha' else 'json'}"))
        require(len(matches) == 1, f"{condition} {kind} 원문은 정확히 하나여야 한다")
        found[kind] = matches[0].resolve()
    return found


def _validate_api(
    condition: str, condition_result: dict[str, Any], sources: dict[str, Path],
) -> dict[str, Any]:
    request, response = read(sources["request"]), read(sources["response"])
    dispatch, response_sha = read(sources["dispatch"]), read(sources["response_sha"])
    require(request.get("schema") == "lineage_nim_transport_request_v1"
            and request.get("slot_id") == condition and request.get("ordinal") == 1,
            f"{condition} request schema/slot/ordinal이 다르다")
    request_id = request.get("request_id")
    require(isinstance(request_id, str) and request_id.startswith(condition + "-1-"),
            f"{condition} request ID가 다르다")
    require(dispatch.get("schema") == "lineage_nim_transport_dispatch_v1"
            and dispatch.get("request_id") == request_id
            and dispatch.get("request_sha256") == sha(sources["request"]),
            f"{condition} dispatch/request 결합이 다르다")
    require(response.get("schema") == "lineage_nim_transport_response_v1"
            and response.get("request_id") == request_id
            and response.get("request_sha256") == sha(sources["request"])
            and response.get("dispatch_sha256") == sha(sources["dispatch"]),
            f"{condition} response/request/dispatch 결합이 다르다")
    require(response_sha == {"request_id": request_id,
                             "response_sha256": sha(sources["response"])},
            f"{condition} response SHA sidecar가 다르다")
    dialogue = condition_result.get("dialogue_input", {})
    choice = condition_result.get("choice", {})
    require(request.get("messages") == dialogue.get("messages")
            and request.get("context_digest") == choice.get("context_digest")
            and request.get("view_digest") == dialogue.get("view_digest"),
            f"{condition} 안전 model message/view 결합이 다르다")
    _safe_text(sources["request"].read_bytes(), f"{condition} request")
    raw_result = response.get("result")
    require(isinstance(raw_result, dict), f"{condition} response result가 없다")
    raw_result = cast(dict[str, Any], raw_result)
    calls = condition_result.get("engine", {}).get("calls", [])
    require(len(calls) == 1 and calls[0].get("transport") == "completed"
            and calls[0].get("result") == raw_result
            and calls[0].get("messages_digest") == dialogue.get("messages_digest"),
            f"{condition} response와 실행 call 원문이 다르다")
    expected = EXPECTED_CHOICES.get(condition)
    expected = cast(dict[str, Any], expected)
    require(expected is not None and choice.get("model_status") == "valid"
            and choice.get("action") == "continue" and choice.get("probe") == expected["probe"]
            and choice.get("fact_ids") == expected["fact_ids"],
            f"{condition} 선택 또는 인용 fact가 등록 개발 결과와 다르다")
    parsed = json.loads(raw_result.get("content", ""))
    require(parsed == {"action": "continue", "probe": expected["probe"],
                       "fact_ids": expected["fact_ids"]},
            f"{condition} 모델 raw content와 선택이 다르다")
    require(raw_result.get("finish_reason") == "stop" and raw_result.get("reasoning_content") is None,
            f"{condition} response 종료 계약이 다르다")
    input_tokens = _number(raw_result.get("input_tokens"), "input tokens", integer=True)
    output_tokens = _number(raw_result.get("output_tokens"), "output tokens", integer=True)
    total_tokens = _number(raw_result.get("total_tokens"), "total tokens", integer=True)
    wall_s = _number(raw_result.get("wall_s"), "API wall_s")
    require(total_tokens == input_tokens + output_tokens, f"{condition} token 합계가 다르다")
    view = json.loads(request["messages"][1]["content"])
    facts = view.get("evidence_facts", {})
    runtime = {key: value for key, value in facts.items() if key.startswith("runtime:")}
    if condition == "telemetry_only":
        require(runtime == {}, "telemetry-only request에 runtime fact가 있다")
    else:
        require(runtime == EXPECTED_RUNTIME_FACTS, "runtime request의 16개 fact 값이 다르다")
        require(not any(key in choice["fact_ids"] for key in (
            "runtime:case:optimizer_steps_min", "runtime:case:optimizer_steps_max",
            "runtime:case:minibatches_min", "runtime:case:minibatches_max")),
            "모델이 actual observed 4 fact를 인용했다는 잘못된 기록")
    return {
        "request_id": request_id, "probe": choice["probe"],
        "cited_fact_ids": deepcopy(choice["fact_ids"]),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "api_wall_s": wall_s,
        "provider_model": raw_result.get("provider_model"),
    }


def export_development(
    execution_dir: Path, nim_dir: Path, preregistration: Path, inputs_path: Path,
    *, out: Path, private_out: Path,
) -> dict[str, Any]:
    """완료된 seed 2307 개발 비교 하나를 fresh 공개/비공개 폴더로 내보낸다."""
    execution_dir, nim_dir = Path(execution_dir).resolve(), Path(nim_dir).resolve()
    preregistration, inputs_path = Path(preregistration).resolve(), Path(inputs_path).resolve()
    out, private_out = Path(out).resolve(), Path(private_out).resolve()
    require(not out.exists() and not private_out.exists(), "export 출력은 fresh 경로여야 한다")
    inputs = DEV.checked_inputs(inputs_path)
    prereg = read(preregistration)
    result_path = execution_dir / "result.json"
    result = read(result_path)
    require(prereg.get("schema") == DEV.SCHEMA
            and prereg.get("purpose") == "posthoc_same_input_next_probe_development_comparison"
            and tuple(prereg.get("conditions", [])) == CONDITIONS
            and tuple(prereg.get("order", [])) == CONDITIONS,
            "개발 사전등록 schema/purpose/조건/순서가 다르다")
    require(prereg.get("input_sha256") == sha(inputs_path) == result.get("input_sha256")
            and result.get("protocol_sha256") == sha(preregistration),
            "새 입력·사전등록·실행 결과 SHA 결합이 다르다")
    require(result.get("schema") == "runtime_observation_development_result_v1"
            and result.get("status") == "complete" and result.get("runtime_observation_used") is True
            and result.get("changed_next_probe") is True,
            "완료된 development 비교 결과가 아니다")
    expected_scope = {"approval": False, "gpu_execution": False, "new_seed": False,
                      "old_measurement_overwrite": False, "posthoc_design": True}
    require(result.get("scope") == prereg.get("scope") == expected_scope,
            "development scope가 다르다")
    session = read(nim_dir / "session.json")
    anchor = session.get("registration_anchor")
    require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
            "source commit SHA가 없다")
    for binding_name in ("atomic-transport-v2.json", "lineage-transport-v3.json",
                         "lineage-transport-v4.json"):
        require(read(nim_dir / binding_name).get("anchor") == anchor,
                "transport source commit SHA가 서로 다르다")

    conditions, private_sources = [], {}
    for index, condition in enumerate(CONDITIONS):
        condition_result_path = execution_dir / condition / "result.json"
        snapshot_path = execution_dir / condition / "snapshot.json"
        condition_result = read(condition_result_path)
        require(condition_result.get("condition") == condition
                and condition_result.get("approval") is False
                and condition_result.get("probe_executed") is False,
                f"{condition} development 실행 범위가 다르다")
        require(result["conditions"][index] == {
            "condition": condition, "choice": condition_result["choice"],
            "requests": condition_result["engine"]["requests"]},
            f"{condition} summary/condition 결과가 다르다")
        snapshot = read(snapshot_path)
        require(condition_result["dialogue_input"]["snapshot_digest"] == DEV.C.L.digest(snapshot),
                f"{condition} snapshot digest가 다르다")
        api = _api_sources(nim_dir, condition)
        metrics = _validate_api(condition, condition_result, api)
        conditions.append({"condition": condition, **metrics})
        private_sources[condition] = {
            "condition_result": _ref(condition_result_path), "snapshot": _ref(snapshot_path),
            **{kind: _ref(path) for kind, path in api.items()},
        }

    public = {
        "schema": SCHEMA, "status": "complete", "case_id": "development-s2307",
        "seed": 2307, "order": list(CONDITIONS), "conditions": conditions,
        "comparison": {
            "telemetry_only_probe": "P_episode",
            "telemetry_and_runtime_probe": "P_optimizer_updates",
            "changed_next_probe": True,
            "runtime_observation_used": True,
            "runtime_input_facts": deepcopy(EXPECTED_RUNTIME_FACTS),
            "runtime_choice_cited_expected_and_updates_only": True,
            "runtime_choice_cited_observed_four": False,
        },
        "scope": {
            "development_pairs": 1, "condition_order_fixed": True,
            "already_seen_seed": True, "posthoc_design": True, "gpu_jobs": 0,
            "recovery_measured": False, "new_seed_evaluation": False,
            "version_superiority_claimed": False, "video_improvement_claimed": False,
        },
        "wording": "이미 본 seed 2307 한 사례의 개발 비교에서 runtime 입력을 추가하자 다음 검사 선택이 P_episode에서 P_optimizer_updates로 바뀌었다. 회복·일반화·버전 우위·영상 개선의 근거는 아니다.",
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA, "source_commit_sha": anchor,
        "development_input_sha256": sha(inputs_path),
        "preregistration_sha256": sha(preregistration),
        "registered_protocol": {
            "path": _registered_protocol_path(preregistration),
            "sha256_lf": sha_lf(preregistration),
        },
        "measured_preregistration": _logical_ref(
            "preregistration.json", sha(preregistration)),
        "execution_result_sha256": sha(result_path),
        "source_protocol_sha256": result["protocol_sha256"],
        "source_hashes_lf": deepcopy(prereg["source_sha256_lf"]),
        "api_files": {},
        "excluded_private_inputs": ["input_packet", "selection_snapshot"],
    }
    out.mkdir(parents=True, exist_ok=False)
    preregistration_raw = preregistration.read_bytes()
    _safe_text(preregistration_raw, "preregistration.json")
    (out / "preregistration.json").write_bytes(preregistration_raw)
    api_out = out / "api"
    api_out.mkdir()
    for condition in CONDITIONS:
        provenance["api_files"][condition] = {}
        for kind, source in _api_sources(nim_dir, condition).items():
            suffix = "sha256.json" if kind == "response_sha" else f"{kind}.json"
            target = api_out / f"{condition}.{suffix}"
            raw = source.read_bytes()
            _safe_text(raw, target.name)
            target.write_bytes(raw)
            provenance["api_files"][condition][kind] = _logical_ref(
                target.relative_to(out).as_posix(), sha(target))
    write_new(out / "summary.json", public)
    write_new(out / "provenance.json", provenance)
    for path in out.rglob("*"):
        if path.is_file():
            _safe_text(path.read_bytes(), path.relative_to(out).as_posix())
    private = {
        "schema": PRIVATE_SCHEMA,
        "sources": {"inputs": _ref(inputs_path), "preregistration": _ref(preregistration),
                    "result": _ref(result_path), **private_sources},
        "public_output": {path.relative_to(out).as_posix(): sha(path)
                          for path in sorted(out.rglob("*")) if path.is_file()},
        "checked_input_schema": inputs["schema"],
    }
    private_out.mkdir(parents=True, exist_ok=False)
    write_new(private_out / "metadata.json", private)
    return deepcopy(public)


def _checked_ref(item: object, expected: Path, label: str) -> Path:
    require(isinstance(item, dict) and set(item) == {"path", "sha256"},
            f"{label} ref 형식이 다르다")
    ref = cast(dict[str, str], item)
    path = Path(ref["path"]).resolve()
    require(path == expected.resolve() and path.is_file() and sha(path) == ref["sha256"],
            f"{label} 원문 경로 또는 SHA가 다르다")
    return path


def _smoke_checked(training: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(cast(dict[str, str], training["manifest"])["path"]).resolve()
    evidence_path = Path(cast(dict[str, str], training["evidence"])["path"]).resolve()
    owned_path = Path(cast(dict[str, str], training["owned_result"])["path"]).resolve()
    for item, path, label in ((training["manifest"], manifest_path, "smoke manifest"),
                              (training["evidence"], evidence_path, "smoke evidence"),
                              (training["owned_result"], owned_path, "smoke owned result")):
        _checked_ref(item, path, label)
    require(training.get("collected") is None and training.get("runtime_facts") is None,
            "smoke에 300-update collected/facts가 있으면 안 된다")
    manifest = R.TRAIN.TRAIN.validate_manifest(read(manifest_path))
    require(manifest["mode"] == "inspection" and manifest["arm"] == "NONE"
            and manifest["seed"] == R.SMOKE_SEED and manifest["max_iterations"] == 3
            and manifest["runtime_updates_to_observe"] == 3,
            "smoke manifest는 NONE/inspection/3-update여야 한다")
    R.TRAIN.EXEC.GUARD.validate_completion(
        evidence_path.parent, R.TRAIN.TRAIN.manifest_fingerprint(manifest))
    observed = R.TRAIN.TRAIN._validate_residual_outputs(manifest)
    evidence, owned = read(evidence_path), read(owned_path)
    require(evidence.get("updates") == 3 and observed["report"].get("records") == 3,
            "smoke guard/observer는 정확히 3 update여야 한다")
    require(evidence.get("checkpoint") == training.get("checkpoint"),
            "smoke evidence/checkpoint가 다르다")
    execution = training.get("execution", {})
    require(execution.get("exit_code") == 0 and execution.get("natural_exit") is True
            and execution.get("cleanup_verified") is True and execution.get("timed_out") is False,
            "smoke 실행 종단/cleanup 검증이 완료되지 않았다")
    require(owned == execution, "smoke owned result와 training execution이 다르다")
    return manifest, {"evidence": evidence, "observed": observed}


def _stage_checked(root: Path, entry: object) -> tuple[str, dict[str, Any], dict[str, Any]]:
    require(isinstance(entry, dict) and set(entry) == {"stage", "training"},
            "stage 항목 형식이 다르다")
    row = cast(dict[str, Any], entry)
    stage = row["stage"]
    require(stage in {"smoke", "none", "compound", "repair1", "repair2"},
            "알 수 없는 stage label")
    training_path = _checked_ref(row["training"], root / stage / "training-result.json",
                                 f"{stage} training")
    training = read(training_path)
    checkpoint = training.get("checkpoint")
    require(isinstance(checkpoint, dict)
            and re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256"))) is not None,
            f"{stage} checkpoint SHA 형식")
    if stage == "smoke":
        manifest, _private = _smoke_checked(training)
        updates = 3
    else:
        _collected, metadata = R._training_checked(training)
        manifest = read(Path(cast(dict[str, str], training["manifest"])["path"]))
        require(manifest.get("mode") == "train" and manifest.get("max_iterations") == 300
                and metadata["private_training"]["seed"] == R.SEED,
                f"{stage}는 seed 2311의 300-update 학습이어야 한다")
        updates = 300
    return stage, training, {
        "stage": stage, "arm": manifest["arm"], "updates": updates,
        "checkpoint_sha256": cast(dict[str, Any], checkpoint)["sha256"],
    }


def _diagnosis_checked(path: Path, applied: list[str]) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == "runtime_observation_diagnosis_v1"
            and value.get("status") == "complete", "진단 결과 schema/status가 다르다")
    final = value.get("final_snapshot")
    require(isinstance(final, dict), "진단 final snapshot ref가 없다")
    final_path = Path(cast(dict[str, str], final)["path"])
    _checked_ref(final, final_path, "진단 final snapshot")
    R.load_verified_snapshot(value)
    end_state = value.get("end_state")
    require(end_state in {"confirmed", "HOLD"}, "진단 end_state가 다르다")
    if end_state == "confirmed":
        _component, correction = R.verify_diagnosis(value, applied)
        applied.append(correction)
    probes: list[dict[str, Any]] = []
    for step in value.get("steps", []):
        require(isinstance(step, dict), "진단 step 형식")
        for key in ("snapshot", "choice", "dialogue", "request_spec", "authorization",
                    "ledger_receipt"):
            item = step.get(key)
            if item is not None:
                path_value = Path(cast(dict[str, str], item)["path"])
                _checked_ref(item, path_value, f"진단 step {key}")
        for group in ("plans", "receipts"):
            for role, item in step.get(group, {}).items():
                path_value = Path(cast(dict[str, str], item)["path"])
                _checked_ref(item, path_value, f"진단 {group} {role}")
        if step.get("probe") is not None:
            probes.append({"probe": step["probe"], "outcome": step.get("outcome")})
    citations = value.get("runtime_fact_citations", [])
    require(isinstance(citations, list), "runtime citation 형식")
    return {"epoch": value.get("epoch"), "end_state": end_state,
            "hold_reason": value.get("hold_reason"), "probes": probes,
            "runtime_fact_citations": deepcopy(citations),
            "model_requests": value.get("requests", 0)}


def _all_api_sources(nim_dir: Path) -> dict[str, dict[str, Path]]:
    requests = sorted((nim_dir / "requests").glob("*.json"))
    grouped: dict[str, dict[str, Path]] = {}
    for request in requests:
        request_id = read(request).get("request_id")
        require(isinstance(request_id, str) and request_id not in grouped, "API request ID 중복")
        request_id = cast(str, request_id)
        candidates = {
            "request": request,
            "response": nim_dir / "responses" / f"{request_id}.json",
            "dispatch": nim_dir / "dispatches" / f"{request_id}.json",
            "response_sha": nim_dir / "response-sha" / f"{request_id}.sha256",
        }
        require(all(path.is_file() for path in candidates.values()), "API 원문 4종이 완전하지 않다")
        response, dispatch = read(candidates["response"]), read(candidates["dispatch"])
        sidecar = read(candidates["response_sha"])
        require(dispatch.get("request_id") == request_id
                and dispatch.get("request_sha256") == sha(request)
                and response.get("request_id") == request_id
                and response.get("request_sha256") == sha(request)
                and response.get("dispatch_sha256") == sha(candidates["dispatch"])
                and sidecar == {"request_id": request_id,
                                "response_sha256": sha(candidates["response"])},
                "API request/dispatch/response/SHA 결합이 다르다")
        for kind, path in candidates.items():
            _safe_text(path.read_bytes(), f"{request_id}.{kind}")
        grouped[request_id] = candidates
    return grouped


def _criteria(recovery: dict[str, Any]) -> dict[str, bool]:
    checks = recovery.get("verdict", {}).get("checks")
    expected = {"survival_ratio", "fall_frac", "err_xy_ratio", "err_yaw_ratio"}
    require(isinstance(checks, dict) and set(checks) == expected
            and all(type(value) is bool for value in checks.values()),
            "회복 판정은 등록된 4개 bool 기준이어야 한다")
    return deepcopy(cast(dict[str, bool], checks))


def export_prospective(
    result_path: Path, *, out: Path, private_out: Path, nim_dir: Path | None = None,
) -> dict[str, Any]:
    """완료된 actual recovery 결과를 원문 재검증 뒤 경로 안전하게 공개한다."""
    result_path = Path(result_path).resolve()
    out, private_out = Path(out).resolve(), Path(private_out).resolve()
    require(result_path.is_file(), "prospective actual result가 아직 없다")
    require(not out.exists() and not private_out.exists(), "prospective 출력은 fresh여야 한다")
    value = read(result_path)
    require(value.get("schema") == "runtime_observation_recovery_result_v1"
            and value.get("status") == "complete" and value.get("seed") == R.SEED,
            "actual recovery result가 complete/seed2311이 아니다")
    require(value.get("claims") == {"new_fault_family": False,
                                    "model_version_superiority": False,
                                    "video_effect": False, "tokenfactory": False},
            "actual recovery claim scope가 다르다")
    outcome = value.get("outcome")
    require(outcome in {"HOLD", "not_recovered", "recovered"}, "recovery outcome이 다르다")
    _number(value.get("wall_s"), "recovery wall_s")
    root = result_path.parent
    require(result_path == root / "result.json", "result는 recovery root/result.json이어야 한다")
    stage_rows: list[dict[str, Any]] = []
    trainings: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    expected_order = ["smoke", "none", "compound", "repair1", "repair2"]
    seen: list[str] = []
    cache_runs: list[dict[str, Any]] = []
    with GIT_CACHE.proof_scope(repo=ROOT) as cache_stats:
        for entry in value.get("stages", []):
            stage, training, public = _stage_checked(root, entry)
            seen.append(stage)
            trainings[stage] = training
            stage_rows.append(public)
            manifests.append(read(Path(cast(dict[str, str], training["manifest"])["path"])))
    cache_runs.append(deepcopy(cache_stats))
    require(bool(seen) and seen == expected_order[:len(seen)] and "none" in trainings,
            "stage 순서/중복/기준 학습이 다르다")
    anchors = {item["source_commit_sha"] for item in manifests}
    require(len(anchors) == 1, "학습 source commit anchor가 서로 다르다")
    anchor = anchors.pop()
    require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
            "학습 anchor 형식이 다르다")

    fixed_public: dict[str, dict[str, Any]] = {}
    fixed_sources: dict[str, Path] = {}
    fixed_reports: dict[str, dict[str, Any]] = {}
    for stage, training in trainings.items():
        report_path = root / stage / "fixed" / "corrected.json"
        if report_path.is_file():
            report = R._check_report(report_path, training)
            require(len(report["by_condition"]) == 26, "fixed report는 26개 조건이어야 한다")
            _safe_text(report_path.read_bytes(), f"fixed/{stage}.json")
            fixed_sources[stage] = report_path
            fixed_reports[stage] = report
            fixed_public[stage] = {"metrics": deepcopy(report["metrics"]),
                                   "conditions": 26}

    repairs = value.get("repairs", [])
    require(isinstance(repairs, list) and len(repairs) <= 2, "repair ref 개수가 다르다")
    repair_paths = [_checked_ref(item, root / f"repair{index}" / "receipt.json",
                                 f"repair {index}")
                    for index, item in enumerate(repairs, 1)]
    bands = R.D.read(R.OLD.BANDS)["bands"]
    require("none" in fixed_reports, "정상 control fixed26 원문이 없다")
    measured_normal = R.C.D1.recovery(fixed_reports["none"], fixed_reports["none"], bands)
    _criteria(measured_normal)
    normal = value.get("normal_control")
    require(isinstance(normal, dict) and normal.get("host_gate") is True
            and normal.get("model_called") is False and normal.get("probes") == 0
            and normal.get("corrections") == 0 and normal.get("recovery") == measured_normal,
            "정상 control host gate/recovery가 다르다")
    assessments = {"normal": deepcopy(measured_normal)}
    if "compound" in fixed_reports:
        source_recovery = R.C.D1.recovery(
            fixed_reports["compound"], fixed_reports["none"], bands)
        _criteria(source_recovery)
        require(value.get("source_recovery") == source_recovery,
                "compound source recovery 재계산이 다르다")
        assessments["source"] = deepcopy(source_recovery)
    diagnoses: list[dict[str, Any]] = []
    applied: list[str] = []
    verified_repairs: list[dict[str, Any]] = []
    with (GIT_CACHE.proof_scope(repo=ROOT) as cache_stats,
          RV.view_scope(RV.PROTOCOL, anchor)):
        for epoch in (1, 2):
            diagnosis_path = root / f"diagnosis-e{epoch}" / "result.json"
            if diagnosis_path.is_file():
                diagnoses.append(_diagnosis_checked(diagnosis_path, applied))
        for repair_path in repair_paths:
            verified_repairs.append(R.verify_repair(
                repair_path, root / "corrections.ledger.jsonl", trainings["none"], bands))
    cache_runs.append(deepcopy(cache_stats))
    require((outcome != "recovered" or bool(verified_repairs)
             and verified_repairs[-1]["recovery"]["outcome"] == "recovered"),
            "recovered summary와 repair 원문이 다르다")
    require(outcome != "not_recovered" or bool(verified_repairs)
            and verified_repairs[-1]["recovery"]["outcome"] == "not_recovered",
            "not_recovered summary와 repair 원문이 다르다")
    require(outcome != "HOLD" or not any(
        item["recovery"]["outcome"] == "recovered" for item in verified_repairs),
        "HOLD를 회복으로 합산할 수 없다")

    api_sources = {} if nim_dir is None else _all_api_sources(Path(nim_dir).resolve())
    public = {
        "schema": "runtime_observation_recovery_export_v1", "status": "complete",
        "seed": R.SEED, "outcome": outcome, "reason": value.get("reason"),
        "recovery_count": 1 if outcome == "recovered" else 0,
        "normal_control": {"host_gate": True, "model_called": False,
                           "probes": 0, "corrections": 0},
        "stages": stage_rows, "fixed26": fixed_public, "epochs": diagnoses,
        "recovery_assessments": assessments,
        "repairs": [{"epoch": item["epoch"], "outcome": item["recovery"]["outcome"],
                     "criteria": _criteria(item["recovery"])}
                    for item in verified_repairs],
        "api_requests": len(api_sources),
        "scope": {"prospective_seed": True, "fixed_conditions": 26,
                  "recovery_criteria": 4, "unknown_is_recovery": False,
                  "approval": False, "execution_allowed": False,
                  "immutable_git_cache_scope": "cpu_export_read_verification_only",
                  "new_fault_family_claimed": False, "model_version_superiority_claimed": False,
                  "video_improvement_claimed": False},
    }
    _safe_text(json.dumps(public, ensure_ascii=False, allow_nan=False).encode(), "summary.json")

    out.mkdir(parents=True, exist_ok=False)
    for stage, source in fixed_sources.items():
        target = out / "fixed" / f"{stage}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for request_id, sources in api_sources.items():
        for kind, source in sources.items():
            suffix = "sha256.json" if kind == "response_sha" else f"{kind}.json"
            target = out / "api" / f"{request_id}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    write_new(out / "summary.json", public)
    private = {
        "schema": "runtime_observation_recovery_export_metadata_v1",
        "result": _ref(result_path),
        "trainings": {stage: _ref(root / stage / "training-result.json") for stage in trainings},
        "fixed": {stage: _ref(path) for stage, path in fixed_sources.items()},
        "diagnoses": {str(item["epoch"]): _ref(root / f"diagnosis-e{item['epoch']}/result.json")
                      for item in diagnoses},
        "repairs": [_ref(path) for path in repair_paths],
        "api": {request_id: {kind: _ref(path) for kind, path in sources.items()}
                for request_id, sources in api_sources.items()},
        "immutable_git_cache": cache_runs,
        "public_output": {path.relative_to(out).as_posix(): sha(path)
                          for path in sorted(out.rglob("*")) if path.is_file()},
    }
    private_out.mkdir(parents=True, exist_ok=False)
    write_new(private_out / "metadata.json", private)
    return deepcopy(public)


def _paired_diagnosis(path: Path, applied: list[str]) -> dict[str, Any]:
    """진단 ref와 paired receipt raw를 outcome까지 다시 계산한다."""
    public = _diagnosis_checked(path, applied)
    value = read(path)
    public_probes = []
    for step in value.get("steps", []):
        receipts = step.get("receipts", {})
        plans = step.get("plans", {})
        if not receipts:
            continue
        require(set(receipts) == set(plans) == {"case", "reference"},
                "paired probe plan/receipt role이 완전하지 않다")
        verified = {
            role: CONT.PROBE.verify_receipt(
                Path(plans[role]["path"]), Path(receipts[role]["path"]))
            for role in ("case", "reference")
        }
        if all(item.get("status") == "complete" for item in verified.values()):
            measured = R.C.classify_raw(
                step["probe"], verified["case"]["measurement"],
                verified["reference"]["measurement"])
        else:
            measured = "unknown"
        require(measured == step.get("outcome"), "paired receipt raw outcome 재계산 불일치")
        public_probes.append({
            "probe": step["probe"], "outcome": measured,
            "measurements": {
                role: _measurement_projection(
                    step["probe"], verified[role]["measurement"],
                    Path(plans[role]["path"]), Path(receipts[role]["path"]))
                for role in ("reference", "case")
            },
        })
    public["probes"] = public_probes
    return public


def _numeric_measurement(value: object, label: str) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        require(math.isfinite(value), f"{label} 측정값은 유한해야 한다")
        return value
    if isinstance(value, list):
        return [_numeric_measurement(item, label) for item in value]
    if isinstance(value, dict):
        require(all(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", key)
                    for key in value), f"{label} 측정 key 형식")
        return {key: _numeric_measurement(item, f"{label}.{key}")
                for key, item in value.items()}
    raise TypeError(f"{label} 공개 측정은 숫자/bool/null만 허용한다")


def _measurement_projection(
    probe: str, measurement: object, plan_path: Path, receipt_path: Path,
) -> object:
    require(isinstance(measurement, dict), "검증된 probe measurement 객체가 없다")
    if probe != "P_optimizer_updates":
        return _numeric_measurement(measurement, probe)
    measured = cast(dict[str, Any], measurement)
    manifest = measured.get("manifest")
    require(isinstance(manifest, dict), "optimizer measurement manifest가 없다")
    manifest = cast(dict[str, Any], manifest)
    expected_steps = manifest.get("expected_optimizer_steps_per_update")
    expected_batches = manifest.get("expected_minibatches_per_update")
    records = measured.get("records")
    require(type(expected_steps) is int and type(expected_batches) is int
            and type(records) is int and records == 3
            and expected_steps >= 0 and expected_batches >= 0,
            "optimizer canonical expected/records 숫자가 다르다")
    plan, receipt = read(plan_path), read(receipt_path)
    records_path = Path(plan.get("raw_files", {}).get("records", "")).resolve()
    _checked_ref(receipt.get("raw_refs", {}).get("records"), records_path,
                 "optimizer receipt records")
    rows = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        require(isinstance(row, dict), "optimizer record 행 형식")
        rows.append(row)
    require(len(rows) == records and [row.get("update_index") for row in rows] == [0, 1, 2],
            "optimizer 공개 update 행은 0/1/2여야 한다")
    updates = []
    for row in rows:
        expected, observed = row.get("expected"), row.get("observed")
        require(isinstance(expected, dict) and isinstance(observed, dict)
                and expected.get("optimizer_steps") == expected_steps
                and expected.get("minibatches") == expected_batches,
                "optimizer record expected와 manifest가 다르다")
        observed_steps = observed.get("optimizer_steps")
        observed_batches = observed.get("minibatches")
        require(type(observed_steps) is int and type(observed_batches) is int
                and observed_steps >= 0 and observed_batches >= 0,
                "optimizer observed count 형식")
        updates.append({
            "update_index": row["update_index"],
            "expected_optimizer_steps": expected_steps,
            "observed_optimizer_steps": observed_steps,
            "expected_minibatches": expected_batches,
            "observed_minibatches": observed_batches,
        })
    return {"expected_optimizer_steps": expected_steps,
            "expected_minibatches": expected_batches,
            "records": records, "updates": updates}


def _combined_api(queue: Path) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]]]:
    sources = _all_api_sources(queue)
    public = []
    for request_id, files in sources.items():
        require(re.fullmatch(r"[A-Za-z0-9._-]+", request_id) is not None,
                "공개 API request ID 파일명이 안전하지 않다")
        request, response = read(files["request"]), read(files["response"])
        result = response.get("result")
        require(request.get("schema") == "lineage_nim_transport_request_v1"
                and response.get("schema") == "lineage_nim_transport_response_v1"
                and isinstance(result, dict), "continuation API schema/result가 다르다")
        result = cast(dict[str, Any], result)
        input_tokens = _number(result.get("input_tokens"), "continuation input tokens", integer=True)
        output_tokens = _number(result.get("output_tokens"), "continuation output tokens", integer=True)
        total_tokens = _number(result.get("total_tokens"), "continuation total tokens", integer=True)
        require(total_tokens == input_tokens + output_tokens, "continuation token 합계가 다르다")
        public.append({
            "request_id": request_id, "slot_id": request.get("slot_id"),
            "provider_model": result.get("provider_model"),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "api_wall_s": _number(result.get("wall_s"), "continuation API wall_s"),
            "raw": {kind: f"raw/api/{request_id}.{'sha256.json' if kind == 'response_sha' else kind + '.json'}"
                    for kind in files},
        })
    return sources, public


def _runtime_fact_sources(stage: str, training: dict[str, Any]) -> dict[str, Path]:
    checkpoint = training.get("checkpoint")
    facts = training.get("runtime_facts")
    require(isinstance(checkpoint, dict) and isinstance(facts, dict)
            and set(facts) == {"case", "reference"},
            f"{stage} runtime fact/checkpoint ref가 없다")
    checkpoint = cast(dict[str, Any], checkpoint)
    facts = cast(dict[str, Any], facts)
    sources = {}
    for role in ("case", "reference"):
        item = facts[role]
        require(isinstance(item, dict), f"{stage} {role} runtime fact ref 형식")
        path = Path(cast(dict[str, str], item)["path"]).resolve()
        _checked_ref(item, path, f"{stage} {role} runtime facts")
        R.TRAIN.FACT.checked(
            path, expected_checkpoint_sha256=checkpoint["sha256"])
        _safe_text(path.read_bytes(), f"raw/runtime/{stage}-{role}.json")
        sources[role] = path
    return sources


def export_combined(
    inputs_path: Path, continuation_result_path: Path, continuation_queue: Path,
    out: Path, private_provenance: Path,
) -> dict[str, Any]:
    """유효한 첫 미회복 receipt와 별도 continuation 결과를 한 공개 기록으로 묶는다."""
    inputs_path = Path(inputs_path).resolve()
    continuation_result_path = Path(continuation_result_path).resolve()
    continuation_queue = Path(continuation_queue).resolve()
    out, private_provenance = Path(out).resolve(), Path(private_provenance).resolve()
    require(inputs_path.is_file() and continuation_result_path.is_file(),
            "combined input/result 원문이 아직 없다")
    require(not out.exists() and not private_provenance.exists(),
            "combined export 출력은 fresh여야 한다")
    result = read(continuation_result_path)
    require(result.get("schema") == CONT.RESULT_SCHEMA and result.get("status") == "complete"
            and result.get("seed") == R.SEED
            and result.get("origin_status") == "valid_first_not_recovered",
            "continuation result schema/status/origin 결합이 다르다")
    continuation_root = continuation_result_path.parent
    require(continuation_result_path == continuation_root / "result.json",
            "continuation result는 output root/result.json이어야 한다")
    _checked_ref(result.get("origin_inputs"), inputs_path, "continuation origin inputs")
    outcome = result.get("outcome")
    require(outcome in {"HOLD", "not_recovered", "recovered"},
            "continuation outcome label이 다르다")
    _number(result.get("wall_s"), "continuation wall_s")

    with GIT_CACHE.proof_scope(repo=ROOT) as cache_stats:
        inputs = CONT.verify_origin(inputs_path)
        origin = inputs["origin"]
        origin_anchor = inputs["origin_anchor"]
        continuation_anchor = inputs["continuation_anchor"]
        origin_protocol_path = Path(inputs["origin_protocol"]["path"])
        continuation_protocol_path = Path(inputs["continuation_protocol"]["path"])
        _checked_ref(inputs["origin_protocol"], origin_protocol_path, "origin protocol")
        _checked_ref(inputs["continuation_protocol"], continuation_protocol_path,
                     "continuation protocol")
        phase = origin["phase"]
        require(phase.get("status") == "invalid"
                and Path(phase["artifact"]["path"]).name == "failure.json",
                "origin 전체 invalid 상태가 provenance에 보존되지 않았다")
        origin_root = Path(inputs["origin_root"]).resolve()
        freshness_path = origin_root / "freshness-before-execution.json"
        freshness = read(freshness_path)
        require(freshness.get("schema") == "core_seed_freshness_v1"
                and freshness.get("status") == "fresh"
                and freshness.get("seeds") == [R.SEED, R.SMOKE_SEED]
                and freshness.get("evidence") == [],
                "origin seed freshness 원문이 다르다")
        first_receipt_path = _checked_ref(
            origin["receipt"], origin_root / "repair1/receipt.json", "origin first receipt")
        first_receipt = read(first_receipt_path)
        require(first_receipt.get("recovery", {}).get("outcome") == "not_recovered",
                "origin first receipt는 not_recovered여야 한다")
        reference_path = _checked_ref(
            origin["reference"], origin_root / "none/training-result.json", "origin reference")
        compound_path = _checked_ref(
            origin["compound"], origin_root / "compound/training-result.json", "origin compound")
        first_path = _checked_ref(
            origin["first"], origin_root / "repair1/training-result.json", "origin first training")
        reference, compound, first = read(reference_path), read(compound_path), read(first_path)
        bands = R.D.read(R.OLD.BANDS)["bands"]
        with RV.view_scope(RV.PROTOCOL, origin_anchor):
            verified_first = R.verify_repair(
                first_receipt_path, origin_root / "corrections.ledger.jsonl", reference, bands)
            first_applied: list[str] = []
            first_diagnosis = _paired_diagnosis(
                Path(first_receipt["diagnosis"]["path"]), first_applied)
        require(first_applied == first_receipt["applied_after"],
                "origin paired diagnosis와 first correction이 다르다")

        reference_report = R._check_report(
            origin_root / "none/fixed/corrected.json", reference)
        compound_report = R._check_report(
            origin_root / "compound/fixed/corrected.json", compound)
        first_report = R._check_report(
            origin_root / "repair1/fixed/corrected.json", first)
        normal = R.C.D1.recovery(reference_report, reference_report, bands)
        source = R.C.D1.recovery(compound_report, reference_report, bands)
        require(normal["verdict"]["label"] == "healthy"
                and source["verdict"]["label"] == "unhealthy"
                and verified_first["recovery"] == R.C.D1.recovery(
                    first_report, reference_report, bands),
                "origin 정상/source/first recovery 재계산 불일치")
        runtime_fact_sources = {
            "none": _runtime_fact_sources("none", reference),
            "compound": _runtime_fact_sources("compound", compound),
            "repair1": _runtime_fact_sources("repair1", first),
        }

        diagnosis_path = _checked_ref(
            result.get("diagnosis"), continuation_root / "diagnosis-e2/result.json",
            "continuation diagnosis")
        second_applied = list(first_receipt["applied_after"])
        with RV.view_scope(RV.PROTOCOL, continuation_anchor):
            second_diagnosis = _paired_diagnosis(diagnosis_path, second_applied)

        verified_second = None
        second_report = None
        if result.get("continuation_status") == "HOLD":
            require(outcome == "HOLD" and result.get("second_repair_training") == 0
                    and result.get("stages") == [] and result.get("repairs") == []
                    and result.get("receipt") is None,
                    "HOLD continuation이 repair/training을 성공으로 위장한다")
        else:
            require(result.get("continuation_status") == "complete"
                    and result.get("second_repair_training") == 1
                    and len(result.get("stages", [])) == 1
                    and len(result.get("repairs", [])) == 1,
                    "완료 continuation의 second repair 범위가 다르다")
            second_receipt_path = _checked_ref(
                result["repairs"][0], continuation_root / "repair2/receipt.json",
                "continuation second receipt")
            require(result.get("receipt") == result["repairs"][0],
                    "continuation summary/repair receipt ref가 다르다")
            stage = result["stages"][0]
            require(stage.get("stage") == "repair2", "continuation stage는 repair2여야 한다")
            second_training_path = _checked_ref(
                stage.get("training"), continuation_root / "repair2/training-result.json",
                "continuation repair2 training")
            second_training = read(second_training_path)
            with RV.view_scope(RV.PROTOCOL, continuation_anchor):
                verified_second = R.verify_repair(
                    second_receipt_path, continuation_root / "corrections.ledger.jsonl",
                    reference, bands)
            require(verified_second["recovery"]["outcome"] == outcome,
                    "continuation result/second receipt outcome이 다르다")
            second_report = R._check_report(
                continuation_root / "repair2/fixed/corrected.json", second_training)
            runtime_fact_sources["repair2"] = _runtime_fact_sources(
                "repair2", second_training)
    cache_record = deepcopy(cache_stats)

    origin_queue = Path(origin["queue"]["root"]).resolve()
    origin_queue_status = CONT.TRANSPORT.status(origin_queue)
    continuation_queue_status = CONT.TRANSPORT.status(continuation_queue)
    require(isinstance(continuation_queue_status.get("closed"), dict)
            and continuation_queue_status.get("requests")
            == continuation_queue_status.get("dispatches")
            == continuation_queue_status.get("responses") == second_diagnosis["model_requests"]
            and continuation_queue_status.get("pending") == continuation_queue_status.get("unknown")
            == continuation_queue_status.get("failures") == 0,
            "continuation queue가 완료된 provider 요청과 다르다")
    require(isinstance(origin_queue_status.get("closed"), dict)
            and origin_queue_status.get("requests") == origin_queue_status.get("dispatches")
            == origin_queue_status.get("responses") == origin["origin_requests"]
            and origin_queue_status.get("pending") == origin_queue_status.get("unknown")
            == origin_queue_status.get("failures") == 0,
            "origin queue가 완료된 provider 요청과 다르다")
    origin_api_sources, origin_provider = _combined_api(origin_queue)
    continuation_api_sources, continuation_provider = _combined_api(continuation_queue)
    require(len(origin_provider) == first_diagnosis["model_requests"] == origin["origin_requests"],
            "origin API raw 개수와 진단 requests가 다르다")
    require(len(continuation_provider) == second_diagnosis["model_requests"],
            "continuation API raw 개수와 진단 requests가 다르다")
    require(set(origin_api_sources).isdisjoint(continuation_api_sources),
            "origin/continuation API request ID가 중복됐다")
    origin_session, continuation_session = read(origin_queue / "session.json"), read(
        continuation_queue / "session.json")
    require(origin_session.get("registration_anchor") == origin_anchor
            and continuation_session.get("registration_anchor") == continuation_anchor,
            "provider queue registration anchor가 protocol anchor와 다르다")

    trainings = [
        {"stage": "none", "arm": "NONE", "updates": 300,
         "checkpoint_sha256": reference["checkpoint"]["sha256"]},
        {"stage": "compound", "arm": "COMPOUND", "updates": 300,
         "checkpoint_sha256": compound["checkpoint"]["sha256"]},
        {"stage": "repair1", "arm": "FIRST", "updates": 300,
         "checkpoint_sha256": first["checkpoint"]["sha256"]},
    ]
    fixed = {
        "none": {"conditions": len(reference_report["by_condition"]),
                 "metrics": deepcopy(reference_report["metrics"])},
        "compound": {"conditions": len(compound_report["by_condition"]),
                     "metrics": deepcopy(compound_report["metrics"])},
        "repair1": {"conditions": len(first_report["by_condition"]),
                    "metrics": deepcopy(first_report["metrics"])},
    }
    if second_report is not None and verified_second is not None:
        trainings.append({"stage": "repair2", "arm": "BOTH", "updates": 300,
                          "checkpoint_sha256": verified_second["output_training"]["checkpoint"]["sha256"]})
        fixed["repair2"] = {"conditions": len(second_report["by_condition"]),
                            "metrics": deepcopy(second_report["metrics"])}
    fixed_sources = {
        "none": origin_root / "none/fixed/corrected.json",
        "compound": origin_root / "compound/fixed/corrected.json",
        "repair1": origin_root / "repair1/fixed/corrected.json",
    }
    if second_report is not None:
        fixed_sources["repair2"] = continuation_root / "repair2/fixed/corrected.json"
    for name, source_path in fixed_sources.items():
        require(source_path.is_file(), f"{name} fixed raw가 없다")
        _safe_text(source_path.read_bytes(), f"raw/fixed/{name}.json")
        fixed[name]["raw"] = {"path": f"raw/fixed/{name}.json", "sha256": sha(source_path)}
    origin_provider_public = [{"phase": "origin_epoch1", **item} for item in origin_provider]
    continuation_provider_public = [
        {"phase": "continuation_epoch2", **item} for item in continuation_provider]
    provider_requests = [*origin_provider_public, *continuation_provider_public]
    runtime_facts = {
        stage: {role: {"path": f"raw/runtime/{stage}-{role}.json", "sha256": sha(path)}
                for role, path in roles.items()}
        for stage, roles in runtime_fact_sources.items()
    }
    public = {
        "schema": "runtime_observation_combined_export_v1", "status": "complete",
        "seed": R.SEED, "origin_run": {
            "status": "invalid", "queue_closed": True,
            "valid_first_receipt": True, "first_outcome": "not_recovered"},
        "normal_gate": {"model_called": False, "healthy": True,
                        "criteria": _criteria(normal)},
        "source_gate": {"unhealthy": True, "criteria": _criteria(source)},
        "protocols": {
            "origin": {"path": _registered_protocol_path(origin_protocol_path),
                       "sha256_lf": sha_lf(origin_protocol_path), "anchor": origin_anchor},
            "continuation": {"path": _registered_protocol_path(continuation_protocol_path),
                             "sha256_lf": sha_lf(continuation_protocol_path),
                             "anchor": continuation_anchor},
        },
        "anchors": {
            "origin_protocol": origin_anchor,
            "origin_provider_queue": origin_session["registration_anchor"],
            "continuation_protocol_and_provider_queue": continuation_anchor,
        },
        "seed_scope": {"origin_registered_fresh": True,
                       "origin_freshness_sha256": sha(freshness_path),
                       "continuation_same_seen_seed": True,
                       "new_independent_seed": False},
        "epochs": [first_diagnosis, second_diagnosis],
        "continuation": {"status": result["continuation_status"],
                         "outcome": outcome, "reason": result.get("reason")},
        "trainings": trainings, "fixed26": fixed, "runtime_facts": runtime_facts,
        "provider_by_phase": {"origin_epoch1": origin_provider_public,
                              "continuation_epoch2": continuation_provider_public},
        "provider_requests": provider_requests,
        "provider_requests_scope": "origin_epoch1_and_continuation_epoch2",
        "total_provider_request_count": len(provider_requests),
        "scope": {"origin_invalid_preserved": True, "approval": False,
                  "execution_allowed": False, "new_independent_seed": False,
                  "recovery_count": 1 if outcome == "recovered" else 0},
    }
    _safe_text(json.dumps(public, ensure_ascii=False, allow_nan=False).encode(), "summary.json")
    out.mkdir(parents=True, exist_ok=False)
    all_api_sources = {**origin_api_sources, **continuation_api_sources}
    for request_id, sources in all_api_sources.items():
        for kind, source_path in sources.items():
            suffix = "sha256.json" if kind == "response_sha" else f"{kind}.json"
            target = out / "raw/api" / f"{request_id}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_path.read_bytes())
    for name, source_path in fixed_sources.items():
        target = out / "raw/fixed" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    for stage, roles in runtime_fact_sources.items():
        for role, source_path in roles.items():
            target = out / "raw/runtime" / f"{stage}-{role}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_path.read_bytes())
    write_new(out / "summary.json", public)
    for path in out.rglob("*"):
        if path.is_file():
            _safe_text(path.read_bytes(), path.relative_to(out).as_posix())
    private = {
        "schema": "runtime_observation_combined_provenance_v1",
        "inputs": _ref(inputs_path), "continuation_result": _ref(continuation_result_path),
        "origin_failure": deepcopy(phase["artifact"]),
        "origin_first_receipt": _ref(first_receipt_path),
        "bindings": {
            "origin_failure_sha256": phase["artifact"]["sha256"],
            "origin_first_receipt_sha256": sha(first_receipt_path),
            "continuation_inputs_sha256": sha(inputs_path),
            "continuation_result_sha256": sha(continuation_result_path),
        },
        "origin_first_diagnosis": deepcopy(first_receipt["diagnosis"]),
        "continuation_diagnosis": _ref(diagnosis_path),
        "origin_queue": {"status": origin_queue_status,
                         "raw": {request_id: {kind: _ref(path)
                                              for kind, path in sources.items()}
                                 for request_id, sources in origin_api_sources.items()}},
        "continuation_queue": {"status": continuation_queue_status,
                               "raw": {request_id: {kind: _ref(path)
                                                    for kind, path in sources.items()}
                                       for request_id, sources in continuation_api_sources.items()}},
        "freshness": _ref(freshness_path),
        "fixed": {name: _ref(path) for name, path in fixed_sources.items()},
        "runtime_facts": {stage: {role: _ref(path) for role, path in roles.items()}
                          for stage, roles in runtime_fact_sources.items()},
        "immutable_git_cache": cache_record,
        "public_output": {path.relative_to(out).as_posix(): sha(path)
                          for path in sorted(out.rglob("*")) if path.is_file()},
    }
    private_provenance.mkdir(parents=True, exist_ok=False)
    write_new(private_provenance / "metadata.json", private)
    return deepcopy(public)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    development = sub.add_parser("development")
    for name in ("execution_dir", "nim_dir", "preregistration", "inputs_path", "out", "private_out"):
        development.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    prospective = sub.add_parser("prospective")
    for name in ("result_path", "out", "private_out"):
        prospective.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    prospective.add_argument("--nim-dir", type=Path)
    combined = sub.add_parser("combined")
    for name in ("inputs_path", "continuation_result_path", "continuation_queue",
                 "out", "private_provenance"):
        combined.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.command == "development":
        export_development(args.execution_dir, args.nim_dir, args.preregistration,
                           args.inputs_path, out=args.out, private_out=args.private_out)
    elif args.command == "prospective":
        export_prospective(args.result_path, out=args.out, private_out=args.private_out,
                           nim_dir=args.nim_dir)
    else:
        export_combined(args.inputs_path, args.continuation_result_path,
                        args.continuation_queue, args.out, args.private_provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

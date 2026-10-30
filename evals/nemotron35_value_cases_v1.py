"""Nemotron 3.5 영상 입력이 검사 선택에 더하는 값을 가를 사례 입력을 CPU에서 결합한다.

모델·GPU·네트워크를 호출하지 않는다. 숫자 입력은 기존 선택 계약(selection_comparison_view_v1)의
packet·snapshot·messages를 그대로 쓰고, 같은 checkpoint의 과거 검사 receipt와 영상 렌더 근거를
SHA로 결합한다. 유효 프레임이 없으면 영상 조건은 차단하며, 결과가 없는 항목은 pending/HOLD로 둔다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import selection_comparison_view_v1 as V

M, L = V.M, V.L
PROTOCOL = ROOT / "bench/protocols/nemotron35_value_cases_20261030_v1.json"
PROTOCOL_SCHEMA = "nemotron35_value_cases_protocol_v1"
MANIFEST_SCHEMA = "nemotron35_value_cases_manifest_v1"
CHOICES_SCHEMA = "nemotron35_value_cases_choices_v1"
RUN_ID = "nemotron35_value_cases_v1"
SOURCE_FILES = ("evals/nemotron35_value_cases_v1.py", "tests/test_nemotron35_value_cases_v1.py")
BINARY_SUFFIXES = frozenset({".mp4", ".png", ".jpg", ".gif"})
MODEL_ARMS = ("A", "B", "C")
CHOICE_STATES = frozenset({"valid", "semantic_invalid", "infrastructure_failed", "unknown_pending",
                           "model_unavailable"})
TERMINAL_STATES = frozenset({"valid", "semantic_invalid"})
PHASES = frozenset({"development_already_seen", "confirmation_new_seed"})
ROLES = frozenset({"case_of_interest", "specificity_control"})
FRAME_FACT = re.compile(r"vision:frame:f0[1-6]")
CLAIMS = {"image_selection_benefit": False, "model_superiority": False, "cost_saving": False,
          "generalization": False, "new_recovery": False}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    """텍스트는 LF로 정규화하고 영상·이미지는 원문 바이트로 해시한다."""
    raw = path.read_bytes()
    if path.suffix.lower() not in BINARY_SUFFIXES:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(raw: str) -> None:
    raise ValueError(f"비표준 JSON 수치: {raw}")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        require(key not in value, f"중복 JSON key: {key}")
        value[key] = item
    return value


def strict_json(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject_constant)


def _relative(rel: object) -> str:
    require(isinstance(rel, str) and bool(rel), "결합 경로가 문자열이 아니다")
    parts = PurePosixPath(rel).parts
    require(not PurePosixPath(rel).is_absolute() and ".." not in parts and "\\" not in rel,
            f"저장소 상대 경로가 아니다: {rel}")
    return rel


def _bound_path(protocol: dict, root: Path, rel: object) -> Path:
    """사전등록에 SHA가 적힌 일반 파일만 읽는다."""
    rel = _relative(rel)
    expected = protocol["bindings_sha256"].get(rel)
    require(expected is not None, f"사전등록에 결합되지 않은 입력: {rel}")
    path = root / rel
    require(path.is_file() and not path.is_symlink() and os.stat(path).st_nlink == 1,
            f"일반 파일이 아닌 입력: {rel}")
    require(file_sha256(path) == expected, f"입력 SHA가 사전등록과 다르다: {rel}")
    return path


def _bound_json(protocol: dict, root: Path, rel: object) -> dict:
    value = strict_json(_bound_path(protocol, root, rel).read_bytes())
    require(isinstance(value, dict), f"JSON 객체가 아닌 입력: {rel}")
    return value


def load_protocol(path: Path = PROTOCOL) -> dict:
    value = strict_json(Path(path).read_bytes())
    require(isinstance(value, dict), "사전등록이 JSON 객체가 아니다")
    return value


def verify_protocol(protocol: dict, root: Path = ROOT) -> None:
    """DRAFT 상태·소스 결합·입력 파일 SHA를 확인한다. 실행 승인을 만들지 않는다."""
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "사전등록 schema가 다르다")
    require(protocol.get("status") == "DRAFT_not_executed" and protocol.get("executed") is False,
            "미실행 DRAFT 사전등록만 받는다")
    sources = protocol.get("source_sha256_lf")
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "결합 소스 목록이 다르다")
    for rel, expected in sources.items():
        require(file_sha256(ROOT / rel) == expected, f"사전등록 뒤 소스가 바뀌었다: {rel}")
    bindings = protocol.get("bindings_sha256")
    require(isinstance(bindings, dict) and bool(bindings), "입력 결합 표가 없다")
    for rel in bindings:
        _bound_path(protocol, root, rel)
    lineages = protocol.get("lineages")
    require(isinstance(lineages, list) and len(lineages) >= 2, "계보 목록이 부족하다")
    ids = [row.get("lineage_id") for row in lineages]
    require(len(set(ids)) == len(ids) and all(isinstance(item, str) and item for item in ids), "계보 ID 중복")


def _facts(telemetry: dict, action_scale: float) -> dict:
    """생산 packet과 같은 수치 사실: 학습 기록 mean_last_20pct와 action_scale."""
    facts: dict = {"config:action_scale": action_scale}
    for role in ("case", "reference"):
        for name, row in telemetry[role].items():
            key = f"telemetry:{role}:{name}"
            if key in V.BASE_FACT_IDS and isinstance(row, dict):
                value = row.get("mean_last_20pct")
                if value is None or V.number(value):
                    facts[key] = value
    return facts


def _receipts(protocol: dict, root: Path, entry: dict, demo: dict, checkpoint: str) -> dict:
    """같은 case checkpoint의 완료 검사 원문만 결합한다. 새 실행 결과로 표시하지 않는다."""
    rows: dict = {}
    for request in demo["loop"]["requests"]:
        probe = request["probe"]
        require(request.get("state") == "done" and request.get("outcome") in L.OUTCOMES, "완료되지 않은 검사 receipt")
        rel = request["artifact"]["probe"]
        artifact = _bound_json(protocol, root, rel)
        require(artifact["checkpoint"]["sha256"] == checkpoint, "검사 원문 checkpoint가 계보와 다르다")
        require(artifact["probes"][probe]["measurement"] == request["measurement"], "검사 원문 측정값이 판정 파일과 다르다")
        require(probe not in rows, "같은 검사 receipt가 두 번 있다")
        rows[probe] = {"outcome": request["outcome"], "source": "historical_loop_receipt", "artifact": rel,
                       "artifact_sha256": protocol["bindings_sha256"][rel]}
    require(demo["loop"]["observed"] == {probe: row["outcome"] for probe, row in rows.items()},
            "판정 파일의 관측 요약과 receipt가 다르다")
    spec, rel = demo.get("specificity") or {}, entry.get("specificity_artifact")
    require(bool(spec.get("outcomes")) == (rel is not None), "특이도 검사 원문 결합이 판정 파일과 다르다")
    if rel is not None:
        artifact = _bound_json(protocol, root, rel)
        require(artifact["checkpoint"]["sha256"] == checkpoint, "특이도 검사 checkpoint가 계보와 다르다")
        for probe, outcome in spec["outcomes"].items():
            require(outcome in L.OUTCOMES and artifact["probes"][probe]["measurement"] == spec["measurements"][probe],
                    "특이도 검사 원문과 판정 파일이 다르다")
            if probe in rows:
                require(rows[probe]["outcome"] == outcome, "같은 검사의 receipt 판정이 서로 다르다")
                continue
            rows[probe] = {"outcome": outcome, "source": "historical_specificity_receipt", "artifact": rel,
                           "artifact_sha256": protocol["bindings_sha256"][rel]}
    return rows


def _manifest_video(source: dict, rel: str) -> tuple[str, str]:
    if source.get("schema") == "agentic_motor_capture_public_sources_v1":
        rows = [row for row in source["policies"] if row["video"]["path"] == rel]
        require(len(rows) == 1, "출처 manifest에 영상이 없다")
        return rows[0]["video"]["sha256"], rows[0]["checkpoint_sha256"]
    if source.get("schema") == "public_actual_play_video_v1":
        require(source["video"]["path"] == PurePosixPath(rel).name, "출처 manifest에 영상이 없다")
        return source["video"]["sha256"], source["checkpoint_sha256"]
    raise ValueError("등록되지 않은 영상 출처 manifest")


def media_decision(protocol: dict, root: Path, item: dict, checkpoints: dict, stale: set) -> dict:
    """프레임 후보의 계보·렌더 근거를 판정한다. 바꿔끼움은 거부하고 stale/미확인 렌더는 불수용한다."""
    role = item["role"]
    require(role in ("case", "reference"), "영상 role이 다르다")
    sha = file_sha256(_bound_path(protocol, root, item["video"]))
    recorded_sha, checkpoint = _manifest_video(_bound_json(protocol, root, item["source_manifest"]), item["video"])
    require(recorded_sha == sha, "영상 바이트와 출처 manifest SHA가 다르다")
    require(checkpoint == checkpoints[role], "영상 checkpoint가 계보 role과 다르다")
    decision = {"media_id": item["media_id"], "role": role, "video": item["video"], "video_sha256": sha,
                "checkpoint_sha256": checkpoint}
    kind = item["render_evidence"]["kind"]
    evidence = _bound_json(protocol, root, item["render_evidence"]["path"])
    if sha in stale:
        return {**decision, "accepted": False, "reason": "stale_render_observed"}
    if kind == "v9_capture_followup":
        require(evidence.get("schema") == "agentic_motor_render_visibility_followup_v1", "렌더 후속 판정 형식")
        deferred = evidence.get("all_four_model_frame_readiness") == "deferred"
        return {**decision, "accepted": False,
                "reason": "render_sync_unverified_model_use_deferred" if deferred else "no_render_sync_receipt"}
    if kind == "v11_render_preflight":
        require(evidence.get("schema") == "agentic_motor_render_preflight_public_v11", "렌더 시험 형식")
        require(evidence.get("media") == item["video"], "render receipt가 다른 영상을 가리킨다")
        sync = evidence["render_sync"]
        ok = (evidence["checks"].get("render_sync_verified") is True and sync.get("physics_snapshot_exact_equal") is True
              and sync.get("forward_calls") == evidence.get("frames"))
        return {**decision, "accepted": ok, "reason": "render_sync_verified" if ok else "render_sync_check_failed"}
    raise ValueError("등록되지 않은 렌더 근거 형식")


def _continue(registry: dict, hypotheses: list, observed: dict, policy: str) -> str | None:
    if policy == "fixed_eligible":
        return next((probe for probe in registry["fixed_order"]
                     if V.eligible(registry, hypotheses, observed, probe)), None)
    return M._next(registry, hypotheses, observed)


def replay(registry: dict, candidates: list, first: str | None, outcomes: dict, continuation: str,
           budget: int) -> dict:
    """첫 검사 뒤 등록 규칙으로 잇되, 같은 checkpoint의 실제 receipt가 없으면 그 자리에서 pending."""
    require(continuation in ("registry_next", "fixed_eligible"), "등록되지 않은 후속 규칙")
    hypotheses, observed, steps, probe = list(candidates), {}, [], first
    status = "hold" if first is None else "open"
    while probe is not None and len(observed) < budget:
        if probe not in outcomes:
            steps.append({"probe": probe, "outcome": None})
            status = "pending_receipt"
            break
        observed[probe] = outcomes[probe]
        hypotheses, dropped = M.update(registry, hypotheses, probe, outcomes[probe])
        steps.append({"probe": probe, "outcome": outcomes[probe], "dropped": dropped})
        status = M._status(registry, hypotheses, observed, budget - len(observed))
        if "unknown" in observed.values():
            status = "unidentifiable"
        if status != "open":
            break
        probe = _continue(registry, hypotheses, observed, continuation)
    if status == "open":
        status = "unidentifiable"
    return {"first_probe": first, "continuation": continuation, "steps": steps, "status": status,
            "conclusion": hypotheses[0] if status == "confirmed" else None, "probes_used": len(observed),
            "registered_cost_gpu_s": sum(registry["probes"][name]["cost_gpu_s"] for name in observed),
            "replay_kind": "historical_receipts_not_new_execution"}


def _lineage(protocol: dict, root: Path, entry: dict, stale: set) -> tuple[dict, list]:
    require(entry.get("phase") in PHASES and entry.get("role") in ROLES, "계보 phase·role이 다르다")
    seed, case = entry["seed"], entry["case"]
    reference = f"baseline_p0c_s{seed}"
    registry = V.C.registry()
    demo = _bound_json(protocol, root, entry["demo_result"])
    require(demo.get("fault") == entry["fault_family"] and demo.get("seed") == seed and demo.get("case") == case,
            "판정 파일의 결함·seed·case가 계보와 다르다")
    require(demo.get("mechanism") == entry["registered_mechanism"]
            and M._confirming(registry, entry["registered_mechanism"]) == entry["confirming_probe"],
            "등록 원인과 확인 검사가 registry와 다르다")
    gate, reports = demo["evaluation"]["gate"], demo["evaluation"]["reports"]
    require(gate.get("outcome") == "failure_present" and gate["fault_verdict"].get("label") == "unhealthy",
            "실패 존재 판정 파일이 아니다")
    checkpoints, action_scale = {}, None
    for role, name, key in (("case", case, "FAULT"), ("reference", reference, "NONE")):
        path = _bound_path(protocol, root, reports[key]["path"])
        require(file_sha256(path) == reports[key]["sha256_lf"], "고정 평가 보고서가 판정 파일 기록과 다르다")
        report = strict_json(path.read_bytes())
        require(report.get("name") == name, "고정 평가 보고서 정책이 계보 role과 다르다")
        checkpoints[role] = report["checkpoint"]["sha256"]
        action_scale = report["action_scale"] if role == "case" else action_scale
    require(action_scale == protocol["packet"]["action_scale"], "action_scale이 등록 task 설정과 다르다")
    summaries, telemetry_sha = {}, {}
    for role, name in (("case", case), ("reference", reference)):
        rel = entry["telemetry"][role]
        value = _bound_json(protocol, root, rel)
        require(isinstance(value.get("run_dir_name"), str) and value["run_dir_name"].endswith("_" + name),
                "학습 기록 실행 이름이 계보 role과 다르다")
        summaries[role], telemetry_sha[role] = value["summary"], protocol["bindings_sha256"][rel]
    facts = _facts(summaries, action_scale)
    receipts = _receipts(protocol, root, entry, demo, checkpoints["case"])
    media = [media_decision(protocol, root, item, checkpoints, stale) for item in entry.get("media", [])]
    bindings = {role: {"checkpoint_sha256": checkpoints[role], "telemetry_sha256": telemetry_sha[role]}
                for role in ("case", "reference")}
    packet = V.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None, facts=facts,
                              budget_left=protocol["packet"]["budget_left"],
                              execution_budget_s=protocol["packet"]["execution_budget_s"], input_bindings=bindings)
    with tempfile.TemporaryDirectory() as folder:
        snapshot = V.build_snapshot(packet, RUN_ID, protocol["packet"]["policy"], V.file_digest(V.PROTOCOL),
                                    Path(folder) / "probes.ledger.jsonl")
        messages = V.messages(snapshot)
    view = snapshot["view"]
    outcomes = {probe: row["outcome"] for probe, row in receipts.items()}
    rules = {policy: replay(registry, view["remaining_candidates"], V.rule_probe(view, policy), outcomes, policy,
                            view["budget_left"]) for policy in ("registry_next", "fixed_eligible")}
    capsule = {"file": f"capsules/{entry['lineage_id']}.numbers.messages.json",
               "sha256": hashlib.sha256(canonical(messages)).hexdigest()}
    accepted = {role: [row["media_id"] for row in media if row["accepted"] and row["role"] == role]
                for role in ("case", "reference")}
    blocked = [f"render_verified_{role}_frames_missing" for role in ("case", "reference") if not accepted[role]]
    blocked.append("v01_same_play_frame_contract_missing")
    pending = {"status": "pending_api", "modality": "training_numbers", "capsule_sha256": capsule["sha256"]}
    decision = demo["diagnosis"]["decision"]
    row = {
        "lineage_id": entry["lineage_id"], "phase": entry["phase"], "role": entry["role"],
        "fault_family": entry["fault_family"], "registered_mechanism": entry["registered_mechanism"],
        "confirming_probe": entry["confirming_probe"], "seed": seed, "case": case,
        "failure_label": {"label": "unhealthy", "source": entry["demo_result"],
                          "source_sha256": protocol["bindings_sha256"][entry["demo_result"]]},
        "checkpoints": checkpoints, "telemetry_sha256": telemetry_sha,
        "numbers_digest": L.digest(facts), "fact_ids": sorted(facts), "view_digest": snapshot["view_digest"],
        "remaining_candidates": list(view["remaining_candidates"]), "budget_left": view["budget_left"],
        "eligible_probes": list(view["eligible_probes"]), "capsule": capsule,
        "arms": {
            "A": {**pending, "model": protocol["models"]["A"], "if_model_unavailable": "not_evaluable"},
            "B": {**pending, "model": protocol["models"]["B"]},
            "C": {"status": "blocked_new_collection_required", "model": protocol["models"]["C"],
                  "modality": "same_numbers_plus_render_verified_frames", "capsule_sha256": None,
                  "reasons": blocked, "accepted_media": accepted},
            "D": {"status": "computed_cpu", "rules": rules},
        },
        "receipts": receipts,
        "missing_receipts": sorted(set(registry["probes"]) - set(receipts)),
        "media": media,
        "historical_text_decision": {"probe": decision["probe"], "ranking": decision["ranking"],
                                     "contract": "legacy_top3_diagnosis", "counted_as_arm_A": False},
    }
    return row, messages


def _alternative(protocol: dict, root: Path) -> list[dict]:
    """대안 기능의 기존 근거를 바이트에서 다시 읽는다. 라벨은 렌더 receipt·위치 감사에서만 온다."""
    rows = []
    for item in protocol["alternative"]["evidence"]:
        value, kind = _bound_json(protocol, root, item["path"]), item["kind"]
        if kind == "v9_window_metrics":
            for metric in value["rows"]:
                rows.append({"source": item["path"], "clip": metric["policy_id"], "window": metric["window_id"],
                             "body_speed_mps": metric["mean"]["body_horizontal_speed_norm"],
                             "command_mps": metric["mean"]["command_horizontal_norm"]})
        elif kind == "v9_render_followup":
            rows.append({"source": item["path"], "clip": "motor-s2033-none+motor-s2046-none",
                         "mask_rule": value["normal_policy_projection_separated"],
                         "model_frame_readiness": value["all_four_model_frame_readiness"]})
        elif kind == "v11_render_preflight":
            rows.append({"source": item["path"], "clip": value["media"], "mask_rule": value["projection"],
                         "render_sync_verified": value["checks"]["render_sync_verified"]})
        elif kind == "f1_fresh_render":
            for role, policy in value["policies"].items():
                rows.append({"source": item["path"], "clip": policy["video"]["path"], "role": role,
                             "render_sync_calls": policy["render_sync_calls"], "frames": policy["frames"],
                             "body_speed_mps": [policy["body_motion"]["w1.body_horizontal_speed_norm"],
                                                policy["body_motion"]["w2.body_horizontal_speed_norm"]]})
            rows.append({"source": item["path"], "clip": "reference_position_check",
                         "mask_rule": value["normal_render_position_diagnostic"],
                         "case_position_check": value["fault_position_diagnostic"]})
        elif kind == "render_audit":
            for count in value["position_counts"]:
                rows.append({"source": item["path"], "clip": count["source"], "mask_rule": count})
        else:
            raise ValueError("등록되지 않은 대안 근거 형식")
    return rows


def _comparison(lineages: list[dict]) -> dict:
    blocked = any(row["arms"]["C"]["status"].startswith("blocked") for row in lineages)
    return {"A_vs_B_model_change": "pending_api",
            "B_vs_C_image_contribution": "inconclusive_frames_blocked" if blocked else "pending_api",
            "D_registered_rules": "first_choice_computed_followup_per_receipts"}


def build(protocol: dict, root: Path = ROOT) -> tuple[dict, dict]:
    """사전등록 입력을 검증해 manifest와 숫자 입력 capsule을 메모리에서 만든다."""
    verify_protocol(protocol, root)
    audit = _bound_json(protocol, root, protocol["render_audit"])
    stale = {row["video_sha256"] for row in audit["native_source_hashes"].values()}
    require(bool(stale), "stale 렌더 영상 목록이 비었다")
    lineages, capsules, seen = [], {}, set()
    for entry in protocol["lineages"]:
        row, messages = _lineage(protocol, root, entry, stale)
        identity = (row["checkpoints"]["case"], row["telemetry_sha256"]["case"], row["numbers_digest"])
        require(not seen & set(identity), "두 계보가 같은 checkpoint·학습 기록·수치를 쓴다")
        seen |= set(identity)
        lineages.append(row)
        capsules[PurePosixPath(row["capsule"]["file"]).name] = messages
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "inputs_prepared_no_model_calls", "protocol_digest": L.digest(protocol),
        "selection_contract": {"module": "evals/selection_comparison_view_v1.py",
                               "protocol_sha256_lf": V.file_digest(V.PROTOCOL),
                               "registry_digest": M.registry_digest(V.C.registry())},
        "lineages": lineages, "comparison": _comparison(lineages), "outcome": "inconclusive",
        "alternative_evidence": _alternative(protocol, root),
        "external_calls": 0, "new_training": 0, "gpu_runs": 0, "claims": dict(CLAIMS),
    }
    manifest["manifest_digest"] = L.digest(manifest)
    return manifest, capsules


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)


def prepare(protocol_path: Path, output: Path, root: Path = ROOT) -> dict:
    manifest, capsules = build(load_protocol(protocol_path), root)
    output = Path(output)
    require(not output.exists(), "출력 폴더가 이미 있다")
    (output / "capsules").mkdir(parents=True)
    for name, messages in capsules.items():
        _write_new(output / "capsules" / name, canonical(messages))
    _write_new(output / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def check(protocol_path: Path, manifest_path: Path, root: Path = ROOT) -> dict:
    """현재 입력으로 다시 계산한 manifest·capsule이 준비본과 같아야 한다."""
    expected, capsules = build(load_protocol(protocol_path), root)
    manifest_path = Path(manifest_path)
    require(canonical(strict_json(manifest_path.read_bytes())) == canonical(expected),
            "준비된 manifest가 현재 입력과 다르다")
    folder = manifest_path.parent / "capsules"
    require(folder.is_dir() and sorted(path.name for path in folder.iterdir()) == sorted(capsules),
            "capsule 파일 목록이 다르다")
    for name, messages in capsules.items():
        require((folder / name).read_bytes() == canonical(messages), f"capsule이 현재 입력과 다르다: {name}")
    return expected


def _choice(lineage: dict, arm: str, value: dict) -> dict:
    spec = lineage["arms"][arm]
    require(spec["status"] == "pending_api", f"{arm} 입력이 준비되지 않아 결과를 받지 않는다")
    require(isinstance(value, dict) and value.get("capsule_sha256") == spec["capsule_sha256"],
            "다른 계보·조건의 capsule 응답이다")
    status = value.get("status")
    require(status in CHOICE_STATES, "선택 결과 상태가 다르다")
    result = {"status": status, "correct_confirmation": False, "false_confirmation": False, "hold": False,
              "followup": None, "citations": {"telemetry": 0, "frame": 0, "config": 0}}
    if status != "valid":
        result["hold"] = status in TERMINAL_STATES
        return result
    action, probe, facts = value.get("action"), value.get("probe"), value.get("fact_ids")
    require(action in ("continue", "hold") and isinstance(facts, list) and bool(facts)
            and len(set(facts)) == len(facts), "유효 선택 형식이 다르다")
    visible = set(lineage["fact_ids"]) | ({f"vision:frame:f0{i}" for i in range(1, 7)} if arm == "C" else set())
    require(set(facts) <= visible, "보이지 않은 근거를 인용했다")
    require((action == "hold" and probe is None) or (action == "continue" and probe in lineage["eligible_probes"]),
            "선택 probe가 계약과 다르다")
    for fact in facts:
        kind = "frame" if FRAME_FACT.fullmatch(fact) else fact.split(":", 1)[0]
        result["citations"][kind] = result["citations"].get(kind, 0) + 1
    outcomes = {name: row["outcome"] for name, row in lineage["receipts"].items()}
    followup = replay(V.C.registry(), lineage["remaining_candidates"], probe, outcomes, "registry_next",
                      lineage["budget_left"])
    result.update(hold=action == "hold", followup=followup,
                  correct_confirmation=action == "continue" and probe == lineage["confirming_probe"]
                  and outcomes.get(probe) == "abnormal",
                  false_confirmation=followup["status"] == "confirmed"
                  and followup["conclusion"] != lineage["registered_mechanism"])
    return result


def _axis(lineages: list[dict], results: dict, left: str, right: str, need_frame: bool) -> dict:
    rows = {}
    for lineage in lineages:
        pair = results.get(lineage["lineage_id"], {})
        a, b = pair.get(left), pair.get(right)
        if a is None or b is None or a["status"] not in TERMINAL_STATES or b["status"] not in TERMINAL_STATES:
            rows[lineage["lineage_id"]] = "incomplete"
        elif b["correct_confirmation"] and not a["correct_confirmation"]:
            cited = b["citations"]["frame"] > 0
            rows[lineage["lineage_id"]] = "right_better" if cited or not need_frame else "uncited_change"
        elif a["correct_confirmation"] and not b["correct_confirmation"]:
            rows[lineage["lineage_id"]] = "left_better"
        else:
            rows[lineage["lineage_id"]] = "same"
    cases = [row for row in lineages if row["role"] == "case_of_interest"]
    if any(value == "incomplete" for value in rows.values()) or len(cases) < 2:
        outcome = "inconclusive"
    elif any(value == "left_better" for value in rows.values()):
        outcome = "regression"
    elif sum(rows[row["lineage_id"]] == "right_better" for row in cases) >= 2:
        confirmed = all(row["phase"] == "confirmation_new_seed" for row in cases)
        outcome = "supported" if confirmed else "development_observation_only"
    else:
        outcome = "not_supported"
    return {"left": left, "right": right, "per_lineage": rows, "outcome": outcome}


def score(manifest: dict, choices: dict, protocol: dict) -> dict:
    """사전등록 규칙으로 분류한다. 선택 변화만으로는 개선으로 세지 않는다."""
    require(choices.get("schema") == CHOICES_SCHEMA and choices.get("manifest_digest") == manifest["manifest_digest"],
            "다른 manifest의 선택 결과다")
    kind = choices.get("measurement_kind")
    require(kind in ("synthetic_cpu", "registered_real_provider_run"), "측정 종류가 다르다")
    index = {row["lineage_id"]: row for row in manifest["lineages"]}
    results = {}
    for lineage_id, arms in choices["choices"].items():
        require(lineage_id in index and isinstance(arms, dict) and set(arms) <= set(MODEL_ARMS), "등록 밖 계보·조건")
        results[lineage_id] = {arm: _choice(index[lineage_id], arm, value) for arm, value in arms.items()}
    axes = {"model_change": _axis(manifest["lineages"], results, "A", "B", False),
            "image_contribution": _axis(manifest["lineages"], results, "B", "C", True)}
    wording = {name: protocol["outcome_wording"][name][axis["outcome"]] for name, axis in axes.items()}
    return {"schema": "nemotron35_value_cases_score_v1", "measurement_kind": kind, "results": results,
            "axes": axes, "wording": wording, "selection_change_alone_counted": False,
            "frame_citation_is_causal_confirmation": False,
            "claims": dict(CLAIMS) if kind == "synthetic_cpu" else {**CLAIMS, "evaluated_by_registered_rules": True}}


def describe(root: Path = ROOT) -> dict:
    """이미 본 학습 기록의 결함별 비율 범위다. 사후 기술이며 선택 규칙·임계값을 만들지 않는다."""
    families = {key: row["fault"] for key, row in
                strict_json((root / "bench/answer_key_p0c.json").read_bytes())["cases"].items()}
    ratios: dict = {}
    files = {}
    for path in sorted((root / "bench/runs_p0c").glob("h0*_s*.telemetry.json")):
        match = re.fullmatch(r"(h0\d)_s(\d+)\.telemetry\.json", path.name)
        reference = root / f"bench/runs_p0c/baseline_p0c_s{match.group(2)}.telemetry.json" if match else None
        if reference is None or not reference.is_file() or families.get(match.group(1)) not in ("F1", "F2", "F5"):
            continue
        files[path.relative_to(root).as_posix()] = file_sha256(path)
        files[reference.relative_to(root).as_posix()] = file_sha256(reference)
        case, ref = (strict_json(item.read_bytes())["summary"] for item in (path, reference))
        for name in V.TELEMETRY_FIELDS:
            a, b = case[name].get("mean_last_20pct"), ref[name].get("mean_last_20pct")
            if V.number(a) and V.number(b) and b != 0:
                ratios.setdefault(name, {}).setdefault(families[match.group(1)], []).append(a / b)
    fields = {}
    for name, groups in sorted(ratios.items()):
        if set(groups) != {"F1", "F2", "F5"}:
            continue
        ranges = {family: [min(values), max(values), len(values)] for family, values in sorted(groups.items())}
        low, high = ranges["F2"][:2]
        if all(high < ranges[other][0] or low > ranges[other][1] for other in ("F1", "F5")):
            fields[name] = ranges
    return {"schema": "nemotron35_seen_telemetry_ranges_v1", "ratio": "case mean_last_20pct / same-seed NONE",
            "f2_ranges_disjoint_from_f1_and_f5": fields, "input_sha256": files, "posthoc_descriptive": True,
            "already_seen_development": True, "threshold_created": False, "used_for_selection": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--prepare", "--check", "--score", "--describe"):
        mode.add_argument(flag, action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--choices", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            require(args.output is not None, "--output이 필요하다")
            manifest = prepare(args.protocol, args.output)
            result = {"status": "prepared", "outcome": manifest["outcome"], "manifest_digest": manifest["manifest_digest"]}
        elif args.check:
            if args.manifest is None:
                manifest, _ = build(load_protocol(args.protocol))
            else:
                manifest = check(args.protocol, args.manifest)
            result = {"status": "current", "outcome": manifest["outcome"], "manifest_digest": manifest["manifest_digest"]}
        elif args.score:
            require(args.manifest is not None and args.choices is not None, "--manifest와 --choices가 필요하다")
            manifest = check(args.protocol, args.manifest)
            result = score(manifest, strict_json(args.choices.read_bytes()), load_protocol(args.protocol))
        else:
            result = describe()
    except ValueError as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

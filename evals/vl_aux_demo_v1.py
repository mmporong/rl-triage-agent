"""시연 v4의 결정·회복 판정을 유지하고 Play 영상 관찰을 별도 기록한다."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import e2e_demo_v4 as E
import demo_correction_v4 as DC
import preregistered_seed_guard as SEEDS
import record_vl_aux_play_v1 as MEDIA
import vl_aux_observer_v1 as OBS
import vl_aux_worker_v1 as WORKER
import loop as LOOP

SEED, RESTART_SEED = 2080, 3080
TAG = "vl_aux_f1_s2080_20261016"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task23/runs"
PROTOCOL_DIR = ROOT / "bench/protocols/vl_aux_demo_v1"
PROTOCOL = PROTOCOL_DIR / "e2e_demo_v4_f1.json"
NEW_CODE = {
    "evals/vl_aux_demo_v1.py", "evals/vl_aux_observer_v1.py",
    "bench/demo_correction_vl_v1.py", "bench/record_vl_aux_play_v1.py", "bench/run_vl_aux_demo_v1.ps1",
    "tests/test_vl_aux_demo_v1.py", "tests/test_vl_aux_observer_v1.py", "tests/test_record_vl_aux_play_v1.py",
    "evals/preregistered_seed_guard.py", "evals/nemotron_bottleneck_vision_v2.py",
    "evals/nemotron_bottleneck_compare_v2.py", "evals/nemotron_endpoint_ledger_v3.py",
    "evals/nemotron_bottleneck_ledger_v2.py", "evals/nemotron_bottleneck_n2_v2.py",
    "evals/feedback_model_call.py", "bench/record_evidence_media.py", "bench/run_runtime_readback_smoke.py",
    "evals/vl_aux_worker_v1.py", "tests/test_vl_aux_worker_v1.py",
}
B4_RESULT = "evals/results/nemotron_endpoint_bottlenecks_20261016_v3/summary.json"
B4_PROTOCOL = "bench/protocols/nemotron_endpoint_compare_v3.json"
_FIXED = E.fixed
_ACTIVE = False


def auxiliary_contract() -> dict:
    return {
        "role": "auxiliary_only", "decision_input": False,
        "timing": "source 고정 평가 뒤 NONE·FAULT, 교정 고정 평가 뒤 실행한 각 교정 arm",
        "semantic_boundary": "관찰 응답은 진단·교정 packet, probe 분류, 회복 판정 함수에 넣지 않는다",
        "operational_boundary": "추가 녹화·호출 시간은 발생하며 GPU busy·정리 실패 시 후속 GPU 실행은 중단한다",
        "requested_model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "accepted_provider_models": ["private/nvidia/nemotron-3.5-super-120b-a12b"],
        "model_identity": "실측 반환 별칭 계약이며 동일 가중치를 확인한 것은 아니다",
        "frame_count": 6, "frame_sampling": "동결 B4의 시간구간 midpoint 6개",
        "response_contract": {key: ["yes", "no", "unclear"] for key in ("walking", "in_place", "falling")},
        "fixed_eval_label_in_prompt": False, "maximum_policies": 4,
        "api_cap": 8, "infra_retries_per_clip": 1, "semantic_retries": 0,
        "decision_api_cap": 8, "total_api_cap": 16,
        "expected_observations": {"NONE": "walking=yes", "FAULT": "in_place=yes", "corrected": "walking=yes"},
        "agreement_meaning": "Play 행동 관찰과 정책 고정 평가 라벨의 대응이며 영상 행동의 정답률이 아니다",
        "completion": "실행한 고정 평가 정책 모두에 영상·6프레임·유효 관찰이 있으면 auxiliary_complete",
        "success": "기존 demo_success와 auxiliary_complete를 별도 표시하며 회복은 고정 평가만으로 판정한다",
        "wording": {
            "complete": "고정 평가 결과와 같은 checkpoint의 Play 프레임 관찰을 보조 기록으로 연결했다.",
            "incomplete": "회복 판정은 고정 평가 결과이며 영상 보조 관찰은 일부 또는 전부 미완료다.",
            "not_recovered": "고정 평가에서 회복 조건을 통과하지 못했다. 영상 관찰은 별도 보조 기록이다.",
        },
        "design_coupling": "기존 F1 시연과 B4 18영상 결과를 본 뒤 연결 기능을 설계한 새 seed 한 번의 통합 확인이다",
    }


def fixed(variant: str, decision_path: str) -> dict:
    value = deepcopy(_FIXED(variant, decision_path))
    value.update({
        "schema": "e2e_demo_protocol_v4_vl_aux_v1",
        "question": "새 seed 보상 결함 시연의 고정 평가 뒤 Play 영상 관찰을 판정 외 보조 근거로 연결할 수 있는가",
        "seed_selection": "2080·3080을 실행 전 로컬 기록에서 다시 검사하며 충돌하면 실행하지 않는다",
        "auxiliary": auxiliary_contract(),
        "changes_from_v3_1": ["시연 v4 피드백 경로 유지", "F1 새 seed 2080·restart 3080", "판정 외 영상 보조 관찰 추가"],
    })
    value["correction"]["executor"] = "bench/demo_correction_vl_v1.py (동결 v4 실행기 재사용)"
    value["correction"]["approval"] = "등록 범위의 운영자 위임 승인·소비, 승인 문자열 신원 인증 아님"
    value["correction"]["auto_approval"] = "execute의 --approve-registered-actions 명시가 필요하다"
    return value


class OwnedProxy:
    """교정 자식 진입점 하나만 새 adapter로 바꾸고 나머지는 원형 전달한다."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run_owned(self, folder, argv, cwd, timeout):
        args = list(argv)
        if len(args) > 1 and Path(args[1]).resolve() == (ROOT / "bench/demo_correction_v4.py").resolve():
            args[1] = str(ROOT / "bench/demo_correction_vl_v1.py")
        return self.original.run_owned(folder, args, cwd, timeout)


@contextmanager
def configured():
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("VL adapter context cannot be nested")
    _ACTIVE = True
    values = {name: getattr(E, name) for name in ("VARIANTS", "PRIVATE", "PROTOCOL_DIR", "CODE", "fixed",
                                                  "PREDICTIONS", "dynamic_fields", "correction_commands")}
    original_p = DC.P
    base_dynamic = E.dynamic_fields
    try:
        E.VARIANTS = {"f1": {"fault": "F1", "case_id": "h02", "mechanism": "reward", "seed": SEED}}
        E.PRIVATE, E.PROTOCOL_DIR = PRIVATE, PROTOCOL_DIR
        E.CODE = values["CODE"] | NEW_CODE | {B4_RESULT, B4_PROTOCOL}
        E.fixed = fixed
        E.PREDICTIONS = {"f1": {
            "source_failure": "F1이 같은 seed NONE 대비 unhealthy이고 정지 비율 ≥ 0.9",
            "alarm": "R1 경고가 F1에 울리고 NONE에는 울리지 않는다",
            "diagnosis": "첫 판단 continue, 상위 세 가설에 reward 포함",
            "loop": "P_reward abnormal 뒤 reward 확정", "specificity": "게이트 없음",
            "correction_rule": "기존 reward 규칙: 정지 비율 ≥ 0.9면 restart, 그렇지 않으면 resume",
            "correction": "교정 정책 recovered",
        }}
        def dynamic(decision_path, isaaclab):
            result = base_dynamic(decision_path, isaaclab)
            # 기존 dynamic key 집합을 유지하고 새 녹화 계약은 고정 계약의 일부로 넣는다.
            result["source_files"] = {name: E.sha(ROOT / name, True) for name in sorted(E.CODE)}
            return result
        E.dynamic_fields = dynamic
        E.correction_commands = lambda *_: ["등록 실행은 bench/run_vl_aux_demo_v1.ps1 -Execute 한 번으로 수행한다"]
        DC.P = OwnedProxy(original_p)
        yield E
    finally:
        for name, value in values.items():
            setattr(E, name, value)
        DC.P = original_p
        _ACTIVE = False


def protocol_data(isaaclab: Path) -> dict:
    with configured():
        value = {**E.fixed("f1", "single_feedback"), **E.dynamic_fields("single_feedback", isaaclab)}
    value["auxiliary"]["recording"] = MEDIA.build_recording_contract(isaaclab)
    return value


def check(isaaclab: Path, committed: bool = False, experiment: str | None = None) -> dict:
    value = E.read(PROTOCOL)
    if value != protocol_data(isaaclab):
        raise ValueError("VL preregistration or bound sources differ")
    with configured():
        bound = E.CODE | {PROTOCOL.relative_to(ROOT).as_posix(), E._rel(E.FIXED_EVAL), E._rel(E.BANDS_SOURCE),
                          E._rel(E.V4_PROTOCOL), E._rel(E.FEEDBACK_SUMMARY),
                          *(f"{E.RUNS_DIR}/{name}.telemetry.json" for name in E.REFERENCE_RUNS)}
        commit = E.P.verify_committed(ROOT, bound) if committed else None
        if committed and experiment:
            commit = E.V2.frozen_commit(experiment, commit, bound)
        if E.S._git_head(isaaclab) != value["isaaclab_commit"] or not E.S._git_tracked_clean(isaaclab):
            raise ValueError("Isaac checkout differs")
    return {"status": "pass", "committed": committed, "commit": commit, "protocol_sha256": E.sha(PROTOCOL, True)}


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    if not path.drive or len(path.drive) != 2:
        raise ValueError("WSL bridge requires a local Windows drive path")
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def decision_call(kind: str, packet: Path, output: Path) -> dict:
    if kind not in {"diagnosis", "correction"} or output.exists():
        raise ValueError("invalid or previously used model call")
    E.write_new(output.with_suffix(".reserved.json"), {"kind": kind, "maximum_http_requests": 2,
                                                        "packet_sha256": E.sha(packet), "model": E.MODEL})
    WORKER.invoke_worker(kind, packet, output)
    record = E.read(output)
    if type(record.get("requests")) is not int or not 1 <= record["requests"] <= 2:
        raise ValueError("decision worker request count differs")
    return record


def observe(clip_id: str, artifacts: dict, report: Path, folder: Path, isaaclab: Path, protocol: dict,
            *, arm: str, label: str) -> dict:
    stage = folder / "auxiliary" / clip_id
    result = {"clip_id": clip_id, "arm": arm, "fixed_evaluation_label": label,
              "used_for_decision": False, "fixed_report_sha256": E.sha(report),
              "checkpoint_sha256": artifacts["checkpoint"]["sha256"]}
    try:
        selected = {key: {name: artifacts[key][name] for name in ("path", "sha256")}
                    for key in ("checkpoint", "environment", "agent")}
        media = MEDIA.record_clip(selected, report, stage / "play", isaaclab,
                                 {**protocol["auxiliary"]["recording"], "fixed_evaluation_label": label})
    except Exception as error:
        # 녹화 실패는 GPU 정리 상태를 별도로 확인해야 하므로 후속 GPU 작업을 시작하지 않는다.
        E.write_new(stage / "result.json", {**result, "status": "recording_error", "error_type": type(error).__name__})
        raise
    result["media_manifest_sha256"] = E.sha(Path(media["manifest_path"]))
    try:
        packet = OBS.prepare(Path(media["video_path"]), stage / "frames",
                             {"checkpoint_sha256": result["checkpoint_sha256"],
                              "fixed_eval_sha256": result["fixed_report_sha256"]})
        observation = OBS.invoke(packet, stage / "observation.json", folder / "auxiliary_api")
        result.update(status=observation["status"], observation_sha256=E.sha(stage / "observation.json"))
    except Exception as error:
        result.update(status="observer_error", error_type=type(error).__name__)
    if E.sha(report) != result["fixed_report_sha256"]:
        raise ValueError("fixed evaluation changed during observation")
    E.write_new(stage / "result.json", result)
    return result


def _corrected_artifacts(folder: Path, arm: str) -> dict:
    receipt = E.read(folder / "fix" / arm / "train/receipt.json")
    return {"checkpoint": receipt["checkpoint"], **receipt["configs"]}


def pipeline(tag: str, isaaclab: Path, approver: str, preflight: dict | None = None) -> dict:
    """configured context에서 실행한다. 관찰 반환값은 판정 호출에 전달하지 않는다."""
    E.begin(tag, "f1", isaaclab)
    folder = E.folder_for(tag)
    if preflight is not None:
        E.write_new(folder / "entry_preflight.json", preflight)
    for arm in ("NONE", "FAULT"):
        E.train(tag, arm, isaaclab)
    E.alarm(tag, isaaclab)
    evaluated = E.evaluate(tag, isaaclab)
    for index, arm in enumerate(("NONE", "FAULT"), 1):
        label = ("healthy" if evaluated["gate"]["none_walks"] else "unknown") if arm == "NONE" else \
            evaluated["gate"]["fault_verdict"]["label"]
        observe(f"clip{index}", E._check_artifacts(tag, arm), ROOT / evaluated["reports"][arm]["path"],
                folder, isaaclab, E.read(PROTOCOL), arm=arm, label=label)
    if (folder / "terminal.json").exists():
        return E.summarize(tag)
    E.reference(tag, isaaclab)
    E.packet(tag, isaaclab)
    for attempt in (1, 2):
        record = folder / "model" / f"diagnosis.feedback{attempt}.json"
        decision_call("diagnosis", folder / "model/diagnosis.packet.json", record)
        outcome = E.loop_start(tag, isaaclab, record)
        if outcome.get("status") != "retry_allowed":
            break
    if (folder / "terminal.json").exists():
        return E.summarize(tag)
    case = E.ctx_for(tag).cases["FAULT"]
    for _ in range(E.LOOP_BUDGET):
        _, state, ledger = LOOP._load(case)
        if state["status"] != "open" or state["pending"] is None:
            break
        rid = state["pending"]
        ledger.approve(rid, approver)
        LOOP.run(case, rid, lambda c, p: LOOP.isaac_execute(c, p, f"{tag}_loop", metered=True),
                 preflight=lambda c, p: LOOP.isaac_preflight(c, p, f"{tag}_loop"))
    E.loop_close(tag, isaaclab)
    if (folder / "terminal.json").exists():
        return E.summarize(tag)
    E.specificity(tag, isaaclab)
    E.correction_evidence(tag, isaaclab)
    for attempt in (1, 2):
        record = folder / "model" / f"correction.feedback{attempt}.json"
        saved = decision_call("correction", folder / "model/correction.packet.json", record)
        if not saved.get("infra_error"):
            break
    proposed = E.correction_propose(tag, isaaclab, record)
    rid = proposed["request_id"]
    E.approve_correction(tag, rid, approver)
    corrected = E.correction(tag, isaaclab, rid, record)
    base_result = E.summarize(tag)
    fix = corrected.get("result")
    if fix:
        for index, arm in enumerate(fix["arms"], 3):
            if arm == fix["treatment"]:
                label = (fix.get("verdict") or {}).get("label", "unknown")
            else:
                label = fix["reference"]["against_original_none"]["label"]
            observe(f"clip{index}", _corrected_artifacts(folder, arm), folder / "fix/eval" / f"{arm}.json",
                    folder, isaaclab, E.read(PROTOCOL), arm=arm, label=label)
    return base_result


def output_preflight(tag: str) -> dict:
    folder = E.folder_for(tag)
    paths = [folder, folder / "auxiliary", folder / "auxiliary_api", folder / "model", folder / "vl_aux_result.json",
             *(E.RESULTS / (tag + suffix) for suffix in ("", "_ref", "_loop", "_spec")),
             LOOP._dir(f"h02_s{SEED}")]
    if any(path.exists() for path in paths):
        raise FileExistsError("demo, sidecar, ledger or result output already exists")
    return {"absent_outputs": [path.relative_to(ROOT).as_posix() for path in paths]}


def runtime_preflight(isaaclab: Path) -> dict:
    for executable in ("ffmpeg", "ffprobe", "wsl", "pwsh"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"required executable missing: {executable}")
    if not (isaaclab / "_isaac_sim/python.bat").is_file():
        raise FileNotFoundError("Isaac Python launcher missing")
    code = "import os,openai; assert os.environ.get('NVIDIA_API_KEY'), 'NVIDIA credential unavailable'"
    command = ('. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.quote(code))
    completed = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                               capture_output=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError("WSL credential/package preflight failed; no HTTP request sent")
    return {"executables_present": True, "wsl_credential_and_package_present": True, "api_requests": 0}


def execute(tag: str, isaaclab: Path, approver: str, approved: bool) -> dict:
    if not approved or not approver.strip():
        raise ValueError("registered-action operator approval is required")
    check(isaaclab, committed=True)
    fresh = SEEDS.assert_fresh_seeds([SEED, RESTART_SEED], isaaclab)
    with configured():
        folder = E.folder_for(tag)
        preflight = {**output_preflight(tag), "seed_freshness": fresh, "runtime": runtime_preflight(isaaclab)}
        # begin은 프로토콜의 추가 녹화 계약을 포함한 검증 함수를 사용한다.
        original_verify = E.verify_protocol
        E.verify_protocol = lambda variant, lab, experiment=None: (
            E.read(PROTOCOL), check_active(lab, experiment)["commit"])
        try:
            result = pipeline(tag, isaaclab, approver, preflight)
        except Exception as error:
            E.write_new(folder / "execution_failure.json", {"error_type": type(error).__name__,
                                                           "base_result_exists": (folder / "result.json").is_file(),
                                                           "base_verdict_changed_by_observer": False})
            raise
        finally:
            E.verify_protocol = original_verify
        auxiliary = [E.read(path) for path in sorted((folder / "auxiliary").glob("clip*/result.json"))]
        expected = 2 + len((E.step_record(tag, "correction").get("result") or {}).get("arms", [])) \
            if E.step_path(tag, "correction").is_file() else 2
        complete = len(auxiliary) == expected and all(row["status"] == "valid" for row in auxiliary)
        decision_records = [E.read(path) for path in sorted((folder / "model").glob("*.feedback[12].json"))]
        decision_requests = sum(row["requests"] for row in decision_records)
        observer_ledger = OBS.RequestLedger(folder / "auxiliary_api").audit()
        observer_requests = len(observer_ledger)
        if decision_requests > 8 or observer_requests > 8:
            raise ValueError("registered API budget exceeded")
        report = {"schema": "vl_aux_demo_result_v1", "base_result_sha256": E.sha(folder / "result.json"),
                  "outcome": result["outcome"], "demo_success": result["demo_success"],
                  "auxiliary_complete": complete, "expected_policies": expected, "auxiliary": auxiliary,
                  "used_for_decision": False, "seed_freshness": fresh,
                  "api_requests": {"decision": decision_requests, "observer": observer_requests,
                                   "total": decision_requests + observer_requests},
                  "wording": auxiliary_contract()["wording"]["complete" if complete else "incomplete"]}
        E.write_new(folder / "vl_aux_result.json", report)
        return report


def check_active(isaaclab: Path, experiment=None) -> dict:
    """이미 구성한 context에서 재진입 없이 실행·자식 학습의 결합을 확인한다."""
    protocol = E.read(PROTOCOL)
    expected = {**E.fixed("f1", "single_feedback"), **E.dynamic_fields("single_feedback", isaaclab)}
    expected["auxiliary"]["recording"] = MEDIA.build_recording_contract(isaaclab)
    if protocol != expected:
        raise ValueError("active VL source binding differs")
    bound = E.CODE | {PROTOCOL.relative_to(ROOT).as_posix(), E._rel(E.FIXED_EVAL), E._rel(E.BANDS_SOURCE),
                      E._rel(E.V4_PROTOCOL), E._rel(E.FEEDBACK_SUMMARY),
                      *(f"{E.RUNS_DIR}/{name}.telemetry.json" for name in E.REFERENCE_RUNS)}
    head = E.P.verify_committed(ROOT, bound)
    if E.S._git_head(isaaclab) != protocol["isaaclab_commit"] or not E.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac checkout differs")
    return {"commit": head if experiment is None else E.V2.frozen_commit(experiment, head, bound)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--approver", default="claude-main-user-delegated")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    lab = args.isaaclab.resolve()
    if args.command == "prepare":
        E.write_new(PROTOCOL, protocol_data(lab))
        result = {"status": "prepared_commit_before_execute", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}
    elif args.command in {"check", "verify"}:
        result = check(lab, committed=args.command == "verify")
    else:
        result = execute(args.tag, lab, args.approver, args.approve_registered_actions)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

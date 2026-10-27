"""비교 시작 뒤 준비한 개발 E2E. 고정 target과 비교 전 capsule commit을 검증한다.

Native CORRECTED recorder: 두 교정 완료 뒤 bind_protocol 한 번, bound protocol
메인 커밋과 fresh resource proof 뒤 NativeCorrectedRecordService(protocol).record(
resource_proof=Path, resource_sha256=str, isaaclab=Path, python=Path, stage=Path,
logical_id='corrected-s2220'/'corrected-s2221', correction_result=Path,
approval_receipt=Path, receipt_output=Path, normal_reference=None)를 호출한다.
반환은 {status:'complete', job:str, native_receipt:{path,sha256,bytes}}이고 원문
schema는 vision_corrected_native_record_receipt_v1이다. 아래 export/bind/observe가
실제 API에 연결되며 Writer·from_runtime·legacy top-level result를 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import struct
import sys
import zlib
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    sys.path.insert(0, str(folder))

import grounded_vision_observer_v1 as OBSERVER
import model_decision_journal_v1 as JOURNAL
import vision_brev_training_backend_v2 as TRAINING
import vision_selection_brev_adapter_v1 as ADAPTER
import vision_selection_call_v1 as CALL
import vision_selection_e2e_v1 as CONTRACT
import vision_transferred_input_acceptance_v2 as INPUTS
import vision_transferred_selection_v1 as SELECTION

from rl_triage.correction_ledger import CorrectionLedger

SCHEMA = "vision_transferred_e2e_development_blueprint_v2"
SOURCE_SCHEMA = "vision_transferred_e2e_source_cache_v2"
SCOPE_SCHEMA = "vision_transferred_e2e_host_use_v2"
SCOPE = "post_comparison_development_fixed_transferred_v05_e2e_native_post_observation"
METHOD = ROOT / "bench/protocols/vision_transferred_e2e_v2.json"
SOURCE = ROOT / "bench/protocols/vision_brev_source_20261026_v8.json"
SUMMARY = ROOT / "evals/results/vision_source_prefix_dev_20261026x/summary.json"
SEEDS = {"source-01": 2220, "source-02": 2221}
COMPARISON_REGISTRATION = "8dddefba25bcd382cc67124faf190f2e26b70ad9"
DEVELOPMENT = "post_comparison_development_no_effect_or_generalization_claim"


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": ADAPTER.file_sha(path)}


def read_ref(value: Mapping[str, Any]) -> tuple[Path, dict]:
    path = Path(str(value["path"])).resolve(strict=True)
    require(ADAPTER.file_sha(path) == value["sha256"], "참조 원문 SHA 불일치")
    return path, ADAPTER._read(path)


class ObjectEvidence:
    """보관 commit/tree/blob의 읽기 전용 SHA-1 검증. Git 프로세스를 실행하지 않는다."""

    def __init__(self, repo: Path):
        metadata = repo / ".git"
        if metadata.is_file():
            metadata = (repo / metadata.read_text().strip().removeprefix("gitdir: ")).resolve()
        common = metadata / "commondir"
        self.directory = (metadata / common.read_text().strip()).resolve() if common.exists() else metadata
        self.cache: dict[str, tuple[str, bytes]] = {}
        self.packed: dict[str, tuple[Path, int]] = {}
        for idx in (self.directory / "objects/pack").glob("*.idx"):
            raw = idx.read_bytes()
            require(raw[:8] == b"\xfftOc\x00\x00\x00\x02", "미지원 보관 object index")
            count = struct.unpack_from(">I", raw, 8 + 255 * 4)[0]
            ids, offsets, large = 8 + 1024, 8 + 1024 + count * 24, 8 + 1024 + count * 28
            for n in range(count):
                oid = raw[ids + n * 20:ids + (n + 1) * 20].hex()
                offset = struct.unpack_from(">I", raw, offsets + n * 4)[0]
                if offset & 0x80000000:
                    offset = struct.unpack_from(">Q", raw, large + (offset & 0x7fffffff) * 8)[0]
                self.packed[oid] = (idx.with_suffix(".pack"), offset)

    @staticmethod
    def delta(base: bytes, instructions: bytes) -> bytes:
        cursor = 0
        def number() -> int:
            nonlocal cursor
            value, shift = 0, 0
            while True:
                byte = instructions[cursor]; cursor += 1
                value |= (byte & 127) << shift
                if not byte & 128: return value
                shift += 7
        require(number() == len(base), "delta 원문 크기 불일치")
        expected, result = number(), bytearray()
        while cursor < len(instructions):
            op = instructions[cursor]; cursor += 1
            if op & 128:
                offset, size = 0, 0
                for n in range(4):
                    if op & (1 << n): offset |= instructions[cursor] << (8 * n); cursor += 1
                for n in range(3):
                    if op & (1 << (n + 4)): size |= instructions[cursor] << (8 * n); cursor += 1
                size = size or 0x10000
                require(offset + size <= len(base), "delta 복사 경계 초과")
                result.extend(base[offset:offset + size])
            else:
                require(op != 0 and cursor + op <= len(instructions), "delta 삽입 경계 오류")
                result.extend(instructions[cursor:cursor + op]); cursor += op
        require(len(result) == expected, "delta 결과 크기 불일치")
        return bytes(result)

    def pack_object(self, path: Path, offset: int) -> tuple[str, bytes]:
        with path.open("rb") as stream:
            stream.seek(offset)
            byte = stream.read(1)[0]
            kind, size, shift = (byte >> 4) & 7, byte & 15, 4
            while byte & 128:
                byte = stream.read(1)[0]; size |= (byte & 127) << shift; shift += 7
            base = None
            if kind == 6:
                byte = stream.read(1)[0]; distance = byte & 127
                while byte & 128:
                    byte = stream.read(1)[0]; distance = ((distance + 1) << 7) | (byte & 127)
                base = self.pack_object(path, offset - distance)
            elif kind == 7:
                base = self.object(stream.read(20).hex())
            decoder, chunks = zlib.decompressobj(), []
            while not decoder.eof:
                block = stream.read(65536)
                require(bool(block), "보관 pack 조기 종료")
                chunks.append(decoder.decompress(block))
            body = b"".join(chunks)
        require(len(body) == size, "pack 원문 크기 불일치")
        if base is not None: return base[0], self.delta(base[1], body)
        require(kind in {1, 2, 3, 4}, "보관 object 종류 오류")
        return {1: "commit", 2: "tree", 3: "blob", 4: "tag"}[kind], body

    def object(self, oid: str) -> tuple[str, bytes]:
        require(len(oid) == 40 and all(c in "0123456789abcdef" for c in oid), "object SHA 형식 오류")
        if oid not in self.cache:
            loose = self.directory / "objects" / oid[:2] / oid[2:]
            if loose.exists():
                header, body = zlib.decompress(loose.read_bytes()).split(b"\0", 1)
                kind, size = header.decode().split(" ")
                require(int(size) == len(body), "loose object 크기 오류")
            else:
                require(oid in self.packed, "요청한 보관 commit/blob가 없음")
                kind, body = self.pack_object(*self.packed[oid])
            require(hashlib.sha1(f"{kind} {len(body)}".encode() + b"\0" + body).hexdigest() == oid,
                    "보관 object hash 불일치")
            self.cache[oid] = kind, body
        return self.cache[oid]

    def commit(self, oid: str) -> dict:
        kind, body = self.object(oid)
        require(kind == "commit", "commit object가 아님")
        fields = body.decode().split("\n\n", 1)[0].splitlines()
        return {"tree": next(s[5:] for s in fields if s.startswith("tree ")),
                "parents": [s[7:] for s in fields if s.startswith("parent ")],
                "timestamp": int(next(s for s in fields if s.startswith("committer ")).rsplit(" ", 2)[1])}

    def file(self, commit: str, relative: str) -> tuple[str, bytes]:
        tree = self.commit(commit)["tree"]
        for part in Path(relative).parts:
            require(part not in {"..", ".", "/"}, "보관 경로 형식 오류")
            kind, body = self.object(tree)
            require(kind == "tree", "중간 경로가 tree가 아님")
            cursor, found = 0, None
            while cursor < len(body):
                end = body.index(b"\0", cursor)
                mode, name = body[cursor:end].split(b" ", 1)
                oid = body[end + 1:end + 21].hex(); cursor = end + 21
                if name.decode() == part:
                    require(mode in {b"40000", b"100644", b"100755"}, "보관 경로 symlink/gitlink 거부")
                    found = oid; break
            if found is None: raise FileNotFoundError(relative)
            tree = found
        kind, body = self.object(tree)
        require(kind == "blob", "파일 object가 blob이 아님")
        return tree, body


def method_targets(value: dict) -> list[dict]:
    slots = {r["slot_id"]: r for r in value["inputs"]["slots"]}
    return [{**{k: slots[t["slot_id"]][k] for k in ("slot_id", "seed_group", "source_role", "window")},
             "view_kind": "text_and_video"} for t in CONTRACT.TARGETS]


def target_proof(repo: Path, protocol_path: Path, locations_path: Path) -> dict:
    """답변 본문을 읽지 않고 비교 전 등록의 target·실제 capsule 동일성을 검증한다."""
    store = ObjectEvidence(repo)
    relative = protocol_path.resolve().relative_to(repo.resolve()).as_posix()
    blob, raw = store.file(COMPARISON_REGISTRATION, relative)
    require(raw == protocol_path.read_bytes(), "comparison 사전등록 원문과 현재 protocol 불일치")
    protocol = json.loads(raw)
    method_path = protocol["method_protocol"]["path"]
    method_blob, method_raw = store.file(COMPARISON_REGISTRATION, method_path)
    require(method_targets(json.loads(method_raw)) == list(CONTRACT.TARGETS), "사전 고정 target 불일치")
    earliest, cursor = COMPARISON_REGISTRATION, COMPARISON_REGISTRATION
    while True:
        parents = store.commit(cursor)["parents"]
        if not parents: break
        parent = parents[0]
        try: _, previous = store.file(parent, method_path)
        except FileNotFoundError: break
        require(method_targets(json.loads(previous)) == list(CONTRACT.TARGETS), "원래 method의 target이 달랐음")
        earliest, cursor = parent, parent
    legacy_path = "evals/vision_selection_e2e_v1.py"
    legacy_blob, legacy_raw = store.file(COMPARISON_REGISTRATION, legacy_path)
    assignments = [n for n in ast.parse(legacy_raw).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "TARGETS" for t in n.targets)]
    require(len(assignments) == 1 and list(ast.literal_eval(assignments[0].value)) == list(CONTRACT.TARGETS),
            "비교 전 동결 E2E target literal 불일치")
    api = repo / ".triage-runtime/vision-selection" / protocol["execution_id"] / "api/reservations"
    first_path = min(api.glob("*.json"))
    first = ADAPTER._read(first_path)
    first_time = ADAPTER._time(first["reserved_at"], "comparison first call")
    require(store.commit(COMPARISON_REGISTRATION)["timestamp"] <= first_time.timestamp(),
            "comparison 사전등록이 첫 호출보다 늦음")
    locations = ADAPTER._read(locations_path)
    by_call = {r["call_id"]: r for r in locations["rows"]}
    rows = CONTRACT._target_rows(protocol)
    checks = []
    for target, row in zip(CONTRACT.TARGETS, rows, strict=True):
        capsule = by_call[row["call_id"]]["capsule"]
        frozen = CALL.prepared(capsule)
        require(capsule["sha256"] == row["capsule_sha256"] and
                frozen["checkpoint_sha256"] == row["checkpoint_sha256"], "사전등록 capsule/checkpoint 불일치")
        checks.append({"target": dict(target), "call_id": row["call_id"], "seed": row["source_seed"],
                       "checkpoint_sha256": row["checkpoint_sha256"], "capsule_sha256": capsule["sha256"]})
    return {"schema": "vision_transferred_e2e_target_proof_v2", "development_scope": DEVELOPMENT,
            "original_method_first_parent_commit": earliest, "comparison_registration": COMPARISON_REGISTRATION,
            "original_method_blob": store.file(earliest, method_path)[0],
            "comparison_protocol_blob": blob, "method_blob": method_blob, "legacy_target_blob": legacy_blob,
            "first_api_reservation": ref(first_path), "fixed_targets": checks,
            "comparison_response_contents_read": False, "git_commands_executed": 0}


def verify_registration(repo: Path, context: dict, scope: dict) -> None:
    """외부 등록 commit에 묶인 실제 method·실행 protocol·소스만 수용한다."""
    anchor = scope["execution_registration_anchor"]
    store = ObjectEvidence(repo)
    store.commit(anchor)
    references = {METHOD.relative_to(ROOT).as_posix(): ADAPTER.file_sha(METHOD, lf=True)}
    execution_path = context["protocol_reference"]["path"]
    if execution_path.startswith("bench/protocols/"):
        public_registration(context["protocol"])
        _, committed = store.file(anchor, execution_path)
        require(committed == Path(context["protocol_path"]).read_bytes(), "공개 E2E execution protocol commit 불일치")
    else:
        registered = verify_public_seal(repo, scope["execution_registration_ref"], anchor)
        source_plan = repo / context["protocol"]["plan_references"]["source"]["path"]
        require(registered == execution_registration(Path(context["protocol_path"]), source_plan),
                "공개 E2E seal과 private 실행 protocol/plan 불일치")
    expected = set(validate_method()["required_development_sources"])
    if scope.get("model_relay_config"):
        _, config = read_ref(scope["model_relay_config"])
        expected.add(config["source"]["path"])
    bindings = context["protocol"]["execution_bindings"]
    expected.update(bindings[n]["path"] for n in ("correction_backend", "record_backend", "record_verifier"))
    sources = scope.get("development_sources_lf", {})
    require(set(sources) == expected, "개발 실행 source 등록 목록 불일치")
    references.update(sources)
    for relative, sha in references.items():
        require(ADAPTER.file_sha(repo / relative, lf=True) == sha, "등록 source가 현재 파일과 다름")
        _, body = store.file(anchor, relative)
        require(hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest() == sha,
                "개발 실행 사전등록 commit 원문과 다름")


class RelayPending(BaseException):
    """Journal의 provider 예외 정규화를 통과해 미완료 reservation을 보존한다."""


class RelayComplete:
    """같은 callback과 증가 sequence를 사용하며 retry를 추가하지 않는다."""

    def __init__(self, config: dict, allocation_root: Path):
        import vision_model_spool_v1 as spool
        require(set(config) == {"schema", "spool", "job_id", "source", "timeout_s", "poll_interval_s"}
                and config["schema"] == "vision_transferred_e2e_relay_config_v2", "relay config 형식 불일치")
        registered = ADAPTER._file_ref(ROOT, {"path": config["source"]["path"],
            "sha256": config["source"]["sha256_lf"]}, "model spool source", lf=True)
        require(Path(inspect.getsourcefile(spool.remote_complete)).resolve() == registered,
                "relay callback source가 등록 파일과 다름")
        require(spool.JOB_ID.fullmatch(config["job_id"]) is not None and Path(config["spool"]).is_absolute(),
                "relay job ID/spool 경로 불일치")
        self.config, self.path, self.spool = config, allocation_root, spool

    def __call__(self, messages: list[dict], params: dict) -> dict:
        require(params == OBSERVER.REQUEST_PARAMS,
                "relay params가 OBSERVER.REQUEST_PARAMS와 다름; 교정 proposal params를 변환하지 않음")
        claims = self.path / "allocations"
        claims.mkdir(parents=True, exist_ok=True)
        rows = sorted(claims.glob("*.json"))
        require([p.name for p in rows] == [f"{i:04d}.json" for i in range(1, len(rows) + 1)],
                "relay sequence 원장 불연속")
        if rows and not (self.path / "completed" / rows[-1].name).exists():
            raise RelayPending("기존 relay 요청이 미완료라 새 sequence를 발급하지 않음")
        sequence = len(rows) + 1
        require(sequence <= 32, "E2E relay 32 요청 상한")
        claim = {"job_id": self.config["job_id"], "sequence": sequence,
            "messages_sha256": JOURNAL.digest(messages), "params_sha256": JOURNAL.digest(params)}
        ADAPTER._write_new(claims / f"{sequence:04d}.json", claim)
        try:
            result = self.spool.remote_complete(messages, params, spool=Path(self.config["spool"]),
                job_id=self.config["job_id"], sequence=sequence, timeout_s=self.config["timeout_s"],
                poll_interval_s=self.config["poll_interval_s"])
        except self.spool.SpoolPendingError:
            raise RelayPending("spool 응답 미완료; Journal pending 예약과 요청 원문을 보존") from None
        except self.spool.SpoolError as error:
            raise RelayPending(type(error).__name__ + ": relay 결합 오류; 새 요청을 발급하지 않음") from None
        ADAPTER._write_new(self.path / "completed" / f"{sequence:04d}.json", {
            **claim, "provider_result_sha256": JOURNAL.digest(result)})
        return result


def native_components() -> tuple[Any, Any]:
    import run_vision_corrected_pull_record_v1 as runner
    import vision_corrected_record_service_v1 as service
    return runner, service


def public_registration(value: dict) -> None:
    """공개 seal에 사용자·private 절대경로가 포함되지 않는지 검사한다."""
    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items(): walk(key); walk(child)
        elif isinstance(item, list):
            for child in item: walk(child)
        elif isinstance(item, str):
            lowered = item.replace("\\", "/").lower()
            require(not any(s in lowered for s in ("/home/", "/users/", "bench/private", "file://"))
                    and not item.startswith(("/", "\\"))
                    and not (len(item) > 1 and item[1] == ":"), "공개 registration에 사용자/private 경로가 있음")
    walk(value)


def execution_registration(protocol: Path, plan_path: Path) -> dict:
    value, plan = ADAPTER._read(protocol), ADAPTER._read(plan_path)
    require(value["source_plan_sha256"] == ADAPTER.file_sha(plan_path)
            and value["selection_plan_sha256"] == plan["selection_plan"]["sha256"], "E2E 등록 plan 결합 불일치")
    seal = {"schema": "vision_transferred_e2e_execution_registration_v2",
        "development_scope": DEVELOPMENT, "execution_protocol_sha256": ADAPTER.file_sha(protocol),
        "source_plan_sha256": ADAPTER.file_sha(plan_path), "selection_plan_sha256": plan["selection_plan"]["sha256"],
        "method": {"path": METHOD.relative_to(ROOT).as_posix(), "sha256": ADAPTER.file_sha(METHOD)},
        "source_commit": value["source_anchor"], "target_proof_sha256": plan["target_proof_sha256"]}
    public_registration(seal)
    return seal


def native_registration(protocol: Path, template: Path) -> dict:
    """private bound 원문은 유지하고 공개 가능한 hash seal만 계산한다."""
    runner, _ = native_components()
    design = runner.validate_template(template)
    bound = ADAPTER._read(protocol)
    require(bound.get("schema") == runner.PROTOCOL_SCHEMA
            and bound["record_profile"] == design["record_profile"] and bound["limits"] == design["limits"]
            and bound["fixed_flags"] == design["fixed_flags"]
            and [r["logical_id"] for r in bound["checkpoints"]] == list(runner.CHECKPOINT_IDS),
            "native bound protocol과 학습 전 template/catalog 불일치")
    rows = []
    for row in bound["checkpoints"]:
        logical = row["logical_id"]
        verified = runner._verified_binding(logical, Path(row["correction_result"]["path"]),
                                             Path(row["approval_receipt"]["path"]))
        require(row == verified, "native bound checkpoint/result/approval 결합 변경")
        rows.append({"logical_id": logical, "checkpoint_sha256": row["checkpoint"]["sha256"],
            "correction_result_sha256": row["correction_result"]["sha256"],
            "approval_receipt_sha256": row["approval_receipt"]["sha256"]})
    value = {"schema": "vision_corrected_record_registration_v1", "bound_protocol_sha256": ADAPTER.file_sha(protocol),
        "template": {"path": "bench/protocols/vision_corrected_pull_record_template_v1.json", "sha256": ADAPTER.file_sha(template)},
        "checkpoints": rows, "source_commit": bound["source_commit"], "registered_at_utc": bound["registered_at_utc"]}
    require(CALL._HEX40.fullmatch(value["source_commit"]) is not None, "native source commit 형식 오류")
    ADAPTER._time(value["registered_at_utc"], "native registration")
    public_registration(value)
    return value


def verify_public_seal(repo: Path, reference: dict, anchor: str) -> dict:
    relative = reference["path"]
    require(relative.startswith("bench/protocols/") and relative.endswith(".registration.json")
            and "private" not in relative.lower(), "공개 registration seal 상대경로가 필요함")
    path, value = ADAPTER._ref(repo, reference, "public registration seal")
    public_registration(value)
    _, committed = ObjectEvidence(repo).file(anchor, relative)
    require(committed == path.read_bytes(), "공개 registration seal이 지정 commit blob과 다름")
    return value


def verify_native_registration(repo: Path, reference: dict, anchor: str, protocol: Path, template: Path) -> dict:
    sealed = verify_public_seal(repo, reference, anchor)
    require(sealed == native_registration(protocol, template), "공개 seal과 private bound protocol/template/두 catalog 불일치")
    return sealed


def record_approval(ledger: CorrectionLedger, target: dict, correction_path: Path, output: Path) -> dict:
    correction = ADAPTER._read(correction_path)
    requests = [r for r in ledger.requests().values()
                if r.get("evidence", {}).get("source_job_id") == target["source_job_id"]]
    require(len(requests) == 1, "교정 완료 요청이 유일하지 않음")
    request = requests[0]
    require(request["state"] == "done" and request["receipt"]["verdict"] == "recovered"
            and request["receipt"]["result_sha"] == ADAPTER.digest(correction)
            and request["evidence"]["source_checkpoint_sha256"] == target["checkpoint_sha256"]
            and correction["job_id"] == target["corrected_job_id"]
            and correction["fixed_eval"] == {"none": "healthy", "corrected": "healthy"},
            "native 녹화 승인에 유효한 교정 완료 receipt가 필요함")
    value = {"schema": "vision_corrected_record_approval_v1", "approved": True,
        "request_id": request["request_id"], "corrected_job_id": target["corrected_job_id"],
        "seed": target["source_seed"], "correction_id": request["correction_id"],
        "recovery_mode": request["recovery_mode"], "protocol_case_arm": request["protocol_case_arm"],
        "correction_result_sha256": ADAPTER.file_sha(correction_path), "receipt_verdict": "recovered"}
    if output.exists():
        require(ADAPTER._read(output) == value, "기존 녹화 승인 export 변경")
    else:
        ADAPTER._write_new(output, value)
    return value


def record_kwargs(payload: dict, logical_id: str, correction: Path, approval: Path) -> dict:
    """NativeCorrectedRecordService.record의 keyword-only 계약을 사용한다."""
    return {"resource_proof": Path(payload["resource_proof"]), "resource_sha256": payload["resource_sha256"],
        "isaaclab": Path(payload["isaaclab"]), "python": Path(payload["python"]),
        "stage": Path(payload["stage"]), "logical_id": logical_id,
        "correction_result": correction, "approval_receipt": approval,
        "receipt_output": Path(payload["receipt_output"]), "normal_reference": payload.get("normal_reference")}


def native_ref(value: dict) -> Path:
    require(set(value) == {"path", "sha256", "bytes"}, "native receipt 참조 형식 오류")
    path = Path(value["path"]).resolve(strict=True)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size == value["bytes"]
            and ADAPTER.file_sha(path) == value["sha256"], "native receipt 원문/SHA/크기 불일치")
    return path


def compute_native_contracts(artifacts: dict, output: Path, checkpoint_sha: str) -> list[dict]:
    """recorder가 쓰는 실제 V01 compute 함수에 연결한다. 녹화를 실행하지 않는다."""
    _, service = native_components()
    return service._v01_contracts(artifacts, output, checkpoint_sha)


def native_contract(receipt: dict, checkpoint_sha: str) -> Path:
    require(receipt.get("schema") == "vision_corrected_native_record_receipt_v1"
            and receipt.get("checkpoint_sha256") == checkpoint_sha
            and all(receipt.get(k) is False for k in ("capture_approved", "model_ready", "used_for_decision")),
            "native 관찰 receipt identity/사용 경계 오류")
    require([r["window"] for r in receipt["v01_contracts"]] == [1, 2], "native 두 V01 window 필요")
    paths = []
    for row in receipt["v01_contracts"]:
        path = native_ref(row["contract"])
        checked = INPUTS.V01.verify(path)
        require(checked["provenance"]["checkpoint"]["sha256"] == checkpoint_sha
                and checked["behavior_ground_truth"] is None
                and checked["fixed_evaluation_used_as_behavior_ground_truth"] is False,
                "native V01 checkpoint/ground truth 경계 오류")
        paths.append(path)
    return paths[0]


def recheck_native_result(value: dict, protocol: Path, logical: str, correction: Path, approval: Path) -> dict:
    runner, service = native_components()
    require(value.get("status") == "complete" and set(value) == {"status", "job", "native_receipt"},
            "NativeCorrectedRecordService 반환 형식 오류")
    receipt_path = native_ref(value["native_receipt"])
    receipt = ADAPTER._read(receipt_path)
    binding = runner._verified_binding(logical, correction, approval)
    require(receipt.get("logical_id") == logical and receipt.get("correction_job_id") == binding["correction_job_id"]
            and native_ref(receipt["correction_result"]) == correction.resolve()
            and native_ref(receipt["approval_receipt"]) == approval.resolve(), "native 결과의 교정 승인 결합 불일치")
    job = Path(value["job"])
    validated = runner.validate_job(job, require_fresh=False)
    require(validated["protocol_path"] == protocol.resolve(), "native job의 bound protocol 불일치")
    stage = job.resolve().parent
    result_path = stage / "output/result.json"
    execution_path, execution = service._verify_execution(stage, validated, result_path)
    aaba_path, aaba_raw = service._verify_aaba(stage, validated, execution)
    artifacts, manifest = service._verify_worker(stage, validated, ADAPTER._read(result_path))
    require(native_ref(receipt["technical_execution"]) == execution_path.resolve()
            and native_ref(receipt["native_capture_manifest"]) == manifest.resolve()
            and native_ref(receipt["native_result"]) == result_path.resolve()
            and native_ref(receipt["aaba_raw_source"]["report"]) == aaba_path.resolve()
            and receipt["aaba_raw_source"]["artifacts"] == aaba_raw
            and all(native_ref(receipt["artifacts"][key]) == path.resolve() for key, path in artifacts.items()),
            "native receipt가 실제 AABA/625프레임/8 artifact와 다름")
    native_contract(receipt, binding["checkpoint"]["sha256"])
    return receipt


def audit_source_cache(archive: Path, output: Path, *, summary_path: Path = SUMMARY,
                       source_path: Path = SOURCE) -> dict:
    """보관된 네 source를 봉인한다. 완료 학습 원장이나 manifest를 새로 만들지 않는다."""
    archive = archive.resolve(strict=True)
    summary = ADAPTER._read(summary_path)
    proof = INPUTS.verify_archive(archive, summary)
    inventory = ADAPTER._read(archive / "inventory.json")
    cache = {"schema": SOURCE_SCHEMA, "archive": str(archive), "summary": ref(summary_path),
             "source_protocol": ref(source_path), "archive_verification": proof, "jobs": {}}
    source = ADAPTER._read(source_path)
    source_root = archive / "tree/rl-triage-brev-e2e-20261024/repo/.triage-runtime/vision-brev/vision-source-20261026x"
    for public in summary["training"]:
        seed, arm, job = public["seed"], public["arm"], public["job_id"]
        require(seed in SEEDS.values() and arm in {"NONE", "F1"}, "source 네 학습 밖의 행")
        matches = [row for row in inventory["files"] if row["sha256"] == public["checkpoint_sha256"]]
        require(len(matches) == 1, "보관 checkpoint가 유일하지 않음")
        checkpoint = archive / "tree" / matches[0]["path"]
        actions = list((source_root / "actions").glob("*/raw.json"))
        raw = [p for p in actions if ADAPTER.file_sha(p) == public["raw_sha256"]]
        require(len(raw) == 1, "공개 학습 raw가 보관 자료와 다름")
        stage = source_root / "training" / job
        material = INPUTS.verify_training(stage, raw[0], checkpoint, public, summary)
        require(any(row["job_id"] == job and row["arm"] == arm and row["seed"] == seed
                    for row in source["training_plan"]), "source protocol 학습 결합 불일치")
        cache["jobs"][job] = {"stage": str(stage), "raw_action": ref(raw[0]),
                              "checkpoint": ref(checkpoint), "public": public,
                              "files": material["files"]}
    require(len(cache["jobs"]) == 4, "네 source 학습이 모두 필요함")
    ADAPTER._write_new(output, cache)
    return cache


class SourceCache:
    """원문을 매번 검증하며 경로 두 개만 호출용 사본에 바꾼다."""

    def __init__(self, reference: Mapping[str, Any]):
        self.reference = dict(reference)
        _, self.index = read_ref(reference)
        require(self.index.get("schema") == SOURCE_SCHEMA and len(self.index.get("jobs", {})) == 4,
                "네 source cache index가 아님")
        require(self.index["summary"] == ref(SUMMARY) and self.index["source_protocol"] == ref(SOURCE),
                "source cache가 동결 공개 summary/protocol과 다름")
        summary = ADAPTER._read(SUMMARY)
        require(INPUTS.verify_archive(Path(self.index["archive"]), summary) == self.index["archive_verification"],
                "source 보관 inventory 재검증 불일치")
        require(set(self.index["jobs"]) == {r["job_id"] for r in summary["training"]}
                and all(self.index["jobs"][r["job_id"]]["public"] == r for r in summary["training"]),
                "source cache가 기존 네 학습과 다름")

    def _completed(self, job_id: str) -> dict | None:
        _, current = read_ref(self.reference)
        require(current == self.index, "source cache index 변경")
        item = current["jobs"].get(job_id)
        if item is None:
            return None
        _, summary = read_ref(current["summary"])
        read_ref(current["source_protocol"])
        raw, _ = read_ref(item["raw_action"])
        checkpoint = Path(item["checkpoint"]["path"])
        require(ADAPTER.file_sha(checkpoint) == item["checkpoint"]["sha256"], "source checkpoint 변경")
        material = INPUTS.verify_training(Path(item["stage"]), raw, checkpoint, item["public"], summary)
        require(material["files"] == item["files"], "source cache 파일 결합 변경")
        return material["artifact"]


class TransferredTrainingBackend(TRAINING.VisionBrevTrainingBackendV2):
    """기존 CORRECTED runner는 유지하고 source 네 건만 읽기 전용 cache로 공급한다."""

    @classmethod
    def from_runtime(cls, *args: Any, source_cache_reference: Mapping[str, Any] | None = None,
                     **kwargs: Any) -> Any:
        require(source_cache_reference is not None, "실제 transferred source cache 참조가 필요함")
        cache = SourceCache(source_cache_reference)
        backend = super().from_runtime(*args, **kwargs)
        backend.transferred_source_cache = cache
        return backend

    def _completed(self, job_id: str) -> dict | None:
        row = self.plan_by_id.get(job_id)
        require(row is not None, "등록되지 않은 학습 job")
        if row["arm"] in {"NONE", "F1"}:
            return self.transferred_source_cache._completed(job_id)
        return super()._completed(job_id)

    def train(self, job_id: str, *, approval_reference: Mapping[str, Any] | None = None) -> dict:
        require(job_id in {f"vision-s{seed}-corrected" for seed in SEEDS.values()},
                "새 source 학습 금지; 기존 CORRECTED 두 건만 허용")
        return super().train(job_id, approval_reference=approval_reference)


def target_jobs(source: dict, row: dict) -> dict:
    group, seed = row["seed_group"], row["source_seed"]
    require(SEEDS.get(group) == seed, "고정 source seed 불일치")
    jobs = [p for p in source["training_plan"] if p["seed"] == seed]
    by_arm = {p["arm"]: p for p in jobs}
    require(set(by_arm) == {"NONE", "F1", "CORRECTED"} and len(jobs) == 3, "source 학습 계획 불일치")
    require(by_arm["CORRECTED"]["source_job_id"] == by_arm["F1"]["job_id"]
            and all(p["iterations"] == 300 and p["num_envs"] == 4096 for p in jobs),
            "restart parent 또는 학습 규모 불일치")
    return {arm: by_arm[arm]["job_id"] for arm in by_arm}


def prepare_blueprint(repo: Path, protocol_path: Path, locations_path: Path, source_cache: Path,
                      output: Path | None, *, execution_id: str) -> dict:
    """Git/API 없이 실제 24개 capsule을 검증해 고정 두 대상을 준비한다."""
    repo = repo.resolve(strict=True)
    validate_method()
    proof = target_proof(repo, protocol_path, locations_path)
    protocol, locations = ADAPTER._read(protocol_path), ADAPTER._read(locations_path)
    verified = SELECTION.verify_input_verifier(repo, protocol, locations_path)
    targets = CONTRACT._target_rows(protocol)
    cache = SourceCache(ref(source_cache))
    _, source = read_ref(cache.index["source_protocol"])
    acceptance_path = SELECTION._relative(repo, locations["acceptance"]["path"])
    acceptance = SELECTION._read(acceptance_path)
    policies = SELECTION._policy_catalog(acceptance)
    rows = []
    by_call = {r["call_id"]: r for r in verified["rows"]}
    for target, row in zip(CONTRACT.TARGETS, targets, strict=True):
        jobs = target_jobs(source, row)
        current = cache._completed(jobs["F1"])
        normal = cache._completed(jobs["NONE"])
        policy = policies[(target["seed_group"], "registered-f1")]
        require(current is not None and normal is not None and
                current["checkpoint"]["sha256"] == row["checkpoint_sha256"] == policy["checkpoint_sha256"],
                "선택 capsule과 실제 학습 checkpoint 불일치")
        capsule = by_call[row["call_id"]]["capsule"]
        frozen = CALL.prepared(capsule)
        require(frozen["view_kind"] == "text_and_video" and
                frozen["bundle"]["operational_base"]["case_checkpoint_sha256"] == row["checkpoint_sha256"],
                "고정 capsule identity 불일치")
        trace = SELECTION._read(policy["trace"])
        training = next(p for pair in acceptance["pairs"] for p in
                        SELECTION._read(SELECTION._absolute_ref(pair["training_receipt"], "training"))["policies"]
                        if p["logical_id"] == policy["logical_id"])
        window = next(w for w in training["windows"] if w["window"] == 1)
        contract_path = SELECTION._absolute_ref(window["contract"], "window01 contract")
        require(ADAPTER.file_sha(contract_path) == row["contract_sha256"], "window01 contract 결합 불일치")
        snapshot_path = SELECTION._absolute_ref(policy["snapshot"], "snapshot")
        snapshot = ADAPTER._read(snapshot_path)
        binding = {"contract_sha256": ADAPTER.file_sha(contract_path), "source_lineage_id": row["source_lineage_id"]}
        CALL.CONTRACT.verify(frozen["bundle"], snapshot, contract_path, binding)
        source_binding = snapshot["inputs"]["packet"]["input_bindings"]
        require(source_binding["case"]["telemetry_sha256"] == current["telemetry"]["sha256"]
                and source_binding["reference"]["telemetry_sha256"] == normal["telemetry"]["sha256"],
                "선택 snapshot의 실제 학습 telemetry 불일치")
        require(trace["job_id"] + "-same-play" == row["source_lineage_id"], "record job/계보 불일치")
        rows.append({"target": dict(target), "call_id": f"e2e-{target['seed_group']}",
                     "source_call_id": row["call_id"], "source_seed": row["source_seed"],
                     "source_job_id": jobs["F1"], "record_job_id": trace["job_id"],
                     "source_lineage_id": row["source_lineage_id"], "corrected_job_id": jobs["CORRECTED"],
                     "checkpoint_sha256": row["checkpoint_sha256"],
                     "reference_checkpoint_sha256": normal["checkpoint"]["sha256"],
                     "capsule_sha256": capsule["sha256"], "capsule": capsule,
                     "eligible_probes": frozen["bundle"]["selection_view"]["eligible_probes"],
                     "source_protocol_sha256_lf": protocol["source_protocol"]["sha256_lf"],
                     "live_snapshot": ref(snapshot_path), "live_contract": ref(contract_path), "source_binding": binding})
    value = {"schema": SCHEMA, "status": "development_prepared_requires_execution_registration", "execution_id": execution_id,
             "development_scope": DEVELOPMENT, "target_proof": proof,
             "original_before_comparison_policy": "v1_preserved_not_satisfied_not_waived",
             "api_cap": 32, "comparison_api_cap": 72, "comparison_protocol": ref(protocol_path),
             "input_locations": ref(locations_path), "acceptance": ref(acceptance_path),
             "source_cache": ref(source_cache), "method": ref(METHOD), "rows": rows,
             "initial_reuse": "same_e2e_terminal_only_no_comparison_terminal_copy",
             "correction": {"id": "restore_body_frame_reward", "recovery_mode": "restart"},
             "post_observation_used_for_decision": False, "new_source_trainings": 0}
    if output is not None:
        ADAPTER._write_new(output, value)
    return value


def verify_blueprint(path: Path, repo: Path) -> dict:
    value = ADAPTER._read(path)
    require(value.get("schema") == SCHEMA, "E2E blueprint schema 불일치")
    for name in ("comparison_protocol", "input_locations", "source_cache", "acceptance", "method"):
        read_ref(value[name])
    require(Path(value["method"]["path"]).resolve() == METHOD.resolve(), "등록 method 경로 불일치")
    fresh = prepare_blueprint(repo, Path(value["comparison_protocol"]["path"]),
        Path(value["input_locations"]["path"]), Path(value["source_cache"]["path"]), None,
        execution_id=value["execution_id"])
    require(fresh == value, "E2E blueprint의 capsule·실제 source 결합 변경")
    return value


def seal_plan(repo: Path, blueprint_path: Path, output: Path, *, source_anchor: str) -> dict:
    """실제 capsule의 CALL manifest를 소유 출력에 봉인한다. canonical 배치는 메인 단계다."""
    blueprint = verify_blueprint(blueprint_path, repo)
    calls = [{"call_id": row["call_id"], "run_id": f"vision-selection-{blueprint['execution_id']}-{row['call_id']}",
              "capsule": row["capsule"]} for row in blueprint["rows"]]
    require(CALL._HEX40.fullmatch(source_anchor) is not None, "source anchor 형식 오류")
    manifest = planned_manifest(repo, blueprint["execution_id"], source_anchor, calls)
    manifest_path = output.with_name(output.stem + ".selection-manifest.json")
    CALL._write_new(manifest_path, manifest)
    reference = {"path": str((repo / ".triage-runtime/vision-selection" / blueprint["execution_id"] / "manifest.json").resolve()),
                 "sha256": ADAPTER.file_sha(manifest_path), "status": "planned"}
    plan = {"schema": CONTRACT.SCHEMA, "api_cap": 32, "execution_id": blueprint["execution_id"],
            "source_anchor": source_anchor, "protocol_sha256": ADAPTER.file_sha(METHOD),
            "selection_plan": reference, "rows": blueprint["rows"], "transferred_blueprint": ref(blueprint_path),
            "planned_selection_manifest": ref(manifest_path),
            "selection_activation": "main_activate_selection_before_first_API",
            "development_scope": DEVELOPMENT, "target_proof_sha256": ADAPTER.digest(blueprint["target_proof"]),
            "correction": blueprint["correction"], "fixed_eval_decides_recovery": True,
            "post_observation_used_for_decision": False}
    ADAPTER._write_new(output, plan)
    return plan


def planned_manifest(repo: Path, execution_id: str, anchor: str, calls: list[dict]) -> dict:
    require(CALL._IDENTIFIER.fullmatch(execution_id) is not None, "E2E execution ID 형식 오류")
    rows = []
    for index, row in enumerate(calls, 1):
        frozen = CALL.prepared(row["capsule"])
        rows.append({"ordinal": index, "call_id": row["call_id"], "run_id": row["run_id"],
            "capsule_path": Path(row["capsule"]["path"]).resolve().relative_to(repo.resolve()).as_posix(),
            "capsule_sha256": row["capsule"]["sha256"], "view_kind": frozen["view_kind"],
            "checkpoint_sha256": frozen["checkpoint_sha256"]})
    return {"schema": CALL.PLAN_SCHEMA, "execution_id": execution_id, "phase": "e2e", "api_cap": 32,
            "source_anchor": anchor, "protocol_sha256": ADAPTER.file_sha(METHOD),
            "request_params": deepcopy(OBSERVER.REQUEST_PARAMS), "rows": rows}


def activate_selection(repo: Path, plan_path: Path) -> dict:
    plan = ADAPTER._read(plan_path)
    blueprint_path, _ = read_ref(plan["transferred_blueprint"])
    verify_blueprint(blueprint_path, repo)
    _, manifest = read_ref(plan["planned_selection_manifest"])
    calls = [{"call_id": r["call_id"], "run_id": f"vision-selection-{plan['execution_id']}-{r['call_id']}",
              "capsule": r["capsule"]} for r in plan["rows"]]
    require(planned_manifest(repo, plan["execution_id"], plan["source_anchor"], calls) == manifest,
            "사전 봉인 CALL manifest와 현재 입력 불일치")
    reference = CALL.initialize(repo, plan["execution_id"], "e2e", plan["source_anchor"], ADAPTER.file_sha(METHOD), calls)
    require(reference == plan["selection_plan"], "실제 CALL consumer manifest가 CPU plan과 다름")
    return {"status": "selection_manifest_activated", "selection_plan": reference, "model_api_requests": 0}


def verify_scope(scope: dict, plan: dict, context: dict) -> None:
    require(scope.get("schema") == SCOPE_SCHEMA and scope.get("scope") == SCOPE
            and scope.get("model_use_authorized") is True
            and scope.get("capture_approved") is False and scope.get("used_for_decision") is False
            and CALL._HEX40.fullmatch(str(scope.get("execution_registration_anchor", ""))) is not None
            and isinstance(scope.get("approval_id"), str) and bool(scope["approval_id"].strip()),
            "명시된 transferred E2E host-use 범위 승인이 필요함")
    require(scope.get("execution_protocol_sha256") == context["protocol_sha256"]
            and scope.get("plan_sha256") == context["protocol"]["source_plan_sha256"]
            and scope.get("blueprint_sha256") == plan["transferred_blueprint"]["sha256"]
            and scope.get("resource_sha256") == context["lease"]["resource"]["sha256"]
            and scope.get("capsules") == {r["target"]["seed_group"]: r["capsule_sha256"] for r in plan["rows"]}
            and scope.get("lineages") == {r["target"]["seed_group"]: r["source_lineage_id"] for r in plan["rows"]},
            "host-use 승인이 현재 자원·선택 capsule·계보와 다름")
    now = datetime.now(timezone.utc)
    issued = ADAPTER._time(scope.get("issued_at_utc"), "host-use issued")
    expires = ADAPTER._time(scope.get("expires_at_utc"), "host-use expires")
    require(issued <= now < expires, "host-use 승인 유효 기간 밖")


def validate_method(path: Path = METHOD, *, require_bound: bool = False) -> dict:
    value = ADAPTER._read(path)
    require(value.get("schema") == "vision_transferred_e2e_method_v2"
            and value.get("development_scope") == DEVELOPMENT
            and value.get("execution_registration") == "external_committed_bundle_proof_required"
            and value.get("targets") == list(CONTRACT.TARGETS)
            and value.get("limits") == {"comparison_api_requests": 72, "e2e_api_requests": 32,
                "probe_pairs": 2, "gpu_probe_runs": 4, "corrected_runs": 2, "new_source_trainings": 0}
            and value.get("training") == {"num_envs": 4096, "iterations": 300,
                "seeds": [2220, 2221], "correction_id": "restore_body_frame_reward", "mode": "restart"}
            and value.get("request_params") == OBSERVER.REQUEST_PARAMS
            and value.get("source_flags_preserved") == {"capture_approved": False,
                "model_ready": False, "used_for_decision": False}
            and value.get("post_observation_used_for_decision") is False,
            "transferred E2E method 고정값 불일치")
    for relative, value_ref in value.get("immutable_baseline", {}).items():
        require(ADAPTER.file_sha(ROOT / relative, lf=True) == value_ref["sha256_lf"], "기존 protocol/consumer 변경")
    return value


def probe_stage(root: Path, protocol: dict, group: str) -> Path:
    relative = Path(protocol.get("execution_root", ""))
    require(group in SEEDS and not relative.is_absolute() and ".." not in relative.parts
            and relative.as_posix().startswith(".triage-runtime/vision-transferred-e2e-development-v2/"),
            "probe canonical execution root 또는 group 불일치")
    return (root / protocol["execution_root"] / group).resolve()


def development_probe(repo: Path, execution_id: str, *, require_history: bool = False) -> dict:
    import vision_transferred_probe_v1 as PROBE
    method = validate_method()
    base_ref = method["probe_base"]
    path = repo / base_ref["path"]
    require(ADAPTER.file_sha(path, lf=True) == base_ref["sha256_lf"], "동결 P_reward base protocol 변경")
    protocol = deepcopy(PROBE.validate_protocol(path))
    protocol["protocol_id"] = method["protocol_id"] + "-fresh-probes"
    protocol["execution_root"] = f".triage-runtime/vision-transferred-e2e-development-v2/{execution_id}/probes"
    protocol["purpose"] = "fresh_E2E_confirmation_not_comparison_validation_receipt"
    protocol["_path"] = str(METHOD)
    for group in protocol["groups"]:
        for arm in ("NONE", "F1"):
            group["execution"][arm]["run_name"] = f"vision_e2e_s{group['seed']}_{arm.lower()}"
    if require_history:
        metadata = repo / ".git"
        require(not (metadata / "shallow").exists(), "E2E probe는 완전한 source history가 필요함")
        ObjectEvidence(repo).commit(protocol["probe_runtime"]["source_anchor"])
    return protocol


def run_development_probe(root: Path, protocol: dict, archive: Path, group_name: str,
                          lab: Path, python: Path, approver: str) -> dict:
    """별도 개발 E2E 승인과 새 stage로 기존 실제 probe consumer를 실행한다."""
    import vision_transferred_probe_v1 as PROBE
    inputs = PROBE.verify_source_inputs(root, archive, protocol)
    group = next(r for r in protocol["groups"] if r["group"] == group_name)
    stage = probe_stage(root, protocol, group_name)
    require(not stage.exists(), "새 E2E probe stage만 허용")
    stage.mkdir(parents=True)
    auth = PROBE.authorize_pair(stage, group, ADAPTER.file_sha(METHOD), approver)
    actual, descriptors = {}, {}
    try:
        runtime = inputs["runtime_protocol"]
        backend = PROBE.BACKEND.Backend(lab, python, {**runtime,
            "protocol_path": str(root / protocol["probe_runtime"]["source_protocol"]["path"]),
            "protocol_sha256_lf": protocol["probe_runtime"]["source_protocol"]["sha256_lf"]},
            protocol["probe_runtime"]["source_anchor"])
        expected = PROBE.expected_probe_source(protocol, inputs)
        for arm, side in (("NONE", "reference"), ("F1", "case")):
            path = PROBE._run_arm(backend, root, inputs["groups"][group_name]["arms"][arm], arm,
                                  group["execution"][arm], stage)
            descriptors[arm] = PROBE._ref(root, path)
            actual[arm] = PROBE.CONSUMER._brev_actual_probe(root, PROBE.read(path), side=side, probe="P_reward",
                checkpoint_sha256=group["checkpoints"][arm]["sha256"], expected_source=expected)
        wall = sum(float(a["gpu_s"]) for a in actual.values())
        require(wall <= 300, "E2E probe pair 상한 초과")
        outcomes = {arm: PROBE.CONTROLLER.classify_raw("P_reward", actual[arm]["measurement"],
                    actual["NONE"]["measurement"]) for arm in ("NONE", "F1")}
    except Exception as error:
        wall = sum(float(a["gpu_s"]) for a in actual.values())
        if isinstance(error, PROBE.BACKEND.BackendError): wall += float(error.receipt.get("wall_s", 0))
        PROBE._append(auth["ledger"], {"schema": PROBE.LEDGER_SCHEMA, "event": "receipt",
            "request_id": auth["request_id"], "request_sha256": PROBE.digest(auth["request"]),
            "at": datetime.now(timezone.utc).isoformat(), "outcomes": {"NONE": "unknown", "F1": "unknown"},
            "gpu_s": wall, "descriptors": descriptors, "status": "failed", "error_type": type(error).__name__})
        raise
    PROBE._append(auth["ledger"], {"schema": PROBE.LEDGER_SCHEMA, "event": "receipt",
        "request_id": auth["request_id"], "request_sha256": PROBE.digest(auth["request"]),
        "at": datetime.now(timezone.utc).isoformat(), "outcomes": outcomes, "gpu_s": wall,
        "descriptors": descriptors})
    result = PROBE.validate_pair(root, protocol, inputs, stage)
    ADAPTER._write_new(stage / "result.json", result)
    return result


def verify_probe_runtime_binding(context: dict, protocol: dict, group: str) -> dict:
    entry = next(e for e in context["protocol"]["groups"] if e["group"] == group)
    rows = [r for r in protocol["groups"] if r["group"] == group]
    require(len(rows) == 1 and rows[0]["seed"] == entry["seed"]
            and {arm: rows[0]["checkpoints"][arm]["sha256"] for arm in ("NONE", "F1")} == entry["checkpoints"]
            and protocol["source_training"]["source_protocol"]["sha256"] == entry["source_protocol_sha256_lf"]
            and protocol["resources"]["instance_id"] == context["resource"]["resource_id"]
            and protocol["resources"]["compute_quote_usd_per_h"] ==
                context["protocol"]["resources"]["compute_quote_usd_per_h"],
            "새 probe가 E2E source·GPU 자원·quote와 다름")
    return entry


def consumed_action(ledger: CorrectionLedger, request_id: str, group: str,
                    action: Callable[[], dict], failure_path: Path) -> dict:
    """실패한 소비를 unknown receipt로 닫고 동일 effect 재실행을 막는다."""
    request = ledger.requests().get(request_id)
    require(isinstance(request, dict) and request["state"] == "approved", "새 승인만 소비할 수 있음")
    ledger.consume(request_id, request["loop_state_digest"])
    try:
        result = action()
        require(isinstance(result, dict), "교정 실행 결과가 object가 아님")
        if result.get("status") == "complete":
            require(isinstance(result.get("fixed_eval"), dict)
                    and set(result["fixed_eval"]) == {"none", "corrected"}
                    and type(result.get("gpu_s")) in {int, float} and result["gpu_s"] >= 0
                    and result.get("exit_code") == 0, "교정 완료 receipt 필수 필드 누락")
    except Exception as error:  # noqa: BLE001 - 소비된 승인에는 실패 receipt를 남긴다.
        failure = {"status": "HOLD", "reason": "execution_failed_after_consume", "group": group,
                   "error_type": type(error).__name__, "error": str(error)}
        ledger.receipt(request_id, group, "unknown", ADAPTER.digest(failure), None, 1, reason=type(error).__name__)
        ADAPTER._write_new(failure_path, failure)
        return failure
    if result.get("status") != "complete":
        ledger.receipt(request_id, group, "unknown", ADAPTER.digest(result), result.get("gpu_s"), 1,
                       reason=str(result.get("reason", "incomplete_execution")))
        return result
    recovered = result["fixed_eval"] == {"none": "healthy", "corrected": "healthy"}
    ledger.receipt(request_id, group, "recovered" if recovered else "not_recovered",
                   ADAPTER.digest(result), result["gpu_s"], result["exit_code"])
    return result


def project_probe(root: Path, protocol: dict, source_inputs: dict, stage: Path,
                  selection: dict, target: dict, intent: dict) -> dict:
    import vision_transferred_probe_v1 as PROBE
    require(intent["selection_terminal_sha256"] == selection["terminal_sha256"]
            and intent["activation_bridge_sha256"] == selection["activation_bridge_sha256"]
            and intent["checkpoint_sha256"] == target["checkpoint_sha256"]
            and intent["protocol_sha256"] == ADAPTER.file_sha(Path(protocol["_path"])),
            "새 검사 intent와 현재 activation 결합 불일치")
    verified = PROBE.validate_pair(root, protocol, source_inputs, stage)
    require(verified["group"] == target["target"]["seed_group"] and verified["seed"] == target["source_seed"],
            "probe group/seed 불일치")
    result = {"probe": "P_reward", "gpu_s": verified["gpu_s"], "source_receipt": verified}
    for arm, key in (("NONE", "none"), ("F1", "f1")):
        descriptor_path = PROBE._reference(root, verified["descriptors"][arm], "probe descriptor")
        descriptor = ADAPTER._read(descriptor_path)
        execution = PROBE._reference(root, descriptor["sources"]["execution"], "probe execution")
        body = ADAPTER._read(execution)
        auth_path = PROBE._reference(root, verified["authorization"], "probe authorization")
        auth = ADAPTER._read(auth_path)
        events = PROBE._ledger(stage / "approval.ledger.jsonl")
        require(ADAPTER.digest(events[:3]) == auth["ledger_prefix_sha256"], "새 probe 승인 prefix 불일치")
        at = datetime.fromisoformat(events[0]["at"].replace("Z", "+00:00"))
        require(at >= datetime.fromisoformat(intent["created_at_utc"]), "activation 전 consumed probe 재사용 금지")
        started = ADAPTER._time(body.get("started_at"), "probe started")
        finished = ADAPTER._time(body.get("finished_at"), "probe finished")
        consumed = ADAPTER._time(events[2].get("at"), "probe consumed")
        receipted = ADAPTER._time(events[3].get("at"), "probe receipt")
        require(at <= consumed <= started <= finished <= receipted <= datetime.now(timezone.utc),
                "새 activation·승인·실제 실행·receipt 시각 순서 불일치")
        result[key] = {"status": "complete", "outcome": verified["outcomes"][arm], "seed": verified["seed"],
                       "arm": arm, "checkpoint_sha256": descriptor["checkpoint_sha256"],
                       **{name + "_sha256": descriptor["sources"][name]["sha256"]
                          for name in ("job", "worker", "execution", "report_jobs", "report")},
                       "gpu_before_sha256": ADAPTER.digest(body["gpu_before"]),
                       "gpu_after_sha256": ADAPTER.digest(body["gpu_after"]),
                       "authorization_sha256": ADAPTER.file_sha(auth_path),
                       "seal_sha256": ADAPTER.file_sha(descriptor_path)}
    return result


class Runtime:
    """실제 실행은 기존 source/lease/SDK consumer 검증을 통과한 뒤만 허용한다."""

    def __init__(self, repo: Path, plan_path: Path, context: dict, scope_path: Path, *, require_active: bool = True):
        self.root, self.plan_path, self.context = repo.resolve(strict=True), plan_path, context
        self.require_active = require_active
        self.plan, self.scope = ADAPTER._read(plan_path), ref(scope_path)
        self.refresh()
        self.path = self.root / ".triage-runtime/vision-selection-e2e" / self.plan["execution_id"]
        self.corrections = CorrectionLedger(self.path / "corrections.ledger.jsonl")

    def refresh(self) -> None:
        validate_method()
        ADAPTER.refresh_context(self.context, require_active=self.require_active)
        _, scope = read_ref(self.scope)
        require(ADAPTER.file_sha(self.plan_path) == self.context["protocol"]["source_plan_sha256"], "E2E plan 변경")
        blueprint_path, _ = read_ref(self.plan["transferred_blueprint"])
        blueprint = verify_blueprint(blueprint_path, self.root)
        require(self.plan["rows"] == blueprint["rows"] and self.plan["api_cap"] == 32
                and self.plan["schema"] == CONTRACT.SCHEMA, "sealed plan과 blueprint 불일치")
        _, _, manifest = CALL._plan(self.plan["selection_plan"])
        require(manifest["phase"] == "e2e" and manifest["api_cap"] == 32
                and manifest["protocol_sha256"] == ADAPTER.file_sha(METHOD)
                and manifest["source_anchor"] == self.plan["source_anchor"]
                and [(r["call_id"], r["capsule_sha256"]) for r in manifest["rows"]] ==
                    [(r["call_id"], r["capsule_sha256"]) for r in self.plan["rows"]], "E2E API32 원장 불일치")
        for row in self.plan["rows"]:
            group = row["target"]["seed_group"]
            entry = next(e for e in self.context["protocol"]["groups"] if e["group"] == group)
            require(entry["seed"] == row["source_seed"] and entry["checkpoints"] == {
                "NONE": row["reference_checkpoint_sha256"], "F1": row["checkpoint_sha256"]}
                and entry["corrected_job_id"] == row["corrected_job_id"]
                and entry["source_protocol_sha256_lf"] == row["source_protocol_sha256_lf"],
                "execution protocol group이 고정 학습 계보와 다름")
        verify_scope(scope, self.plan, self.context)
        verify_registration(self.root, self.context, scope)

    def row(self, group: str) -> dict:
        return CONTRACT._row(self.plan, group)

    def selection(self, group: str) -> dict:
        row = self.row(group)
        path = self.path / "selections" / f"{group}.json"
        value = ADAPTER._read(path)
        _, stage, manifest = CALL._plan(self.plan["selection_plan"])
        _, call, frozen = CALL._row(self.root, manifest, row["call_id"])
        terminal = CALL._replay(stage, manifest, call, frozen)
        bridge_path, _ = read_ref(value["bridge_reference"])
        snapshot_path, snapshot = read_ref(row["live_snapshot"])
        del snapshot_path
        contract_path, _ = read_ref(row["live_contract"])
        checked = CALL.LOOP.verify_bridge(value["bridge_reference"], snapshot, contract_path, row["source_binding"])
        engine = terminal["engine_result"]
        require(value["terminal_sha256"] == ADAPTER.file_sha(stage / "decisions" / row["call_id"] / "terminal.json")
                and value["activation_bridge_sha256"] == ADAPTER.file_sha(bridge_path)
                and value["model_status"] == engine["model_status"]
                and all(value.get(k) == v for k, v in engine["selection"].items())
                and checked["model_content"] == engine["final_content"], "선택 원문 변경")
        CALL.LOOP.host_request_spec(value["bridge_reference"], snapshot, contract_path, row["source_binding"])
        return value

    def initial(self, group: str, complete: Callable) -> dict:
        self.refresh()
        row = self.row(group)
        output = self.path / "selections" / f"{group}.json"
        if output.exists():
            return self.selection(group)
        journal = CALL.run(self.plan["selection_plan"], row["call_id"], complete)["journal"]
        if journal.get("status") not in {"choice_ready", "terminal_hold"}:
            return {"status": "HOLD", "reason": "selection_pending", "journal": journal}
        _, snapshot = read_ref(row["live_snapshot"])
        contract, _ = read_ref(row["live_contract"])
        active = CALL.activate(self.plan["selection_plan"], row["call_id"], snapshot, contract,
                               row["source_binding"], self.path / "bridges" / f"{group}.json")
        engine = journal["engine_result"]
        value = {"group": group, "model_status": engine["model_status"], **engine["selection"],
                 "terminal_sha256": active["terminal_sha256"], "bridge_reference": active["bridge_reference"],
                 "activation_bridge_sha256": ADAPTER.file_sha(Path(active["bridge_reference"]["path"]))}
        ADAPTER._write_new(output, value)
        return value

    def probe(self, group: str, protocol_path: Path, archive: Path, lab: Path, python: Path, approver: str) -> dict:
        import vision_transferred_probe_v1 as PROBE
        self.refresh()
        selected, row = self.selection(group), self.row(group)
        if selected.get("model_status") != "valid" or selected.get("action") != "continue" or selected.get("probe") != "P_reward":
            return {"status": "HOLD", "reason": "no_valid_P_reward_selection"}
        require(protocol_path.resolve(strict=True) == METHOD.resolve(), "E2E 전용 method probe contract만 허용")
        protocol = development_probe(self.root, self.plan["execution_id"], require_history=True)
        inputs = PROBE.verify_source_inputs(self.root, archive, protocol)
        require(ADAPTER.file_sha(inputs["groups"][group]["arms"]["F1"]["checkpoint"]) == row["checkpoint_sha256"]
                and ADAPTER.file_sha(inputs["groups"][group]["arms"]["NONE"]["checkpoint"]) == row["reference_checkpoint_sha256"],
                "probe source checkpoint 불일치")
        entry = verify_probe_runtime_binding(self.context, protocol, group)
        ADAPTER.require_action_window(self.context, entry, "probe_pair")
        intent_path, stage = self.path / "intents" / f"probe-{group}.json", probe_stage(self.root, protocol, group)
        if intent_path.exists() or stage.exists():
            return {"status": "HOLD", "reason": "probe_already_reserved_no_auto_rerun"}
        intent = {"selection_terminal_sha256": selected["terminal_sha256"],
                  "activation_bridge_sha256": selected["activation_bridge_sha256"],
                  "checkpoint_sha256": row["checkpoint_sha256"], "protocol_sha256": ADAPTER.file_sha(protocol_path),
                  "protocol": ref(protocol_path), "archive": str(archive.resolve(strict=True)),
                  "stage": str(stage),
                  "created_at_utc": datetime.now(timezone.utc).isoformat()}
        ADAPTER._write_new(intent_path, intent)
        result = run_development_probe(self.root, protocol, archive, group, lab, python, approver)
        if result.get("status") != "complete":
            return result
        projected = project_probe(self.root, protocol, inputs, stage, selected, row, intent)
        ADAPTER._action_cost(self.context, entry, "probe_pair", projected["gpu_s"])
        ADAPTER._write_new(self.path / "pairs" / f"{group}.json", projected)
        return projected

    def pair(self, group: str) -> dict:
        import vision_transferred_probe_v1 as PROBE
        selected, row = self.selection(group), self.row(group)
        intent = ADAPTER._read(self.path / "intents" / f"probe-{group}.json")
        protocol_path, _ = read_ref(intent["protocol"])
        require(protocol_path == METHOD.resolve(), "cached E2E method probe protocol 변경")
        protocol = development_probe(self.root, self.plan["execution_id"])
        entry = verify_probe_runtime_binding(self.context, protocol, group)
        inputs = PROBE.verify_source_inputs(self.root, Path(intent["archive"]), protocol)
        stage = probe_stage(self.root, protocol, group)
        require(str(stage) == intent["stage"], "probe canonical stage 변경")
        projected = project_probe(self.root, protocol, inputs, stage, selected, row, intent)
        ADAPTER._action_cost(self.context, entry, "probe_pair", projected["gpu_s"])
        require(projected == ADAPTER._read(self.path / "pairs" / f"{group}.json"), "probe 원문 재검증 결과 변경")
        return projected

    def candidate(self, group: str) -> tuple[dict | None, dict, dict]:
        selected, pair = self.selection(group), self.pair(group)
        candidate = CONTRACT.approval_candidate(self.plan, group, selected,
                                                {k: pair[k] for k in ("probe", "none", "f1")})
        return candidate, selected, pair

    def propose(self, group: str, complete: Callable | None = None) -> dict:
        # 기존 선택·검사 원문만 소비한다. callback은 relay driver의 공통 주입 계약이며 호출하지 않는다.
        del complete
        self.refresh()
        candidate, selected, pair = self.candidate(group)
        if candidate is None:
            return {"status": "HOLD", "reason": "registered_confirmation_not_met"}
        requests = self.corrections.requests().values()
        require(not any(r.get("evidence", {}).get("source_job_id") == candidate["source_job_id"]
                        for r in requests), "같은 source 교정 요청 재사용 금지")
        require(sum(e["event"] == "proposed" for e in self.corrections.events()) < 2, "교정 두 건 상한")
        return self.corrections.propose(ADAPTER.digest({"selection": selected, "pair": pair}), "reward",
            ADAPTER.digest(pair), "restore_body_frame_reward", "restart", {"case": "case_01", "arm": "corrected"},
            3600, evidence={**candidate, "pair_result_sha256": ADAPTER.digest(pair)})

    def approve(self, group: str, request_id: str, approver: str) -> dict:
        self.refresh()
        candidate, selected, pair = self.candidate(group)
        request = self.corrections.requests().get(request_id)
        require(candidate is not None and request is not None and request["state"] == "pending"
                and request["evidence"] == {**candidate, "pair_result_sha256": ADAPTER.digest(pair)}
                and request["loop_state_digest"] == ADAPTER.digest({"selection": selected, "pair": pair})
                and bool(approver.strip()), "교정 승인 대상과 현재 증거 불일치")
        return self.corrections.approve(request_id, approver)

    def correct(self, group: str, request_id: str, service: Any) -> dict:
        self.refresh()
        row = self.row(group)
        request = self.corrections.requests().get(request_id)
        candidate, selected, pair = self.candidate(group)
        require(candidate is not None and request is not None and request["state"] == "approved"
                and request["evidence"] == {**candidate, "pair_result_sha256": ADAPTER.digest(pair)}
                and request["loop_state_digest"] == ADAPTER.digest({"selection": selected, "pair": pair}),
                "현재 선택·검사와 교정 승인 불일치 또는 소비된 요청")
        ADAPTER.verify_correction_service(self.context, service)
        require(service.preflight() is True, "실제 NONE source cache가 필요함")
        entry = next(e for e in self.context["protocol"]["groups"] if e["group"] == group)
        ADAPTER.require_action_window(self.context, entry, "correction")
        def action() -> dict:
            approval = {"path": str(self.corrections.path.resolve()), "sha256": ADAPTER.file_sha(self.corrections.path),
                        "request_id": request_id, "loop_state_digest": request["loop_state_digest"]}
            result = ADAPTER.run_correction(self.context, group, approval, row["corrected_job_id"], service.train, service.evaluate)
            if result.get("status") == "complete":
                ADAPTER.validate_correction_result(self.context, group, result)
                ADAPTER._write_new(self.path / "corrections" / f"{group}.json", result)
            return result
        return consumed_action(self.corrections, request_id, group, action, self.path / "failures" / f"{group}.json")

    def recording_bindings(self) -> dict:
        self.refresh()
        verified = {}
        # 두 결과를 먼저 검증한 뒤에만 승인 export를 쓴다.
        for group in SEEDS:
            path = self.path / "corrections" / f"{group}.json"
            correction = ADAPTER.validate_correction_result(self.context, group, ADAPTER._read(path))
            require(correction["fixed_eval"] == {"none": "healthy", "corrected": "healthy"}, "두 seed 교정 완료 전 bind 금지")
            verified[group] = path
        bindings = {}
        for group, path in verified.items():
            approval = self.path / "record-approvals" / f"{group}.json"
            record_approval(self.corrections, self.row(group), path, approval)
            bindings[f"corrected-s{SEEDS[group]}"] = {"correction_result": path, "approval_receipt": approval}
        return bindings

    def bind_recording(self, payload: dict) -> dict:
        runner, _ = native_components()
        bindings = self.recording_bindings()
        intent = self.path / "intents/bind-native-protocol.json"
        require(not intent.exists(), "두 seed native bind_protocol은 한 번만 호출")
        template_ref = self.context["protocol"]["execution_bindings"]["corrected_render_plan"]
        template, _ = ADAPTER._ref(self.root, template_ref, "corrected native template")
        runner.validate_template(template)
        output = Path(payload["output"])
        require(not output.exists(), "bound native protocol 출력은 새 경로여야 함")
        ADAPTER._write_new(intent, {"bindings": {k: {n: ref(p) for n, p in v.items()} for k, v in bindings.items()},
            "payload_sha256": ADAPTER.digest(payload), "output": str(output.resolve())})
        value = runner.bind_protocol(template, bindings, source_commit=payload["source_commit"],
            source_sha256_lf=payload["source_sha256_lf"], recording_target=payload["recording_target"],
            resources=payload["resources"], registered_at_utc=payload["registered_at_utc"], output=output)
        return {"status": "native_protocol_bound_requires_main_commit_and_fresh_resource",
                "bound_protocol": ref(output), "checkpoints": len(value["checkpoints"]), "gpu_executions": 0}

    def observe(self, group: str, payload: dict, complete: Callable) -> dict:
        self.refresh()
        runner, native = native_components()
        bindings = self.recording_bindings()
        logical = f"corrected-s{SEEDS[group]}"
        binding = bindings[logical]
        protocol = Path(payload["bound_protocol"]).resolve(strict=True)
        anchor = str(payload["recording_registration_anchor"])
        template, _ = ADAPTER._ref(self.root, self.context["protocol"]["execution_bindings"]["corrected_render_plan"],
                                   "pretraining corrected native template")
        verify_native_registration(self.root, payload["recording_registration_ref"], anchor, protocol, template)
        runner.validate_protocol(protocol)
        service = native.NativeCorrectedRecordService(protocol)
        ADAPTER.verify_record_service(self.context, service.record)
        cache = self.path / "native-record" / f"{group}.json"
        intent = self.path / "intents" / f"native-{group}.json"
        if not cache.exists():
            require(not intent.exists(), "native 녹화 소비 intent 재실행 금지")
            kwargs = record_kwargs(payload, logical, binding["correction_result"], binding["approval_receipt"])
            ADAPTER._write_new(intent, {"protocol": ref(protocol), "logical_id": logical,
                "resource_proof": {"path": str(kwargs["resource_proof"]), "sha256": kwargs["resource_sha256"]},
                "stage": str(kwargs["stage"]), "used_for_decision": False})
            result = service.record(**kwargs)
            if result.get("status") != "complete": return result
            recheck_native_result(result, protocol, logical, binding["correction_result"], binding["approval_receipt"])
            ADAPTER._write_new(cache, result)
        result = ADAPTER._read(cache)
        receipt = recheck_native_result(result, protocol, logical, binding["correction_result"], binding["approval_receipt"])
        contract = native_contract(receipt, receipt["checkpoint_sha256"])
        binding = OBSERVER.binding_template(contract, run_id=f"{self.plan['execution_id']}-post-{group}",
            source_anchor=self.context["protocol"]["source_anchor"], protocol_sha256=self.context["protocol_sha256"])
        _, stage, manifest = CALL._plan(self.plan["selection_plan"])
        require(manifest["phase"] == "e2e" and manifest["api_cap"] == 32, "후속 관찰 API 원장 혼합 금지")
        journal = JOURNAL.DecisionJournal(self.path / "post-observation" / group / "decision", stage / "api",
            OBSERVER.checked_binding(contract, binding), OBSERVER.REQUEST_PARAMS, retry_delay_s=30, api_cap=32)
        observation = journal.advance(lambda fn: OBSERVER.engine(contract, fn), complete, lambda: datetime.now(timezone.utc))
        return {"status": observation["status"], "native_receipt": result["native_receipt"],
                "observation": observation, "used_for_decision": False, "development_scope": DEVELOPMENT}



def runtime_cli(args: argparse.Namespace, *, complete: Callable | None = None) -> dict:
    """등록/lease 확인 이후에만 provider와 생산 backend를 생성한다."""
    validate_method()
    root = args.repo.resolve(strict=True)
    def portable(path: Path, sha: str) -> dict:
        return {"path": path.resolve(strict=True).relative_to(root).as_posix(), "sha256": sha}
    active = args.operation not in {"bind-recording", "observe"}
    context = ADAPTER.verify_context(root, portable(args.execution_protocol, args.execution_protocol_sha256),
                                    portable(args.lease, args.lease_sha256), require_active=active)
    runtime = Runtime(root, args.plan, context, args.scope, require_active=active)
    if getattr(args, "relay_config", None) is not None:
        _, scope = read_ref(runtime.scope)
        require(scope.get("model_relay_config") == ref(args.relay_config), "등록 relay config SHA/경로 불일치")
        if complete is None:
            complete = RelayComplete(ADAPTER._read(args.relay_config), runtime.path / "model-relay")
    callback = complete if complete is not None else OBSERVER.default_complete
    group, operation = args.group, args.operation
    payload = ADAPTER._read(args.payload) if args.payload else {}
    if operation == "verify":
        return {"status": "verified", "execution_authorized": True, "api_cap": 32, "groups": list(SEEDS)}
    if operation == "initial":
        return runtime.initial(group, callback)
    if operation == "probe":
        return runtime.probe(group, Path(payload["probe_protocol"]), Path(payload["archive"]),
                             Path(payload["lab"]), Path(payload["python"]), payload["approver"])
    if operation == "propose":
        return runtime.propose(group, callback)
    if operation == "approve":
        return runtime.approve(group, payload["request_id"], payload["approver"])
    if operation == "bind-recording":
        return runtime.bind_recording(payload)
    if operation == "correct":
        bindings = context["protocol"]["execution_bindings"]
        registered = bindings["correction_backend"]
        require(registered == {"path": "evals/vision_transferred_e2e_v2.py",
                "sha256_lf": ADAPTER.file_sha(Path(__file__), lf=True), "class": "TransferredTrainingBackend"},
                "transferred source cache backend 등록 불일치")
        source_path = ADAPTER._file_ref(root, {"path": bindings["source_protocol"]["path"],
            "sha256": bindings["source_protocol"]["sha256_lf"]}, "source protocol", lf=True)
        source = ADAPTER._read(source_path)
        _, blueprint = read_ref(runtime.plan["transferred_blueprint"])
        backend = TransferredTrainingBackend.from_runtime(Path(payload["lab"]), Path(payload["python"]),
            {**source, "protocol_path": str(source_path), "protocol_sha256_lf": ADAPTER.file_sha(source_path, lf=True)},
            context["protocol"]["source_anchor"], root / ".triage-runtime/vision-brev" / source["protocol_id"],
            source["training_plan"], resource_proof=context["resource"], source_cache_reference=blueprint["source_cache"])
        jobs = target_jobs(source, {"seed_group": group, "source_seed": SEEDS[group]})
        service = ADAPTER.CorrectionService(backend, backend.runtime_root, jobs["NONE"], f"e2e-{group}-corrected")
        return runtime.correct(group, payload["request_id"], service)
    return runtime.observe(group, payload, callback)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-sources")
    audit.add_argument("--archive", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, default=ROOT)
    prepare.add_argument("--selection-protocol", type=Path, required=True)
    prepare.add_argument("--input-locations", type=Path, required=True)
    prepare.add_argument("--source-cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--execution-id", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--repo", type=Path, default=ROOT)
    seal.add_argument("--blueprint", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--source-anchor", required=True)
    activate = sub.add_parser("activate-selection")
    activate.add_argument("--repo", type=Path, default=ROOT)
    activate.add_argument("--plan", type=Path, required=True)
    registration = sub.add_parser("execution-registration")
    registration.add_argument("--execution-protocol", type=Path, required=True)
    registration.add_argument("--plan", type=Path, required=True)
    registration.add_argument("--output", type=Path, required=True)
    native_seal = sub.add_parser("native-registration")
    native_seal.add_argument("--bound-protocol", type=Path, required=True)
    native_seal.add_argument("--template", type=Path, default=ROOT / "bench/protocols/vision_corrected_pull_record_template_v1.json")
    native_seal.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("check-draft")
    inspect.add_argument("--protocol", type=Path, default=METHOD)
    run = sub.add_parser("run")
    run.add_argument("--operation", choices=("verify", "initial", "probe", "propose", "approve", "correct", "bind-recording", "observe"), required=True)
    run.add_argument("--group", choices=tuple(SEEDS), required=True)
    run.add_argument("--repo", type=Path, default=ROOT)
    for name in ("plan", "scope", "execution-protocol", "lease"):
        run.add_argument("--" + name, type=Path, required=True)
    for name in ("execution-protocol-sha256", "lease-sha256"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--payload", type=Path)
    run.add_argument("--relay-config", type=Path, help="host scope에 SHA 등록한 relay 설정; 외부 driver가 complete callback 주입")
    run.add_argument("--result-output", type=Path, help="한 단계 결과를 새 JSON에 보존")
    args = parser.parse_args(argv)
    if args.command == "audit-sources":
        result = audit_source_cache(args.archive, args.output)
        print(json.dumps({"source_jobs": len(result["jobs"]), "files_verified": result["archive_verification"]["files_verified"]}))
    elif args.command == "prepare":
        result = prepare_blueprint(args.repo, args.selection_protocol, args.input_locations, args.source_cache,
                                   args.output, execution_id=args.execution_id)
        print(json.dumps({"status": result["status"], "targets": len(result["rows"])}))
    elif args.command == "seal":
        result = seal_plan(args.repo, args.blueprint, args.output, source_anchor=args.source_anchor)
        print(json.dumps({"selection_plan": result["selection_plan"], "execution_authorized": False}))
    elif args.command == "run":
        require(args.result_output is None or not args.result_output.exists(), "단계 결과 출력은 새 파일이어야 함")
        try:
            result = runtime_cli(args)
        except RelayPending as error:
            result = {"status": "blocked_pending", "reason": str(error), "automatic_resend": False}
        if args.result_output is not None:
            ADAPTER._write_new(args.result_output, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif args.command == "activate-selection":
        print(json.dumps(activate_selection(args.repo, args.plan), ensure_ascii=False))
    elif args.command in {"execution-registration", "native-registration"}:
        value = execution_registration(args.execution_protocol, args.plan) if args.command == "execution-registration" else native_registration(args.bound_protocol, args.template)
        ADAPTER._write_new(args.output, value)
        print(json.dumps({"status": "registration_seal_created_main_commit_required", "seal": ref(args.output)}))
    else:
        protocol = validate_method(args.protocol)
        require(protocol.get("registered_before_execution") is False, "이 검사는 실행 전 초안 전용")
        print(json.dumps({"status": "HOLD", "reason": "draft_not_committed_no_host_use_or_resource_approval", "gpu": 0, "api": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

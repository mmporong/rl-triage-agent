"""ANYmal-C U1 조합 실행 v2의 공개 결과를 만들고 검증한다.

허용한 산출물만 옮기고, 사용자 경로가 든 문자열은 자리표시로 바꾼다.
원본 LF SHA와 공개본 LF SHA를 provenance에 함께 남긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAG = "combination_anymal_u1_s2151_20261017"
SOURCE = ROOT / "bench/private/codex_support_20261015/task39/runner/runs" / TAG
OUTPUT = ROOT / "evals/results" / TAG
PROTOCOL = "bench/protocols/combination_trial_v2.json"
ARMS = ("source_none", "source_u1", "family_restart")
TRAIN_FILES = ("manifest.json", "receipt.json", "correction.json", "family_environment.json",
               "training_evidence.json", "close_ready.json")
FILES = (
    ["result.json", "seed_freshness.json", "combination_manifest.json", "combination_state.json",
     "contract_observation.json", "contract_request.json", "correction_ledger.jsonl",
     "source_eval/source_none.json", "source_eval/source_u1.json", "restart_eval/family_restart.json"]
    + [f"{arm}/train/{name}" for arm in ARMS for name in TRAIN_FILES]
)
ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|/home/|/mnt/")
EVALUATIONS = {"source_none": "source_eval/source_none.json", "source_u1": "source_eval/source_u1.json",
               "family_restart": "restart_eval/family_restart.json"}


def sha_lf(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _prefixes(isaaclab: Path) -> list[tuple[str, str]]:
    roots = [(ROOT, "<repo>"), (isaaclab, "<isaaclab>")]
    sim = isaaclab / "_isaac_sim"
    if sim.exists():
        roots.append((sim.resolve(), "<isaacsim>"))
    roots.append((Path.home(), "<home>"))
    pairs = []
    for path, label in roots:
        for text in {str(path), str(path.resolve()), path.as_posix(), path.resolve().as_posix()}:
            pairs.append((text, label))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def _scrub(value: Any, prefixes: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _scrub(item, prefixes) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, prefixes) for item in value]
    if isinstance(value, str) and ABSOLUTE.search(value):
        lowered = value.lower()
        for prefix, label in prefixes:
            if lowered.startswith(prefix.lower()):
                return label + value[len(prefix):].replace("\\", "/")
        raise ValueError(f"자리표시로 바꿀 수 없는 절대경로가 있다: {value[:80]}")
    return value


def _public_bytes(path: Path, prefixes: list[tuple[str, str]]) -> tuple[bytes, bool]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if not ABSOLUTE.search(text):
        return raw.replace(b"\r\n", b"\n"), False
    if path.suffix == ".jsonl":
        rows = [json.dumps(_scrub(json.loads(line), prefixes), ensure_ascii=False, sort_keys=True)
                for line in text.splitlines() if line.strip()]
        out = "\n".join(rows) + "\n"
    else:
        out = json.dumps(_scrub(json.loads(text), prefixes), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if ABSOLUTE.search(out):
        raise ValueError(f"공개본에 절대경로가 남았다: {path.name}")
    return out.encode("utf-8"), True


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(folder: Path) -> dict[str, Any]:
    result = _read(folder / "result.json")
    metrics = {arm: _read(folder / rel)["metrics"] for arm, rel in EVALUATIONS.items()}
    standstill = _read(folder / EVALUATIONS["source_none"])["standstill"]
    state = result["contract"]
    manifest = _read(folder / "combination_manifest.json")
    return {
        "schema": "combination_trial_public_summary_v2",
        "tag": TAG, "protocol": PROTOCOL, "protocol_sha256_lf": result["protocol_sha256_lf"],
        "experiment_commit": result["source_commit_sha"], "seeds": result["seeds"],
        "combination": {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")},
        "status": result["status"], "outcome": result["outcome"], "wording": result["wording"],
        "source_gate": result["source_gate"],
        "contract_status": state["status"],
        "contract_stages": [{"stage": row["stage"], "status": row["status"]} for row in state["trace"]],
        "restoration_targets": state["restoration_targets"],
        "restart_verdict": result["restart_verdict"],
        "correction_receipt": {key: result["correction_receipt"][key]
                               for key in ("verdict", "over_budget", "gpu_s", "exit_code")},
        "fixed_eval_metrics": metrics, "standstill_reference": standstill,
        "model_calls": result["model_calls"], "selection_actor": result["selection_actor"],
        "owned_gpu_process_wall_s_by_stage": result["owned_gpu_process_wall_s_by_stage"],
        "time_totals": result["time_totals"],
    }


def export(isaaclab: Path) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError("공개 결과 폴더가 이미 있다. 덮어쓰지 않는다")
    prefixes = _prefixes(isaaclab)
    staged: dict[str, bytes] = {}
    provenance = []
    for rel in FILES:
        source = SOURCE / rel
        data, scrubbed = _public_bytes(source, prefixes)
        staged[rel] = data
        provenance.append({"path": rel, "source_sha256_lf": sha_lf(source.read_bytes()),
                           "public_sha256_lf": sha_lf(data), "paths_replaced": scrubbed})
    summary = _summary(SOURCE)
    if summary["correction_receipt"]["verdict"] != summary["outcome"]:
        raise ValueError("원장 receipt 판정과 결과 판정이 다르다")
    staged["summary.json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    staged["provenance.json"] = (json.dumps({
        "schema": "combination_trial_public_provenance_v2", "tag": TAG,
        "excluded": "lease, GPU 점유 조회, 실행 launch·execution, stdout·stderr, 빈 records",
        "path_placeholders": ["<repo>", "<isaaclab>", "<isaacsim>", "<home>"],
        "files": provenance,
        "summary_sha256_lf": sha_lf(staged["summary.json"]),
    }, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for rel, data in staged.items():
        target = OUTPUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
    return verify()


def verify(folder: Path = OUTPUT) -> dict[str, Any]:
    provenance = _read(folder / "provenance.json")
    expected = {row["path"] for row in provenance["files"]} | {"summary.json", "provenance.json"}
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != expected or set(FILES) != {row["path"] for row in provenance["files"]}:
        raise ValueError("공개 파일 집합이 provenance와 다르다")
    for row in provenance["files"]:
        data = (folder / row["path"]).read_bytes()
        if sha_lf(data) != row["public_sha256_lf"] or ABSOLUTE.search(data.decode("utf-8")):
            raise ValueError(f"공개 파일이 provenance와 다르거나 절대경로가 있다: {row['path']}")
        if not row["paths_replaced"] and row["public_sha256_lf"] != row["source_sha256_lf"]:
            raise ValueError(f"그대로 옮긴 파일의 SHA가 원본과 다르다: {row['path']}")
    summary_bytes = (folder / "summary.json").read_bytes()
    if sha_lf(summary_bytes) != provenance["summary_sha256_lf"]:
        raise ValueError("summary SHA가 provenance와 다르다")
    if _summary(folder) != json.loads(summary_bytes):
        raise ValueError("summary를 공개 파일에서 다시 계산한 값과 다르다")
    summary = json.loads(summary_bytes)
    expected_outcome = {"healthy": "recovered", "unhealthy": "not_recovered"}.get(summary["restart_verdict"]["label"])
    if (summary["source_gate"]["outcome"] != "failure_present" or summary["contract_status"] != "ready"
            or any(row["status"] != "passed" for row in summary["contract_stages"])
            or summary["outcome"] != expected_outcome or summary["correction_receipt"]["verdict"] != expected_outcome
            or summary["model_calls"] != 0):
        raise ValueError("공개 summary의 관문·판정 연결이 다르다")
    return {"status": "valid", "files": len(actual), "outcome": summary["outcome"],
            "contract_status": summary["contract_status"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    try:
        out = export(args.isaaclab) if args.command == "export" else verify()
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

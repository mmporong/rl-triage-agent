"""T1 holdout 학습 뒤 짝별 blind 입력을 등록·생성한다. 정답표는 복사하지 않는다.

python bench/build_workspace_t1.py register
python bench/build_workspace_t1.py build
등록한 텔레메트리·기준 params·manifest와 이 코드를 커밋한 뒤 순위를 실행한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.leakcheck import OUTPUT_DIRS, blind_markers, check_blind_workspace, sha256  # noqa: E402

USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/")
FAULT_ID = re.compile(r"(?<!\w)X[123](?:_REF)?(?!\w)")


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cells(key: dict) -> list[dict]:
    seeds = key["holdout_seeds"]
    if seeds != [2026, 2027, 2028] or not key["cases"]:
        raise ValueError("T1 holdout seed·사례가 사전등록과 다르다")
    out = []
    for seed in seeds:
        for cid, case in sorted(key["cases"].items()):
            ref = case["reference"]
            if not re.fullmatch(r"e\d{2}", cid) or ref not in (f"ref_{cid}", "baseline_p0c"):
                raise ValueError("허용되지 않은 사례·기준 실행 이름")
            out.append({"case_id": cid, "seed": seed, "run": f"{cid}_s{seed}",
                        "reference": f"{ref}_s{seed}", "workspace": f"workspace_t1/seed{seed}_{cid}"})
    return out


def telemetry(name: str) -> Path:
    folder = "runs_p0c" if name.startswith("baseline_p0c_") else "runs_t1"
    return ROOT / "bench" / folder / f"{name}.telemetry.json"


def check_text(path: Path, key: dict) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    cases = [{"case_id": c, "overrides": []} for c in key["cases"]]
    markers = blind_markers(cases, key) + [str(key["secret_seed"])]
    if USER_PATH.search(text) or FAULT_ID.search(text) or any(m in text for m in markers):
        raise ValueError(f"blind 입력 경계 위반: {path.name}")


def register(key: dict, log_root: Path) -> None:
    manifest_path = ROOT / "bench/reference/t1_manifest.json"
    if manifest_path.exists():
        raise ValueError("T1 manifest가 이미 있다. 덮어쓰지 않는다")
    rows, copies = [], []
    for cell in cells(key):
        run, ref = telemetry(cell["run"]), telemetry(cell["reference"])
        ref_tele = json.loads(ref.read_text(encoding="utf-8"))
        run_dir = log_root / ref_tele["run_dir_name"]
        if not run_dir.name.endswith("_" + cell["reference"]):
            raise ValueError("기준 텔레메트리와 params 실행이 다르다")
        dst = ROOT / "bench/reference/params" / f"t1_{cell['case_id']}_seed{cell['seed']}"
        if dst.exists():
            raise ValueError("T1 기준 params 폴더가 이미 있다")
        params = {n: run_dir / "params" / n for n in ("env.yaml", "agent.yaml")}
        for path in (run, ref, *params.values()):
            check_text(path, key)
        rows.append({**cell, "telemetry": run.relative_to(ROOT).as_posix(),
                     "reference_telemetry": ref.relative_to(ROOT).as_posix(),
                     "params": dst.relative_to(ROOT).as_posix(), "reference_run_dir_name": run_dir.name,
                     "sha256_lf": {"telemetry": sha_lf(run), "reference_telemetry": sha_lf(ref)},
                     "params_sha256": {n: sha256(p) for n, p in params.items()}})
        copies.append((dst, params))
    # 모든 입력 검사가 끝난 뒤 사본을 만든다. YAML은 CRLF까지 원본대로 보존한다.
    for dst, params in copies:
        dst.mkdir(parents=True)
        for name, src in params.items():
            shutil.copyfile(src, dst / name)
    manifest_path.write_text(json.dumps({"kind": "t1_blind_inputs", "cells": rows}, ensure_ascii=False, indent=1)
                             + "\n", encoding="utf-8")


def validate_cell(cell: dict, key: dict, workspace: bool = False) -> Path:
    for field in ("telemetry", "reference_telemetry"):
        path = ROOT / cell[field]
        if sha_lf(path) != cell["sha256_lf"][field]:
            raise ValueError(f"T1 {field} SHA256 불일치")
        check_text(path, key)
    params = ROOT / cell["params"]
    for name, want in cell["params_sha256"].items():
        path = params / name
        if sha256(path) != want:
            raise ValueError("T1 기준 params SHA256 불일치")
        check_text(path, key)
    if json.loads((ROOT / cell["reference_telemetry"]).read_text(encoding="utf-8"))["run_dir_name"] != cell["reference_run_dir_name"]:
        raise ValueError("T1 기준 실행 이름 불일치")
    ws = ROOT / cell["workspace"]
    if workspace:
        expected = {"reference/telemetry.json": ROOT / cell["reference_telemetry"],
                    f"cases/{cell['case_id']}/telemetry.json": ROOT / cell["telemetry"],
                    **{f"reference/params/{n}": params / n for n in cell["params_sha256"]}}
        for path in ws.rglob("*"):
            rel = path.relative_to(ws)
            if path.is_file() and rel.parts[0] not in OUTPUT_DIRS and rel.as_posix() not in expected:
                raise ValueError("T1 작업공간 유출 검사 실패: 허용 목록 밖 입력")
        for rel, src in expected.items():
            if sha256(ws / rel) != sha256(src):
                raise ValueError("T1 작업공간 입력이 등록 사본과 다르다")
        problems = check_blind_workspace(ws, [{"case_id": cell["case_id"], "overrides": []}],
                                         key, cell["params_sha256"])
        if problems:
            raise ValueError("T1 작업공간 유출 검사 실패: " + "; ".join(problems))
    return ws


def load_manifest(key: dict) -> dict:
    manifest = json.loads((ROOT / "bench/reference/t1_manifest.json").read_text(encoding="utf-8"))
    expected = cells(key)
    if len(manifest["cells"]) != len(expected):
        raise ValueError("T1 등록 사례 수 불일치")
    for registered, want in zip(manifest["cells"], expected):
        if any(registered[k] != v for k, v in want.items()):
            raise ValueError("T1 등록 사례·기준 대응 불일치")
    return manifest


def build(key: dict) -> None:
    manifest = load_manifest(key)
    for cell in manifest["cells"]:
        ws = validate_cell(cell, key)
        if ws.exists():
            raise ValueError("T1 작업공간이 이미 있다. 덮어쓰지 않는다")
    for cell in manifest["cells"]:
        ws = ROOT / cell["workspace"]
        (ws / "reference/params").mkdir(parents=True)
        (ws / "cases" / cell["case_id"]).mkdir(parents=True)
        shutil.copyfile(ROOT / cell["reference_telemetry"], ws / "reference/telemetry.json")
        shutil.copyfile(ROOT / cell["telemetry"], ws / "cases" / cell["case_id"] / "telemetry.json")
        for name in cell["params_sha256"]:
            shutil.copyfile(ROOT / cell["params"] / name, ws / "reference/params" / name)
        validate_cell(cell, key, workspace=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("register", "build"))
    ap.add_argument("--run-params", type=Path,
                    default=Path.home() / "IsaacLab/logs/rsl_rl/unitree_go2_flat")
    args = ap.parse_args()
    key = json.loads((ROOT / "bench/private/answer_key_t1.json").read_text(encoding="utf-8"))
    if args.stage == "register":
        register(key, args.run_params)
    else:
        build(key)
    print(f"T1 {args.stage}: {len(cells(key))} cells, blind input check passed")


if __name__ == "__main__":
    main()

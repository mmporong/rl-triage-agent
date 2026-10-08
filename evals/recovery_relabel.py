"""P0-B1: 저장된 실행 전부에 행동 기반 회복 판정을 적용하고 현행 RECOVERY_BAND 판정과 달라지는 실행을 기록한다.

사용: python evals/recovery_relabel.py --tag p0b1_relabel_20261008   (또는 --out <새 폴더>)

판정 계약은 docs/P0-B1-RECOVERY.md, 구현은 src/rl_triage/recovery.py. Isaac을 다시 돌리지 않고 bench/runs의
학습 텔레메트리만 쓴다. 실행 설정은 기준 params(bench/reference/params)에 실행 이름의 override를 적용해 얻는다.
같은 학습 실행을 가리키는 텔레메트리(run_dir_name 동일)는 하나만 센다.
출력: <폴더>/relabel.json. 모델·네트워크·GPU를 쓰지 않는다. 기존 폴더는 덮어쓰지 않는다. 종료 코드 0 완료, 2 입력 오류.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "bench"))
from catalog import BENIGN  # noqa: E402
from catalog_v2 import CANDIDATES  # noqa: E402
from replay import _sha256_lf, _show, code_version  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage import rules as R  # noqa: E402
from rl_triage.eval_bridge import RECOVERY_BAND, recovery_verdict  # noqa: E402
from rl_triage.leakcheck import sha256  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

RUNS = ROOT / "bench" / "runs"
MANIFEST_PATH = ROOT / "bench" / "reference" / "manifest.json"
CASES_PATH = ROOT / "bench" / "cases.json"
CODE_FILES = ("evals/recovery_relabel.py", "src/rl_triage/recovery.py", "src/rl_triage/rules.py",
              "src/rl_triage/eval_bridge.py", "bench/catalog.py", "bench/catalog_v2.py")


class InputError(Exception):
    """입력 오류(종료 코드 2)."""


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise InputError(f"{_show(path)}: 읽을 수 없다 ({type(e).__name__})") from None


def _ref_flat(seed: int, manifest: dict) -> dict:
    ref = manifest["seeds"].get(str(seed))
    if ref is None:
        raise InputError(f"공개 기준 params가 없는 seed {seed}")
    params = ROOT / "bench" / "reference" / "params" / f"seed{seed}"
    for name, want in ref["sha256"].items():
        if sha256(params / name) != want:
            raise InputError(f"{_show(params / name)}: SHA256이 manifest와 다르다")
    load = lambda n: yaml.load((params / n).read_text(encoding="utf-8"), Loader=_TolerantLoader)  # noqa: E731
    return R.flat_config(load("env.yaml"), load("agent.yaml"))


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def relabel(manifest: dict, cases: dict) -> dict:
    v2 = {cid: ov for cid, ov, _ in CANDIDATES}
    files = sorted(RUNS.glob("*.telemetry.json"))
    if not files:
        raise InputError("bench/runs에 텔레메트리가 없다")
    seen, runs, duplicates, unknown, inputs = {}, {}, [], [], []
    for f in files:
        name = f.name.removesuffix(".telemetry.json")
        tele = _json(f)
        inputs.append({"path": _show(f), "sha256_lf": _sha256_lf(f)})
        if tele["run_dir_name"] in seen:
            duplicates.append({"name": name, "same_run_as": seen[tele["run_dir_name"]]})
            continue
        overrides = RC.run_overrides(name, cases, list(BENIGN.values()), v2)
        if overrides is None:
            unknown.append(name)
            continue
        seen[tele["run_dir_name"]] = name
        runs[name] = {"tele": tele, "overrides": overrides, "seed": RC.seed_of(name)}
    if unknown:
        raise InputError(f"설정을 알 수 없는 실행: {unknown}")
    flats = {s: _ref_flat(s, manifest) for s in sorted({r["seed"] for r in runs.values()})}
    refs = {}
    for s in flats:
        ref_path = ROOT / manifest["seeds"][str(s)]["telemetry"]
        ref_name = ref_path.name.removesuffix(".telemetry.json")
        if ref_name not in runs:
            raise InputError(f"seed {s} 기준 실행 {ref_name}이 없다")
        refs[s] = ref_name
    beh = {n: RC.behavior(r["tele"]["summary"], RC.run_config(flats[r["seed"]], r["overrides"], R.apply_overrides))
           for n, r in runs.items()}
    base = [refs[s] for s in sorted(refs)]
    normal = [{"pair": f"{a} vs {b}", **RC.compare(beh[a], beh[b])} for a in base for b in base if a != b]
    normal += [{"pair": f"{n} vs {refs[r['seed']]}", **RC.compare(beh[n], beh[refs[r["seed"]]])}
               for n, r in runs.items() if n.startswith("benign_all_")]
    bands = RC.derive_bands(normal)
    rows, changes = [], []
    for n, r in runs.items():
        ref_name = refs[r["seed"]]
        c = RC.compare(beh[n], beh[ref_name])
        v = RC.verdict(c, bands)
        cur = recovery_verdict(r["tele"], runs[ref_name]["tele"])
        row = {"run": n, "seed": r["seed"], "reference": ref_name, "is_reference": n == ref_name,
               "current_recovered": cur["recovered"],
               "current_failed": [t for t, ch in cur["checks"].items() if not ch["pass"]],
               "behavior_label": v["label"], "behavior_failed": v["failed"], "short_horizon": v["short_horizon"],
               "behavior": _round(beh[n]), "compared": _round(c)}
        rows.append(row)
        agree = (cur["recovered"] and v["label"] == "healthy") or (not cur["recovered"] and v["label"] == "unhealthy")
        if not agree and not row["is_reference"]:
            changes.append({k: row[k] for k in ("run", "current_recovered", "current_failed", "behavior_label",
                                                "behavior_failed")})
    v2_rows = []
    for cid, ov, note in CANDIDATES:
        by_seed = {r["seed"]: r for r in rows if r["run"].startswith(f"v2_{cid}_s")}
        cur = [r["current_recovered"] for r in by_seed.values()]
        new = [r["behavior_label"] for r in by_seed.values()]
        old_label = "missing" if not cur else "benign" if all(cur) else "harmful" if not any(cur) else "ambiguous"
        new_label = ("missing" if not new else "undetermined" if "undetermined" in new else
                     "benign" if all(x == "healthy" for x in new) else
                     "harmful" if all(x == "unhealthy" for x in new) else "ambiguous")
        v2_rows.append({"id": cid, "override": ov, "seeds": sorted(by_seed), "current_label": old_label,
                        "behavior_label": new_label})
    return {"runs": rows, "changes": changes, "bands": _round(bands), "normal": [_round(x) for x in normal],
            "duplicates": duplicates, "v2_labels": v2_rows, "inputs": inputs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="행동 기반 회복 판정으로 저장된 실행을 다시 판정한다(모델·네트워크·GPU 없음)")
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--tag", help="evals/results/<tag>/에 쓴다(기존 폴더는 덮어쓰지 않음)")
    dest.add_argument("--out", type=Path, help="새 폴더에 쓴다")
    args = ap.parse_args(argv)
    try:
        if args.tag is not None:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
                raise InputError(f"--tag는 영문·숫자·._- 로 된 폴더 이름이어야 한다: {args.tag!r}")
            out_dir = ROOT / "evals" / "results" / args.tag
        else:
            out_dir = args.out.resolve()
        if out_dir.exists():
            raise InputError(f"{_show(out_dir)}가 이미 있다. 결과는 덮어쓰지 않는다")
        manifest = _json(MANIFEST_PATH)
        cases = {c["case_id"]: c["overrides"] for c in _json(CASES_PATH)}
        result = relabel(manifest, cases)
    except (InputError, ValueError) as e:
        print(f"relabel 입력 오류: {e}", file=sys.stderr)
        return 2
    inputs = [{"path": _show(p), "sha256_lf": _sha256_lf(p)} for p in (MANIFEST_PATH, CASES_PATH)] + result.pop("inputs")
    command = ["python", "evals/recovery_relabel.py"] + (["--tag", args.tag] if args.tag else ["--out", _show(out_dir)])
    report = {
        "kind": "recovery_relabel", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(command),
        "contract": {"doc": "docs/P0-B1-RECOVERY.md", "behavior_keys": list(RC.BEHAVIOR_KEYS), "margin": RC.MARGIN,
                     "current_band": {k: list(v) for k, v in RECOVERY_BAND.items()}},
        **result, "inputs": inputs, "code": code_version(["bench/runs", "bench/reference"], CODE_FILES),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "model_calls": 0, "network": "not used", "gpu": "not used"},
    }
    out_dir.mkdir(parents=True)
    (out_dir / "relabel.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {}
    for r in report["runs"]:
        key = (r["current_recovered"], r["behavior_label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"runs={len(report['runs'])} duplicates={len(report['duplicates'])} changes={len(report['changes'])}")
    print("bands " + " ".join(f"{k}={v}" for k, v in report["bands"].items()))
    for (cur, lab), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  current_recovered={cur!s:5s} behavior={lab:12s} {n}")
    for ch in report["changes"]:
        print(f"  CHANGE {ch['run']:22s} current={ch['current_recovered']!s:5s} {ch['current_failed']} "
              f"-> {ch['behavior_label']} {ch['behavior_failed']}")
    for v in report["v2_labels"]:
        if v["current_label"] != v["behavior_label"]:
            print(f"  V2 {v['id']} {v['override']}: {v['current_label']} -> {v['behavior_label']}")
    print(f"saved {_show(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""P0-A2 결정적 기준선(LLM 없음)을 저장된 텔레메트리에 적용하고 과제 B와 같은 규칙으로 채점한다.

사용:
  python evals/baselines.py --seeds 7 42 --tag p0a2_dev_20261008
  python evals/baselines.py --seeds 123 --tag p0a2_holdout_20261008   # 규칙·입력 파일이 커밋된 상태에서만
  [--out <새 폴더>]                 # --tag 대신 임의 위치에 쓴다(테스트용)
  [--run-params <Isaac Lab 로그 폴더>]  # B0 입력(기준 params + override)이 실제 실패 실행 params와 같은지 대조

기준선(src/rl_triage/rules.py, docs/P0-A2-BASELINES.md)
- rule_prior: 텔레메트리 없음. dev 정답 빈도순.
- rule_features: 정규화 특징 + 고정 규칙. 과제 B 에이전트와 같은 정보(실패·기준 텔레메트리, 기준 params).
- rule_template: 같은 특징으로 가장 가까운 dev 사례. 평가 seed의 정답은 쓰지 않는다(dev seed는 서로를 쓴다).
- rule_b0: 기준 params에 override를 적용한 설정 diff. 실패 실행 설정을 볼 수 있는 조건이다.

출력: <폴더>/seed<N>.jsonl(evals/replay.py로 재채점), <폴더>/baselines.json(입력·코드 SHA256, 임계값, 표, 명령).
모델·네트워크·GPU를 쓰지 않는다. 기존 폴더는 덮어쓰지 않는다. 종료 코드: 0 완료, 2 입력 오류.
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
from replay import _sha256_lf, _show, code_version  # noqa: E402
from rl_triage import rules as R  # noqa: E402
from rl_triage.leakcheck import sha256  # noqa: E402
from rl_triage.scoring import score_blind  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

KEY_PATH = ROOT / "bench" / "answer_key.json"
CASES_PATH = ROOT / "bench" / "cases.json"
MANIFEST_PATH = ROOT / "bench" / "reference" / "manifest.json"
RUNS = ROOT / "bench" / "runs"
KEY_P0C = ROOT / "bench" / "private" / "answer_key_p0c.json"  # P0-C 평가가 끝나면 공개본으로 옮긴다
# 벤치별 기준 params·텔레메트리 위치. p0c는 숨은 결함 holdout(docs/P0-C-HOLDOUT.md)이다.
BENCHES = {"v1": {"manifest_key": "seeds", "params": "seed", "runs": RUNS},
           "p0c": {"manifest_key": "p0c_seeds", "params": "p0c_seed", "runs": ROOT / "bench" / "runs_p0c"}}
CODE_FILES = ("evals/baselines.py", "evals/replay.py", "src/rl_triage/rules.py", "src/rl_triage/scoring.py",
              "src/rl_triage/triage_tools.py", "src/rl_triage/leakcheck.py")
INPUT_DIRS = ("bench/runs", "bench/reference")
MODES = {"rule_prior": "none", "rule_features": "telemetry", "rule_template": "telemetry+dev_labels",
         "rule_b0": "config_diff"}


class InputError(Exception):
    """입력 오류(종료 코드 2)."""


def _yaml(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_TolerantLoader)


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputError(f"{_show(path)}: 파일이 없다") from None
    except json.JSONDecodeError as e:
        raise InputError(f"{_show(path)}: 잘못된 JSON ({e.msg})") from None


def load_seed(seed: int, manifest: dict, cases: list[dict], inputs: list[dict], bench: str = "v1") -> dict:
    """seed 하나의 기준 params·기준 텔레메트리·실패 텔레메트리. manifest SHA256·실행 이름과 대조한다."""
    spec = BENCHES[bench]
    ref = manifest.get(spec["manifest_key"], {}).get(str(seed))
    if ref is None:
        raise InputError(f"공개 기준 params가 없는 {bench} seed {seed}")
    params = ROOT / "bench" / "reference" / "params" / f"{spec['params']}{seed}"
    for name, want in ref["sha256"].items():
        if not (params / name).exists() or sha256(params / name) != want:
            raise InputError(f"{_show(params / name)}: 없거나 SHA256이 manifest와 다르다")
        inputs.append({"path": _show(params / name), "sha256_lf": _sha256_lf(params / name)})
    ref_path = ROOT / ref["telemetry"]
    ref_tele = _json(ref_path)
    if ref_tele.get("run_dir_name") != ref["source_run_dir_name"]:
        raise InputError(f"{_show(ref_path)}: 기준 텔레메트리 실행 이름이 manifest와 다르다")
    inputs.append({"path": _show(ref_path), "sha256_lf": _sha256_lf(ref_path)})
    runs = {}
    for case in cases:
        path = spec["runs"] / f"{case['case_id']}_s{seed}.telemetry.json"
        runs[case["case_id"]] = _json(path)["summary"]
        inputs.append({"path": _show(path), "sha256_lf": _sha256_lf(path)})
    env, agent = _yaml(params / "env.yaml"), _yaml(params / "agent.yaml")
    return {"cfg": R.ref_config(env), "ref_flat": R.flat_config(env, agent), "ref": ref_tele["summary"], "runs": runs}


def freeze_problem(code: dict, seeds: list[int]) -> str | None:
    """dev 밖 seed는 규칙·입력이 커밋되어 바뀌지 않은 상태에서만 돌린다(규칙을 보고 고치지 못하게)."""
    holdout = [s for s in seeds if s not in R.DEV_SEEDS]
    if not holdout:
        return None
    if not code.get("git_head"):
        return f"holdout seed {holdout}: git 커밋을 확인할 수 없다"
    if code.get("git_dirty") is not False:
        return f"holdout seed {holdout}: 규칙·입력 파일에 커밋하지 않은 변경이 있다. 커밋한 뒤 실행한다"
    return None


def check_run_params(root: Path, seeds: list[int], data: dict, cases: list[dict]) -> dict:
    """실제 실패 실행 params의 diff가 기준 params + override의 diff와 같은지. 경로는 기록하지 않는다."""
    out = {"root": "<external>", "identity_keys_ignored": list(R.IDENTITY_KEYS), "checked": 0, "equal": 0,
           "missing": [], "mismatch": []}
    for seed in seeds:
        ref_flat = data[seed]["ref_flat"]
        for case in cases:
            cell = f"s{seed}/{case['case_id']}"
            dirs = [d for d in root.glob(f"*_{case['case_id']}_s{seed}") if (d / "params" / "env.yaml").exists()]
            if len(dirs) != 1:
                out["missing"].append(f"{cell} ({len(dirs)} dirs)")
                continue
            real = R.flat_config(_yaml(dirs[0] / "params" / "env.yaml"), _yaml(dirs[0] / "params" / "agent.yaml"))
            expected = R.apply_overrides(ref_flat, case["overrides"])
            real_diff, want_diff = R.config_diff(ref_flat, real), R.config_diff(ref_flat, expected)
            out["checked"] += 1
            if real_diff == want_diff:
                out["equal"] += 1
            else:
                got, want = {d["key"] for d in real_diff}, {d["key"] for d in want_diff}
                out["mismatch"].append({"cell": cell, "only_real": sorted(got - want), "only_overrides": sorted(want - got),
                                        "value_differs": sorted(k for k in got & want if real.get(k) != expected.get(k))})
    return out


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def run(seeds: list[int], key: dict, cases: list[dict], manifest: dict, inputs: list[dict], bench: str = "v1",
        dev: tuple[dict, list[dict]] | None = None) -> tuple[dict, dict, dict]:
    """dev=(v1 정답표, v1 cases)는 template 학습용이다. p0c를 평가할 때도 template은 v1 dev seed로만 학습한다."""
    dev_key, dev_cases = dev or (key, cases)
    dev_data = {s: load_seed(s, manifest, dev_cases, inputs, "v1") for s in R.DEV_SEEDS}
    data = {s: (dev_data[s] if bench == "v1" and s in dev_data else load_seed(s, manifest, cases, inputs, bench))
            for s in seeds}
    labelled = {s: [(R.features(dev_data[s]["runs"][c["case_id"]], dev_data[s]["ref"], dev_data[s]["cfg"]),
                     dev_key["cases"][c["case_id"]]["category"]) for c in dev_cases] for s in R.DEV_SEEDS}
    rows, train_seeds = {}, {}
    for seed in seeds:
        d = data[seed]
        train_seeds[seed] = [s for s in R.DEV_SEEDS if not (bench == "v1" and s == seed)]
        train = [x for s in train_seeds[seed] for x in labelled[s]]
        rows[seed] = []
        for case in cases:
            cid = case["case_id"]
            f = R.features(d["runs"][cid], d["ref"], d["cfg"])
            ranked_f, fired = R.feature_ranking(f)
            ranked_t, dist = R.template_ranking(f, train)
            ranked_b, scored = R.b0_ranking(R.config_diff(d["ref_flat"], R.apply_overrides(d["ref_flat"], case["overrides"])))
            outputs = {"rule_prior": (R.prior_ranking(), None),
                       "rule_features": (ranked_f, {"fired": fired, "features": _round(f)}),
                       "rule_template": (ranked_t, {"nearest_distance": dist, "train_seeds": train_seeds[seed]}),
                       "rule_b0": (ranked_b, {"diff": scored})}
            for mode, (ranking, evidence) in outputs.items():
                s = score_blind(cid, ranking, key)
                rows[seed].append({"case_id": cid, "seed": seed, "mode": mode, "task": "blind", "bench": bench,
                                   "attempt": 1, "infra_error": False, "model": None, "info": MODES[mode],
                                   "ranking": ranking, "suspected": ranking[0], "truth": s["truth"],
                                   "correct": s["correct"], "top2": s["top2"], "category": s["category"],
                                   "evidence": evidence})
    table = {}
    for mode in MODES:
        t = table.setdefault(mode, {"info": MODES[mode]})
        for seed in seeds:
            cells = [r for r in rows[seed] if r["mode"] == mode]
            t[f"s{seed}"] = {"cells": len(cells), "top1": sum(r["correct"] for r in cells),
                             "top2": sum(r["top2"] for r in cells)}
        t["total"] = {k: sum(t[f"s{s}"][k] for s in seeds) for k in ("cells", "top1", "top2")}
    return rows, table, {"data": data, "train_seeds": train_seeds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0-A2 결정적 기준선을 저장된 텔레메트리에 적용한다(모델·네트워크·GPU 없음)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--tag", help="evals/results/<tag>/에 쓴다(기존 폴더는 덮어쓰지 않음)")
    dest.add_argument("--out", type=Path, help="새 폴더에 쓴다")
    ap.add_argument("--run-params", type=Path, help="실제 실패 실행 params가 있는 Isaac Lab 로그 폴더(선택)")
    ap.add_argument("--bench", choices=sorted(BENCHES), default="v1",
                    help="p0c: 숨은 결함 holdout(케이스 h01~h06, 설정 변경 없음). 정답은 비공개 정답표에서 읽는다")
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
        code = code_version(list(INPUT_DIRS) + (["bench/runs_p0c"] if args.bench == "p0c" else []), CODE_FILES)
        problem = freeze_problem(code, args.seeds)
        if problem:
            raise InputError(problem)
        key, cases, manifest = _json(KEY_PATH), _json(CASES_PATH), _json(MANIFEST_PATH)
        unknown = [c["case_id"] for c in cases if c["case_id"] not in key["cases"]]
        if unknown:
            raise InputError(f"정답표에 없는 case_id {unknown}")
        inputs = [{"path": _show(p), "sha256_lf": _sha256_lf(p)} for p in (KEY_PATH, CASES_PATH, MANIFEST_PATH)]
        if args.bench == "p0c":
            eval_key = _json(KEY_P0C)
            eval_cases = [{"case_id": cid, "overrides": []} for cid in sorted(eval_key["cases"])]
            inputs.append({"path": _show(KEY_P0C), "sha256_lf": _sha256_lf(KEY_P0C)})
            rows, table, ctx = run(args.seeds, eval_key, eval_cases, manifest, inputs, "p0c", dev=(key, cases))
            cases = eval_cases
        else:
            rows, table, ctx = run(args.seeds, key, cases, manifest, inputs)
        params_check = None
        if args.run_params:
            if not args.run_params.is_dir():
                raise InputError(f"{_show(args.run_params)}: 폴더가 없다")
            params_check = check_run_params(args.run_params, args.seeds, ctx["data"], cases)
    except (InputError, ValueError) as e:
        print(f"baselines 입력 오류: {e}", file=sys.stderr)
        return 2
    command = ["python", "evals/baselines.py", "--seeds", *map(str, args.seeds)] + (
        ["--bench", args.bench] if args.bench != "v1" else [])
    command += ["--tag", args.tag] if args.tag is not None else ["--out", _show(out_dir)]
    if args.run_params:
        command += ["--run-params", "<external>"]
    seen, unique_inputs = set(), []
    for i in inputs:
        if i["path"] not in seen:
            seen.add(i["path"])
            unique_inputs.append(i)
    report = {
        "kind": "rule_baselines", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(command), "bench": args.bench, "seeds": args.seeds, "dev_seeds": list(R.DEV_SEEDS),
        "template_train_seeds": {f"s{s}": v for s, v in ctx["train_seeds"].items()},
        "prior_order": list(R.PRIOR_ORDER), "thresholds": R.THRESHOLDS, "table": table,
        "run_params_check": params_check, "inputs": unique_inputs, "code": code,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "model_calls": 0, "network": "not used", "gpu": "not used"},
    }
    out_dir.mkdir(parents=True)
    for seed, seed_rows in rows.items():
        (out_dir / f"seed{seed}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in seed_rows),
                                                  encoding="utf-8")
    (out_dir / "baselines.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for mode, t in table.items():
        per_seed = "  ".join(f"s{s} {t[f's{s}']['top1']}/{t[f's{s}']['cells']}·{t[f's{s}']['top2']}" for s in args.seeds)
        print(f"{mode:14s} info={t['info']:21s} top-1/top-2 {t['total']['top1']}/{t['total']['cells']} "
              f"{t['total']['top2']}/{t['total']['cells']}  ({per_seed})")
    if params_check:
        print(f"run params: checked={params_check['checked']} equal={params_check['equal']} "
              f"missing={len(params_check['missing'])} mismatch={len(params_check['mismatch'])}")
    print(f"saved {_show(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

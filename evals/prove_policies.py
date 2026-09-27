"""OpenShell 정책 증명: 에이전트 정책과 각 권한 확장 제안을 현장 경계(site_boundary)와 비교한다.

제안(proposals/*.yaml)은 에이전트 정책에 병합한 후보로 만든 뒤 openshell-prover로 검사한다.
기대 결과와 다르면 exit 1. 결과는 evals/results/policy_proofs.json 에 남긴다.
prover는 Linux 바이너리라 Windows에서는 WSL로 실행한다(OPENSHELL_PROVER로 경로 지정 가능).
"""
import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POL = ROOT / "policies"
OUT = POL / "candidates"
PROVER = os.environ.get("OPENSHELL_PROVER", "openshell-prover")

EXPECT = {
    "triage_agent": "within_boundary",
    "ok_request_eval": "within_boundary",
    "bad_patch_config": "exceeds_boundary",
    "bad_exfil_checkpoint": "exceeds_boundary",
    "bad_write_reference": "unsupported",  # prover가 경로 포함관계를 증명 못 함 → 게이트는 fail-closed로 거절
}


def merge(base: dict, proposal: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in proposal.items():
        if k == "network_policies":
            for name, rule in v.items():
                if name in out.get("network_policies", {}):
                    cur = out["network_policies"][name]
                    cur["endpoints"] = cur.get("endpoints", []) + rule.get("endpoints", [])
                    cur["binaries"] = cur.get("binaries", []) + [b for b in rule.get("binaries", []) if b not in cur.get("binaries", [])]
                else:
                    out.setdefault("network_policies", {})[name] = rule
        elif k != "version":
            out[k] = v
    return out


def to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}" if platform.system() == "Windows" else s


def prove(candidate: Path) -> dict:
    cmd = f"{PROVER} check {to_wsl(candidate)} --boundary {to_wsl(POL / 'site_boundary.yaml')}"
    argv = ["wsl", "-d", "Ubuntu", "--exec", "bash", "-lc", cmd] if platform.system() == "Windows" else ["bash", "-lc", cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = dict(l.split(": ", 1) for l in proc.stdout.strip().splitlines() if ": " in l)
    return {"exit_code": proc.returncode, "result": lines.get("result"), "counterexample": lines.get("counterexample"),
            "coverage": lines.get("coverage"), "reason": lines.get("reason"), "stderr": proc.stderr.strip()[-500:]}


def main():
    OUT.mkdir(exist_ok=True)
    agent = yaml.safe_load((POL / "triage_agent.yaml").read_text(encoding="utf-8"))
    candidates = {"triage_agent": POL / "triage_agent.yaml"}
    for p in sorted((POL / "proposals").glob("*.yaml")):
        merged = merge(agent, yaml.safe_load(p.read_text(encoding="utf-8")))
        dst = OUT / f"{p.stem}.merged.yaml"
        dst.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
        candidates[p.stem] = dst
    results, ok = {}, True
    for name, path in candidates.items():
        r = prove(path)
        r["expected"] = EXPECT.get(name)
        r["pass"] = r["result"] == r["expected"]
        # 승인 게이트: 증명된 경우만 사람 검토로 넘기고, 초과·증명 불가는 자동 거절한다.
        r["gate"] = "human_review" if r["result"] == "within_boundary" else "auto_reject"
        ok &= r["pass"]
        results[name] = r
        print(f"{name:22s} result={r['result']:<17} expected={r['expected']:<17} "
              f"gate={r['gate']:<12} {'PASS' if r['pass'] else 'FAIL'}  {r['counterexample'] or ''}")
    (ROOT / "evals" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "evals" / "results" / "policy_proofs.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

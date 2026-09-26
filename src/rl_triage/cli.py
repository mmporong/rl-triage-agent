"""셸 기반 에이전트(NemoClaw의 Hermes·OpenClaw 등)용 CLI. NAT 함수와 같은 구현을 부른다.

예: python -m rl_triage.cli list-changes c01
    python -m rl_triage.cli overview c01
    python -m rl_triage.cli series c01 Train/mean_episode_length
    python -m rl_triage.cli analyze c01 --code "print(max(run['Policy/mean_noise_std']))"
    python -m rl_triage.cli ledger c01
    python -m rl_triage.cli prereg c01 --suspected ch3 --ranking ch3 ch1 ch2 --hypothesis ... \
        --variable ... --gate ... --signature ...
"""
import argparse
import json
import sys

from rl_triage import triage_tools as T


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rl-triage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases")
    for name in ("list-changes", "overview", "ledger"):
        sub.add_parser(name).add_argument("case_id")
    p = sub.add_parser("series"); p.add_argument("case_id"); p.add_argument("tag")
    p = sub.add_parser("analyze"); p.add_argument("case_id"); p.add_argument("--code", required=True)
    p = sub.add_parser("prereg")
    p.add_argument("case_id"); p.add_argument("--suspected", required=True)
    p.add_argument("--ranking", nargs="+", required=True); p.add_argument("--hypothesis", required=True)
    p.add_argument("--variable", required=True); p.add_argument("--gate", required=True)
    p.add_argument("--signature", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        out = T.list_cases()
    elif a.cmd == "list-changes":
        out = T.list_changes(a.case_id)
    elif a.cmd == "overview":
        out = T.telemetry_overview(a.case_id)
    elif a.cmd == "ledger":
        out = T.query_ledger(a.case_id)
    elif a.cmd == "series":
        out = T.get_series(a.case_id, a.tag)
    elif a.cmd == "analyze":
        out = T.run_analysis(a.code, a.case_id)
    else:
        out = T.write_preregistration(a.case_id, a.suspected, a.ranking, a.hypothesis, a.variable, a.gate, a.signature)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()

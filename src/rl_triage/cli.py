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
from pathlib import Path

from rl_triage import triage_tools as T
from rl_triage import run_workspace as W


class _JsonUsageError(Exception):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise _JsonUsageError(message)


def main(argv=None):
    args = list(argv) if argv is not None else sys.argv[1:]
    parser_type = _JsonArgumentParser if args and args[0] in {
        "prepare-run", "inspect-run",
    } else argparse.ArgumentParser
    ap = parser_type(prog="rl-triage")
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
    p = sub.add_parser("prepare-run")
    p.add_argument("--telemetry", required=True); p.add_argument("--workspace", required=True)
    p.add_argument("--case", required=True); p.add_argument("--configuration"); p.add_argument("--facts")
    p = sub.add_parser("inspect-run")
    p.add_argument("case_id"); p.add_argument("--workspace", required=True); p.add_argument("--report")
    try:
        a = ap.parse_args(args)
    except _JsonUsageError as exc:
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
        print()
        return 2
    try:
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
        elif a.cmd == "prepare-run":
            out = W.prepare_run(
                Path(a.telemetry), Path(a.workspace), a.case,
                Path(a.configuration) if a.configuration else None,
                Path(a.facts) if a.facts else None,
            )
        elif a.cmd == "inspect-run":
            out = W.inspect_run(
                a.case_id, Path(a.workspace), Path(a.report) if a.report else None,
            )
        else:
            out = T.write_preregistration(a.case_id, a.suspected, a.ranking, a.hypothesis, a.variable, a.gate, a.signature)
    except (OSError, ValueError, PermissionError) as exc:
        if a.cmd not in ("prepare-run", "inspect-run"):
            raise
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
        print()
        return 2
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

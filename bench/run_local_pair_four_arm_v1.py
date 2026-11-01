"""공용 GPU lease 안에서 등록된 로컬 영상 짝의 네 방식 개발 비교를 실행한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import local_pair_four_arm_v1 as M
import local_pair_nim_transport_v1 as NIM
import local_pair_probe_host_v1 as HOST
import run_prospective_reasoning_confirmation_v1 as R

SOURCE_FILES = tuple(sorted({*M.SOURCE_FILES, *HOST.SOURCE_FILES, *NIM.SOURCE_FILES,
    "bench/run_local_pair_four_arm_v1.py", "tests/test_run_local_pair_four_arm_v1.py",
    "evals/run_prospective_reasoning_confirmation_v1.py"}))


def mounted(path: Path) -> str:
    value = PureWindowsPath(str(path.resolve()))
    M.require(len(value.drive) == 2 and value.drive[1] == ":", "WSL mounted drive 경로")
    return "/mnt/" + value.drive[0].lower() + "/" + "/".join(value.parts[1:])


def worker_command(root: Path, call: Path, protocol: Path, anchor: str) -> list[str]:
    args = ["-B", mounted(ROOT / "evals/local_pair_nim_transport_v1.py"), "--worker",
            "--root", mounted(root), "--call-dir", mounted(call),
            "--protocol", mounted(protocol), "--anchor", anchor]
    shell = 'set -euo pipefail; . "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
    return ["wsl.exe", "--distribution", "Ubuntu", "--exec", "bash", "-lc",
            shell + shlex.join(args)]


def worker_cleanup_command(identity: dict) -> list[str]:
    """소유 worker의 Linux pid/start ticks가 모두 같을 때만 종료한다."""
    M.require(type(identity.get("pid")) is int and identity["pid"] > 1
              and str(identity.get("proc_start_ticks", "")).isdigit(), "worker 종료 신원 부족")
    program = '''import json,os,pathlib,signal,sys,time
i=json.loads(sys.argv[1]);p=int(i["pid"]);expected=str(i["proc_start_ticks"])
def same():
 try:
  text=pathlib.Path(f"/proc/{p}/stat").read_text();parts=text[text.rfind(")")+2:].split()
  return parts[19]==expected and parts[0]!="Z"
 except FileNotFoundError:return False
initial=same()
for sig,limit in [(signal.SIGTERM,5),(signal.SIGKILL,3)]:
 if not same():break
 os.kill(p,sig);end=time.monotonic()+limit
 while same() and time.monotonic()<end:time.sleep(.1)
alive=same();print(json.dumps({"initial_identity_alive":initial,"alive_after":alive,"pid":p,"start_ticks":expected}));sys.exit(1 if alive else 0)
'''
    shell = 'exec "$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.quote(program) + " " + shlex.quote(json.dumps(identity))
    return ["wsl.exe", "--distribution", "Ubuntu", "--exec", "bash", "-lc", shell]


def preflight(protocol: Path, anchor: str, config_path: Path) -> tuple[dict, dict, dict]:
    spec = M.verify_protocol(protocol, anchor)
    NIM._verify_protocol(protocol, anchor)
    M.require(set(SOURCE_FILES) <= set(spec["sources_lf"]), "runner source closure 누락")
    M.require(M.sha(config_path) == spec["execution_config_sha256"], "실행 config SHA")
    config = M.read(config_path)
    M.require(config.get("schema") == "local_pair_four_arm_execution_config_v1", "실행 config schema")
    batch, corpus = map(M._bound_file, (config["source_batch"], config["corpus"]))
    M.require(M.sha(batch) == spec["source_batch_sha256"]
              and M.sha(corpus) == spec["corpus_sha256"], "source/corpus SHA")
    design = M.read(M._bound_file(config["source_design"]))
    rows = R._source_rows(batch, design)
    _, corpus_rows = R._corpus(corpus)
    R._bind_rows(rows, corpus_rows)
    by_case = {row["case_id"]: row for row in rows}
    references = {row["seed"]: row for row in rows if row["category"] == "NONE"}
    lab = Path(config["lab"]).resolve(strict=True)
    tools = spec["tool_protocols"]
    for item in tools.values():
        M.require(M.sha(ROOT / item["path"], lf=True) == item["sha256_lf"], "tool protocol SHA")
    HOST.OBS.verify_protocol(ROOT / tools["paired_probe"]["path"], lab, anchor)
    sources = {}
    M.require(len(config["pairs"]) == len(spec["pairs"]) == 2, "두 개발 pair")
    for item, registered in zip(config["pairs"], spec["pairs"], strict=True):
        M.require(item["id"] == registered["id"], "pair 순서")
        prepared_path = M._bound_file(item["prepared"])
        source = M.load_pair(prepared_path)
        M.require(source["prepared"]["sha256"] == registered["prepared_result_sha256"]
                  and source["pair"]["sha256"] == registered["pair_sha256"], "등록 pair bytes")
        provider_protocol = M.read(M._bound_file(M.read(prepared_path)["protocol"]))
        pending = M.P.INPUT.inspect(M._bound_file(item["input_spec"]), lab,
            execution_anchor=provider_protocol["input_protocol"]["execution_anchor"])
        M.require(pending == source["pending"], "현재 raw 입력 재검산 불일치")
        HOST._packets(source["packet"], source["backend_packet"])
        row = by_case[item["case_id"]]
        reference = references[row["seed"]]
        M.require(row["category"] == "F1" and row["seed"] == registered["seed"], "개발 source 범위")
        inputs = {}
        for role, original in (("case", row), ("reference", reference)):
            base = M._bound_file(original["metadata"]["base_collected"])
            _, _, inspected = HOST.OBS._source(base, lab, role)
            M.require(inspected["checkpoint"]["sha256"]
                      == source["packet"]["input_bindings"][role]["checkpoint_sha256"], "backend CP")
            inputs[role] = base
        sources[item["id"]] = {"source": source, "inputs": inputs, "case_id": item["case_id"]}
    for case_id in spec["normal_controls"]:
        row, entry = by_case[case_id], corpus_rows[case_id]
        M.require(M.sha(R.BASE.RUN.BANDS) == R.BASE.RUN.BANDS_SHA256, "fixed26 bands SHA")
        bands = M.read(R.BASE.RUN.BANDS)["bands"]
        gate = R.BASE.RUN._gate(row, references[row["seed"]], bands)
        M.require(row["category"] == "NONE" and entry["evaluation_role"] == "normal_model_only"
                  and gate["run"] is False and gate["reason"] == "healthy_gate", "정상 host gate")
    return spec, config, sources


def _worker_send(root: Path, protocol: Path, anchor: str, _body: dict, call: Path) -> dict:
    argv = worker_command(root, call, protocol, anchor)
    M.write(call / "parent-launch.json", {"argv": argv, "timeout_s": 150,
                                         "ownership": "one_shot_existing_ubuntu_worker"})
    with subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        M.write(call / "parent-process.json", {"pid": child.pid, "ownership": "Popen_handle",
                                               "started_monotonic_ns": time.monotonic_ns()})
        try:
            stdout, stderr = child.communicate(timeout=150)
        except subprocess.TimeoutExpired as error:
            child.kill()
            stdout, stderr = child.communicate(timeout=10)
            identity = M.read(call / "worker-process.json") if (call / "worker-process.json").is_file() else None
            cleanup = None
            if identity is not None:
                stopped = subprocess.run(worker_cleanup_command(identity), capture_output=True,
                                         timeout=20, check=False)
                (call / "timeout-cleanup-stdout.txt").write_bytes(stopped.stdout)
                (call / "timeout-cleanup-stderr.txt").write_bytes(stopped.stderr)
                cleanup = {"exit_code": stopped.returncode}
            M.write(call / "parent-timeout.json", {"error": type(error).__name__,
                "known_worker_identity": identity, "cleanup": cleanup,
                "cleanup_confirmed": cleanup is not None and cleanup["exit_code"] == 0,
                "further_dispatch_allowed": False})
            (call / "parent-stdout.txt").write_bytes(stdout)
            (call / "parent-stderr.txt").write_bytes(stderr)
            raise
        (call / "parent-stdout.txt").write_bytes(stdout)
        (call / "parent-stderr.txt").write_bytes(stderr)
        M.write(call / "parent-exit.json", {"exit_code": child.returncode})
    candidate = call / ("worker-response.json" if (call / "worker-response.json").is_file()
                        else "worker-error.json")
    M.require(candidate.is_file() and child.returncode in {0, 1}, "NIM worker 완료 원문 부재")
    identity, ended = M.read(call / "worker-process.json"), M.read(call / "worker-exit.json")
    M.require(ended.get("identity") == identity and ended.get("exit_code") == child.returncode,
              "WSL worker 종료 신원/코드 결합")
    return M.read(candidate)


def execute(protocol: Path, anchor: str, config_path: Path, output: Path) -> dict:
    spec, config, sources = preflight(protocol, anchor, config_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    lease = R.SOURCE.LEASE.lease_context(output / "lease-initial")
    M.require(lease["owner"] == R.SOURCE.OWNER, "GPU lease owner")
    NIM.initialize(output / "nim", protocol_path=protocol, execution_anchor=anchor, api_cap=12)
    paired = 0
    results = []

    def context(_request: dict | None = None) -> dict:
        M.require(time.monotonic() - started < spec["wall_cap_s"], "batch wall 상한")
        M.require(M.sha(config_path) == spec["execution_config_sha256"], "실행 config 변경")
        now = R.SOURCE.LEASE.lease_context()
        M.require(now["lease_id"] == lease["lease_id"] and now["owner"] == lease["owner"], "GPU lease 변경")
        for name, expected in spec["sources_lf"].items():
            M.require(M.sha(ROOT / name, lf=True) == expected, "실행 source 변경: " + name)
        return {"approved": True}

    for item in spec["pairs"]:
        data = sources[item["id"]]
        source = data["source"]
        for arm in spec["run_orders"][item["id"]]:
            context()
            def dispatch(snapshot: dict, choice: dict, *, out: Path, _source=source, _data=data) -> dict:
                nonlocal paired
                context()
                M.require(paired < spec["paired_cap"], "paired 전역 상한")
                paired += 1  # intent를 센다. 실패해도 예약을 되돌리지 않는다.
                return HOST.execute_pair(snapshot, choice, backend_packet=_source["backend_packet"],
                    case_input=_data["inputs"]["case"], reference_input=_data["inputs"]["reference"],
                    probe_protocol=ROOT / spec["tool_protocols"]["paired_probe"]["path"],
                    view_protocol=ROOT / spec["tool_protocols"]["view"]["path"], anchor=anchor,
                    lab=Path(config["lab"]), out=out, verify_context=context)

            def complete(request: dict) -> dict:
                context()
                return NIM.complete(output / "nim", request,
                    send=lambda body, call: _worker_send(output / "nim", protocol, anchor, body, call))

            result = M.run_arm(source, arm=arm, run_id=item["id"] + "_" + arm,
                output=output / item["id"] / arm, protocol_path=protocol, execution_anchor=anchor,
                complete=complete if arm in M.MODEL_ARMS else None,
                execute_probe=dispatch, verify_probe=HOST.verify_pair, verify_call=NIM.verify_call)
            results.append({"id": item["id"], "arm": arm, "end_state": result["end_state"],
                "confirmed_subtype": result["confirmed_subtype"],
                "correct_confirmation": result["confirmed_subtype"] == "reward",
                "hold_reason": result["hold_reason"], "paired_probes": len(result["steps"]),
                "model_calls": result["model_calls"], "result": M.ref(output / item["id"] / arm / "result.json")})
    normal_rows = [{"id": case, "arm": arm, "end_state": "HOLD", "reason": "healthy_host_gate",
                    "model_calls": 0, "paired_probes": 0, "model_normal_recognition": False}
                   for case in spec["normal_controls"] for arm in M.ARMS]
    result = {"schema": "local_pair_four_arm_batch_v1", "status": "complete",
        "rows": results, "normal_host_controls": normal_rows, "paired_intents": paired,
        "model_calls": sum(row["model_calls"] for row in results),
        "parent_wall_s": time.monotonic() - started, "lease_id": lease["lease_id"],
        "scope": "development_seen_sources", "independent_source_seeds": 2,
        "heldout": False, "repair": False, "recovery": False,
        "llm_superiority": False, "video_advantage_confirmed": False}
    M.write(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            spec, _, sources = preflight(args.protocol, args.anchor, args.config)
            M.write(args.out, {"status": "complete", "pairs": sorted(sources), "api": 0, "sdk": 0,
                               "protocol_sha256_lf": M.sha(args.protocol, lf=True), "api_cap": spec["api_cap"]})
        else:
            execute(args.protocol, args.anchor, args.config, args.out)
        return 0
    except Exception as error:
        target = args.out.with_name(args.out.name + "-failure.json") if args.preflight else args.out / "failure.json"
        M.write(target, {"status": "invalid", "error_type": type(error).__name__, "message": str(error),
                         "automatic_retry": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

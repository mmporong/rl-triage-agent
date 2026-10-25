"""V05 E2E actual caller의 검증·한 단계 실행 CLI."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import grounded_vision_observer_v1 as OBSERVER
import vision_selection_brev_adapter_v1 as ADAPTER
import vision_selection_e2e_caller_v1 as CALLER


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}은 JSON object여야 한다")
    return value


def _repo_path(repo: Path, raw: str) -> Path:
    path = Path(raw)
    target = (path if path.is_absolute() else repo / path).resolve(strict=True)
    target.relative_to(repo)
    return target


def load_caller(args: argparse.Namespace) -> CALLER.E2ECaller:
    repo = Path(args.repo).resolve(strict=True)
    protocol = _repo_path(repo, args.execution_protocol)
    lease = _repo_path(repo, args.lease)
    plan = _repo_path(repo, args.plan)
    context = ADAPTER.verify_context(
        repo,
        {"path": protocol.relative_to(repo).as_posix(), "sha256": args.execution_protocol_sha256},
        {"path": lease.relative_to(repo).as_posix(), "sha256": args.lease_sha256},
        require_active=False,
    )
    return CALLER.E2ECaller(repo, plan, context, Path(args.runtime))


def _backend(caller: CALLER.E2ECaller, payload: Mapping[str, Any], *,
             binding_name: str = "correction_backend") -> Any:
    bindings = caller.context["protocol"]["execution_bindings"]
    if binding_name not in {"correction_backend", "record_verifier"}:
        raise ValueError("지원하지 않는 production backend binding이다")
    binding = bindings[binding_name]
    registered = ADAPTER._file_ref(
        caller.root,
        {"path": binding["path"], "sha256": binding["sha256_lf"]},
        binding_name, lf=True,
    )
    module_name = f"_registered_vision_backend_{ADAPTER.file_sha(registered, lf=True)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, registered)
    if spec is None or spec.loader is None:
        raise ValueError("등록 production backend module을 load할 수 없다")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    backend_class = getattr(module, binding["class"], None)
    if not isinstance(backend_class, type):
        raise TypeError("등록 production backend class가 module에 없다")
    source = inspect.getsourcefile(backend_class)
    if source is None or Path(source).resolve() != registered:
        raise ValueError("CLI production backend class가 등록 source와 다르다")
    source_path = ADAPTER._file_ref(
        caller.root,
        {"path": bindings["source_protocol"]["path"],
         "sha256": bindings["source_protocol"]["sha256_lf"]},
        "source protocol", lf=True,
    )
    source_protocol = _read(source_path)
    lab = _repo_path(caller.root, str(payload.get("lab")))
    python = Path(str(payload.get("python"))).resolve(strict=True)
    if python.is_dir() or python.is_symlink():
        raise ValueError("Isaac Python은 regular file이어야 한다")
    backend_protocol = {**source_protocol, "protocol_path": str(source_path),
                        "protocol_sha256_lf": ADAPTER.file_sha(source_path, lf=True)}
    return backend_class.from_runtime(
        lab, python, backend_protocol, caller.plan["source_anchor"],
        caller.root / ".triage-runtime/vision-brev" / source_protocol["protocol_id"],
        source_protocol["training_plan"], resource_proof=caller.context["resource"],
    )


def _services(caller: CALLER.E2ECaller, command: Mapping[str, Any],
              payload: Mapping[str, Any]) -> dict[str, Any]:
    kind, group = command["kind"], command["group"]
    if kind == "initial_selection":
        return {
            "complete": OBSERVER.default_complete,
            "live_snapshot": payload["live_snapshot"],
            "live_contract": _repo_path(caller.root, str(payload["live_contract"])),
            "source_binding": payload["source_binding"],
        }
    if kind == "probe_pair":
        return {"commands": payload["commands"], "expected_source": payload["expected_source"],
                "approver": payload.get("approver")}
    if kind == "propose_correction":
        return {}
    if kind == "consume_and_run_correction":
        backend = _backend(caller, payload)
        source_protocol = _read(ADAPTER._file_ref(
            caller.root,
            {"path": caller.context["protocol"]["execution_bindings"]["source_protocol"]["path"],
             "sha256": caller.context["protocol"]["execution_bindings"]["source_protocol"]["sha256_lf"]},
            "source protocol", lf=True,
        ))
        seed = caller._row(group)["source_seed"]
        none = [row for row in source_protocol["training_plan"]
                if row.get("seed") == seed and row.get("arm") == "NONE"]
        if len(none) != 1:
            raise ValueError("CORRECTED 평가용 NONE job이 유일하지 않다")
        service = ADAPTER.CorrectionService(
            backend, backend.runtime_root, str(none[0]["job_id"]), f"e2e-{group}-corrected",
        )
        return {"backend_reference": caller.context["protocol"]["execution_bindings"]["correction_backend"],
                "train": service.train, "evaluate": service.evaluate}
    if kind == "post_observation":
        backend = _backend(caller, payload, binding_name="record_verifier")
        correction = caller._cached_correction(group)
        if correction is None:
            raise ValueError("후속 관찰 전에 검증된 correction이 없다")
        record_job = caller._record_job(group, correction)

        def observation_input(record_result: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
            contract = ADAPTER.materialize_record_contract(
                caller.context, record_result, record_job=record_job,
                output_dir=caller.runtime / "post-observation" / group / "evidence",
            )
            binding = OBSERVER.binding_template(
                contract,
                run_id=f"{caller.plan['execution_id']}-post-{group}",
                source_anchor=caller.plan["source_anchor"],
                protocol_sha256=caller.context["protocol_sha256"],
            )
            return contract, binding

        return {
            "record": backend.record,
            "record_plan_reference": caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
            "observation_input": observation_input,
            "complete": OBSERVER.default_complete,
            "verifier_reference": caller.context["protocol"]["execution_bindings"]["record_verifier"],
        }
    raise ValueError(f"지원하지 않는 명령 {kind!r}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("verify", "remaining_commands", "advance", "summarize"))
    result.add_argument("--repo", required=True)
    result.add_argument("--plan", required=True)
    result.add_argument("--execution-protocol", required=True)
    result.add_argument("--execution-protocol-sha256", required=True)
    result.add_argument("--lease", required=True)
    result.add_argument("--lease-sha256", required=True)
    result.add_argument("--runtime", required=True)
    result.add_argument("--input")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    caller = load_caller(args)
    if args.command == "verify":
        value: Any = {"status": "verified", "execution_id": caller.plan["execution_id"],
                      "api_cap": caller.plan["api_cap"]}
    elif args.command == "remaining_commands":
        value = caller.remaining_commands()
    elif args.command == "summarize":
        value = caller.summarize()
    else:
        payload = _read(Path(args.input)) if args.input else {}
        commands = caller.remaining_commands()
        if not commands:
            request_id = payload.get("approve_request_id")
            approver = payload.get("approver")
            if (not isinstance(request_id, str) or not isinstance(approver, str)
                    or not approver.strip()):
                value = {"status": "HOLD", "reason": "no_runnable_command"}
            else:
                value = caller.approve_correction(request_id, approver)
        else:
            command = commands[0]
            value = caller.advance(command, _services(caller, command, payload))
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

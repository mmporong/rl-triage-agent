# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""검증된 source-v2와 same-Play 결과를 five-policy 등록 후보로 봉인한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_lineage_five_policy_v1 as EXPORT
import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_summary_v1 as SUMMARY
import lineage_input_runtime_matrix_v1 as MATRIX
import lineage_probe_backend_v1 as P1
import lineage_same_play_pair_inputs_v1 as PAIR
import lineage_source_inputs_v2 as INPUT
import lineage_source_probe_inputs_v1 as HOST
import run_lineage_five_policy_v1 as RUNNER
import runtime_observation_view_v1 as VIEW

B = DRIVER.BASE
SCHEMA = "lineage_five_policy_registration_preparation_v1"
CONFIG_SCHEMA = "lineage_five_policy_execution_config_v1"
STATUS = "prepared_not_registered_not_executable"
PROTOCOL_DIR = ROOT / "bench/protocols"
VIEW_PATH = ROOT / VIEW.PROTOCOL_RELATIVE
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
TAG = re.compile(r"[a-z0-9_]{1,64}\Z")
SOURCE_FILES = tuple(
    sorted(
        {
            *EXPORT.SOURCE_FILES,
            *MATRIX.SOURCE_FILES,
            *RUNNER.SOURCE_FILES,
            *SUMMARY.source_files(),
            "bench/prepare_lineage_five_policy_v1.py",
            "tests/test_prepare_lineage_five_policy_v1.py",
        }
    )
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=DRIVER.A.V._unique_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")
        ),
    )
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return cast(dict[str, Any], value)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(_canonical(value))


def _private_ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    _require(
        target.is_file() and not target.is_symlink(),
        "private ref는 일반 파일이어야 한다",
    )
    return {"path": str(target), "sha256": _sha(target)}


def _public_ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    try:
        relative = target.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("공개 protocol은 저장소 내부여야 한다") from error
    return {"path": relative, "sha256_lf": _sha(target, lf=True)}


def _bound_ref(value: object, label: str) -> Path:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256"}
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str),
        label + " ref 형식",
    )
    item = cast(Mapping[str, str], value)
    path = Path(item["path"]).resolve(strict=True)
    _require(
        path.is_file() and not path.is_symlink() and _sha(path) == item["sha256"],
        label + " SHA",
    )
    return path


def _relative_protocol_ref(path: Path, value: object, label: str) -> Path:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256_lf"}
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256_lf"), str),
        label + " protocol ref",
    )
    item = cast(Mapping[str, str], value)
    relative = Path(item["path"])
    _require(
        not relative.is_absolute() and ".." not in relative.parts, label + " 상대경로"
    )
    child = (ROOT / relative).resolve(strict=True)
    _require(
        child.is_file() and _sha(child, lf=True) == item["sha256_lf"],
        label + " protocol SHA",
    )
    _require(child != Path(path).resolve(), label + " protocol cycle")
    return child


def _resolve_component_template(path: Path) -> tuple[Path, dict[str, Any]]:
    """기존 failure/admission wrapper를 따라 동결 P1 component까지 해석한다."""
    current = Path(path).resolve(strict=True)
    seen: set[Path] = set()
    while True:
        _require(current not in seen, "component template cycle")
        seen.add(current)
        value = _read(current)
        schema = value.get("schema")
        if schema == P1.PROTOCOL_SCHEMA:
            _require(set(value) == P1.PROTOCOL_FIELDS, "P1 component template 필드")
            return current, value
        if schema == "lineage_probe_failure_evidence_protocol_v3":
            current = _relative_protocol_ref(
                current, value.get("component_admission_protocol"), "failure evidence"
            )
            continue
        if schema == "lineage_probe_admission_wrapper_protocol_v2":
            current = _relative_protocol_ref(
                current, value.get("component_probe_protocol"), "probe admission"
            )
            continue
        raise ValueError(
            "component template은 P1 또는 등록된 failure/admission wrapper여야 한다"
        )


def _verify_anchor_sources(anchor: str, names: Sequence[str]) -> None:
    _require(HEX40.fullmatch(anchor) is not None, "source anchor 형식")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", anchor, "HEAD"],
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, "source anchor가 현재 HEAD 조상이 아니다")
    for name in names:
        relative = Path(name)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "component source 상대경로",
        )
        path = ROOT / relative
        _require(path.is_file(), "component source가 없다: " + name)
        committed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{anchor}:{name}"],
            capture_output=True,
            check=False,
        )
        _require(committed.returncode == 0, "source anchor blob이 없다: " + name)
        normalized = committed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(
            hashlib.sha256(normalized).hexdigest() == _sha(path, lf=True),
            "source anchor blob 변경: " + name,
        )


def _component_value(template: Mapping[str, Any], lab: Path) -> dict[str, Any]:
    source = template.get("source_sha256_lf")
    runtime = template.get("runtime_source_sha256_lf")
    watchdogs = template.get("watchdogs_s")
    source_map = P1._hash_map(source, "component source closure")
    runtime_map = P1._hash_map(runtime, "component runtime closure")
    _require(
        set(runtime_map) == HOST.BASE_PROBE.RUNTIME_PINS,
        "component runtime pin은 정확히 9개여야 한다",
    )
    _require(
        isinstance(watchdogs, Mapping)
        and set(watchdogs) == {"legacy", "optimizer"}
        and all(
            type(watchdogs[name]) is int and watchdogs[name] > 0 for name in watchdogs
        ),
        "component watchdogs",
    )
    watchdog_map = cast(Mapping[str, Any], watchdogs)
    _require(
        template.get("isaac_python_relative") == "_isaac_sim/python.bat"
        and isinstance(template.get("scope"), str)
        and bool(template["scope"]),
        "component launcher 또는 scope",
    )
    lab = Path(lab).resolve(strict=True)
    launcher = lab / "_isaac_sim/python.bat"
    _require(launcher.is_file() and not launcher.is_symlink(), "Isaac launcher가 없다")
    current_runtime = {}
    for name in sorted(HOST.BASE_PROBE.RUNTIME_PINS):
        path = lab / name
        _require(
            path.is_file() and not path.is_symlink(), "runtime pin이 없다: " + name
        )
        current_runtime[name] = _sha(path, lf=True)
    return {
        "schema": P1.PROTOCOL_SCHEMA,
        "source_sha256_lf": {
            str(name): _sha(ROOT / str(name), lf=True) for name in sorted(source_map)
        },
        "runtime_source_sha256_lf": current_runtime,
        "watchdogs_s": deepcopy(dict(watchdog_map)),
        "isaac_python_relative": "_isaac_sim/python.bat",
        "scope": template.get("scope"),
    }


@dataclass(frozen=True)
class FixtureHooks:
    input_checked: Callable[[Path], tuple[dict[str, Any], dict[str, Any]]]
    pair_load: Callable[[Path, Path], dict[str, Any]]
    bridge_collect: Callable[..., dict[str, Any]]
    bridge_checked: Callable[[Path], tuple[dict[str, Any], dict[str, Any]]]
    source_checked: Callable[[dict[str, Any]], dict[str, Any]]
    protocol_value: Callable[..., dict[str, Any]]
    bridge_protocol: Callable[[Path], dict[str, Any]]
    view_value: Callable[[], dict[str, Any]]
    view_verify: Callable[[Path, str], dict[str, Any]]
    anchor_sources: Callable[[str, Sequence[str]], None]


def _production_hooks() -> FixtureHooks:
    return FixtureHooks(
        input_checked=INPUT.checked,
        pair_load=PAIR.load_pair,
        bridge_collect=HOST.collect,
        bridge_checked=HOST.checked,
        source_checked=DRIVER.checked_source,
        protocol_value=DRIVER.protocol_value,
        bridge_protocol=HOST.prepare_protocol,
        view_value=VIEW.prepare_protocol,
        view_verify=lambda path, anchor: VIEW.verify_protocol(path, anchor),
        anchor_sources=_verify_anchor_sources,
    )


def _case_plan(
    public: Mapping[str, Any], metadata: Mapping[str, Any], study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sources = metadata.get("sources")
    cases = metadata.get("cases")
    public_sources = public.get("sources")
    public_cases = public.get("case_ids")
    study_cases = study.get("case_plan")
    _require(
        isinstance(sources, Mapping)
        and len(sources) == 20
        and isinstance(public_sources, Mapping)
        and set(public_sources) == {f"source-{index:02d}" for index in range(1, 21)},
        "source-v2 index는 익명 source 20개여야 한다",
    )
    _require(
        isinstance(cases, list)
        and len(cases) == 10
        and isinstance(public_cases, list)
        and isinstance(study_cases, list)
        and [row.get("case_id") for row in cases] == public_cases
        and [row.get("case_id") for row in study_cases] == public_cases,
        "index/study case 10건 순서가 다르다",
    )
    source_map = cast(Mapping[str, Any], sources)
    case_rows = cast(list[dict[str, Any]], cases)
    _require(
        all(
            isinstance(row, dict)
            and row.get("source_job") in source_map
            and row.get("reference_job") in source_map
            for row in case_rows
        ),
        "case role source binding",
    )
    return deepcopy(case_rows)


def _checked_pair_batch(
    path: Path,
    index: Path,
    case_order: Sequence[str],
    pair_load: Callable[[Path, Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Path]]:
    batch = _read(path)
    pairs = batch.get("pairs")
    _require(
        set(batch)
        == {
            "schema",
            "status",
            "sdk_intents",
            "api",
            "source_artifacts",
            "pairs",
            "parent_wall_s",
            "diagnosis",
            "recovery",
        }
        and batch.get("schema") == "lineage_same_play_input_batch_result_v1"
        and batch.get("status") == "complete"
        and batch.get("sdk_intents") == 19
        and batch.get("api") == 0
        and batch.get("source_artifacts") == 20
        and batch.get("diagnosis") is False
        and batch.get("recovery") is False
        and isinstance(pairs, Mapping)
        and list(pairs) == list(case_order),
        "same-Play actual batch complete20/pairs10/API0/SDK19 계약",
    )
    sources: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for case_id in case_order:
        result_path = _bound_ref(
            cast(Mapping[str, Any], pairs)[case_id], case_id + " pair"
        )
        source = pair_load(result_path, index)
        _require(source.get("case_id") == case_id, "pair source case identity")
        sources[case_id] = source
        paths[case_id] = result_path
    return batch, sources, paths


def _public_has_private_path(value: object) -> bool:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    return any(
        token in raw for token in ("bench/private", "\\users\\", "/users/", "/home/")
    )


def prepare(
    index: Path,
    pair_batch_result: Path,
    study: Path,
    component_template: Path,
    lab: Path,
    *,
    source_anchor: str,
    tag: str,
    private_root: Path,
    fixture: FixtureHooks | None = None,
    protocol_dir: Path = PROTOCOL_DIR,
) -> dict[str, Any]:
    """실행 권한 없이 private 입력과 새 공개 등록 후보를 write-once 생성한다."""
    _require(TAG.fullmatch(tag) is not None, "registration tag")
    hooks = fixture or _production_hooks()
    index = Path(index).resolve(strict=True)
    pair_batch_result = Path(pair_batch_result).resolve(strict=True)
    study = Path(study).resolve(strict=True)
    lab = Path(lab).resolve(strict=True)
    private_root = Path(private_root).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    _require(
        fixture is not None or protocol_dir == PROTOCOL_DIR.resolve(),
        "production protocol 경로는 bench/protocols여야 한다",
    )
    targets = {
        "component": protocol_dir / f"{tag}_probe_component.json",
        "paired_probe": protocol_dir / f"{tag}_probe_bridge.json",
        "protocol": protocol_dir / f"{tag}_five_policy.json",
    }
    _require(
        not private_root.exists(), "private registration root는 fresh 경로여야 한다"
    )
    _require(
        all(not path.exists() for path in targets.values()),
        "public protocol target은 fresh 파일이어야 한다",
    )

    public, metadata = hooks.input_checked(index)
    design = _read(study)
    _require(
        design.get("schema") == "agentic_five_type_evaluation_design_v1",
        "five-policy study schema",
    )
    cases = _case_plan(public, metadata, design)
    case_order = [cast(str, row["case_id"]) for row in cases]
    _batch, sources, pair_paths = _checked_pair_batch(
        pair_batch_result, index, case_order, hooks.pair_load
    )
    template_path, template = _resolve_component_template(component_template)
    component = _component_value(template, lab)
    hooks.anchor_sources(source_anchor, tuple(component["source_sha256_lf"]))
    _require(
        hooks.view_value() == _read(VIEW_PATH),
        "고정 runtime view가 현재 prepare 값과 다르다",
    )
    hooks.view_verify(VIEW_PATH, source_anchor)

    private_root.mkdir(parents=True, exist_ok=False)
    source_metadata = cast(Mapping[str, Mapping[str, Any]], metadata["sources"])
    bridges: dict[str, Path] = {}
    bridge_values: dict[str, dict[str, Any]] = {}
    for source_id in sorted(cast(Mapping[str, Any], public["sources"])):
        path = private_root / "bridges" / f"{source_id}.json"
        hooks.bridge_collect(index, path, source_id=source_id)
        collected, _validated = hooks.bridge_checked(path)
        _require(collected.get("source_id") == source_id, "bridge source identity")
        bridges[source_id] = path
        bridge_values[source_id] = collected

    source_paths: dict[str, Path] = {}
    config_cases: dict[str, Any] = {}
    for row in cases:
        case_id = cast(str, row["case_id"])
        source = hooks.source_checked(sources[case_id])
        source_path = private_root / "sources" / f"{case_id}.json"
        _write_new(source_path, source)
        source_paths[case_id] = source_path
        role_paths: dict[str, Path] = {}
        for role, job_key in (("case", "source_job"), ("reference", "reference_job")):
            job = cast(str, row[job_key])
            source_id = cast(str, source_metadata[job]["public_id"])
            bridge = bridge_values[source_id]
            binding = source["packet"]["input_bindings"][role]
            _require(
                bridge["binding"]["checkpoint_sha256"] == binding["checkpoint_sha256"]
                and bridge["binding"]["telemetry_sha256"]
                == binding["telemetry_sha256"],
                "source/bridge role SHA binding",
            )
            role_paths[role] = bridges[source_id]
        config_cases[case_id] = {
            "source": _private_ref(source_path),
            "probe_inputs": {
                role: _private_ref(path) for role, path in role_paths.items()
            },
            "pair_input": _private_ref(pair_paths[case_id]),
        }

    config = {
        "schema": CONFIG_SCHEMA,
        "lab": str(lab),
        "source_index": _private_ref(index),
        "cases": config_cases,
    }
    config_path = private_root / "execution-config.json"
    _write_new(config_path, config)

    _write_new(targets["component"], component)
    bridge_protocol = hooks.bridge_protocol(targets["component"])
    _write_new(targets["paired_probe"], bridge_protocol)
    protocol_ref = (
        _public_ref
        if fixture is None
        else lambda path: {"path": Path(path).name, "sha256_lf": _sha(path, lf=True)}
    )
    extra_sources = tuple(sorted(set(SOURCE_FILES)))
    r3 = hooks.protocol_value(
        study, [sources[case] for case in case_order], extra_sources=extra_sources
    )
    r3.update(
        {
            "source_anchor": source_anchor,
            "execution_config_sha256": _sha(config_path),
            "source_index_sha256": _sha(index),
            "same_play_batch_result_sha256": _sha(pair_batch_result),
            "tool_protocols": {
                "paired_probe": protocol_ref(targets["paired_probe"]),
                "view": _public_ref(VIEW_PATH),
            },
            "paired_cap": 150,
            "sdk_role_cap": 300,
            "wall_cap_s": 144000,
        }
    )
    _require(
        set(SOURCE_FILES) <= set(r3.get("sources_lf", {})),
        "R3 analysis/runner/preparer source closure",
    )
    _require(
        not _public_has_private_path(component),
        "component protocol에 private 경로가 있다",
    )
    _require(
        not _public_has_private_path(bridge_protocol),
        "bridge protocol에 private 경로가 있다",
    )
    _require(not _public_has_private_path(r3), "R3 protocol에 private 경로가 있다")
    _write_new(targets["protocol"], r3)

    registration = {
        "schema": SCHEMA,
        "status": STATUS,
        "fixture": fixture is not None,
        "source_anchor": source_anchor,
        "execution_anchor": None,
        "executable": False,
        "source_index": _private_ref(index),
        "same_play_batch_result": _private_ref(pair_batch_result),
        "component_template": _private_ref(template_path),
        "execution_config": _private_ref(config_path),
        "public_protocols": {
            name: protocol_ref(path) for name, path in targets.items()
        },
        "counts": {"sources": 20, "cases": 10, "probe_inputs": 20, "pair_inputs": 10},
        "caps": {"paired": 150, "sdk_roles": 300, "wall_s": 144000, "api": 72},
        "claims": {
            "registered": False,
            "executable": False,
            "gpu": False,
            "api": False,
        },
        "next": "commit public protocols and source closure, then run preflight with that execution anchor",
    }
    registration_path = private_root / "registration.json"
    _write_new(registration_path, registration)
    return deepcopy(registration)


def preflight(
    protocol: Path,
    execution_anchor: str,
    config: Path,
    receipt: Path,
    *,
    runner_preflight: Callable[
        [Path, str, Path], tuple[dict, dict, dict]
    ] = RUNNER.preflight,
) -> dict[str, Any]:
    """공개 등록 commit 뒤 실제 five-policy runner preflight를 호출한다."""
    value, private, sources = runner_preflight(protocol, execution_anchor, config)
    _require(
        len(sources) == 10 and set(sources) == set(value["case_order"]),
        "runner preflight case10",
    )
    _require(
        all(
            row.get("source") is not None and "input_hold" not in row
            for row in sources.values()
        ),
        "등록 입력이 모두 available이어야 한다",
    )
    result = {
        "schema": "lineage_five_policy_registration_preflight_v1",
        "status": "registered_preflight_complete_not_executed",
        "execution_anchor": execution_anchor,
        "protocol": _private_ref(Path(protocol)),
        "execution_config": _private_ref(Path(config)),
        "case_count": len(sources),
        "source_index": deepcopy(private["source_index"]),
        "gpu": 0,
        "api": 0,
        "execution_started": False,
    }
    _write_new(receipt, result)
    return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--pair-batch-result", type=Path, required=True)
    build.add_argument("--study", type=Path, required=True)
    build.add_argument("--component-template", type=Path, required=True)
    build.add_argument("--lab", type=Path, required=True)
    build.add_argument("--source-anchor", required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--private-root", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--protocol", type=Path, required=True)
    check.add_argument("--execution-anchor", required=True)
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.index,
            args.pair_batch_result,
            args.study,
            args.component_template,
            args.lab,
            source_anchor=args.source_anchor,
            tag=args.tag,
            private_root=args.private_root,
        )
    else:
        preflight(
            args.protocol,
            args.execution_anchor,
            args.config,
            args.receipt,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

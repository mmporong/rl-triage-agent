"""등록된 post-actuator NONE context만 기존 actuator child로 전달한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePath
from typing import Any


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "bench/actuator_envelope_runtime.py").is_file():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _find_repo()
RUN_ROOT = ROOT / "bench/private/post_actuator_confirmation/runs"
CONSUMED_ROOT = ROOT / "bench/private/post_actuator_confirmation/consumed"
FORBIDDEN = "ri_codex2_tokenfactory_live_160"
SHA = re.compile(r"[0-9a-f]{64}\Z")
CONTEXT_FIELDS = {
    "schema", "operation_id", "scope_key", "condition", "outer_protocol", "intent",
    "consumption", "producer_manifest", "checkpoint_sha256", "recovery_receipt_sha256",
    "authority_sha256", "effect_key", "body_entrypoint",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lexical(raw: object, label: str, roots: tuple[Path, ...]) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError(f"{label} path differs")
    text = os.fspath(raw)
    normalized = text.replace("\\", "/")
    if (not text or FORBIDDEN in normalized.casefold() or not PurePath(text).is_absolute()
            or ".." in PurePath(text).parts):
        raise ValueError(f"{label} path is forbidden")
    candidate = Path(os.path.abspath(text))
    folded = str(candidate).casefold()
    if not any(folded == str(root).casefold()
               or folded.startswith(str(root).rstrip("\\/").casefold() + os.sep.casefold())
               for root in roots):
        raise ValueError(f"{label} path is outside the allowed root")
    return candidate


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _ref(value: object, label: str, roots: tuple[Path, ...]) -> tuple[dict[str, str], Path]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = _lexical(value.get("path"), label, roots)
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA.fullmatch(digest) is None:
        raise ValueError(f"{label} SHA differs")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or _sha(resolved) != digest:
        raise ValueError(f"{label} reference changed")
    return {"path": str(resolved), "sha256": digest}, resolved


def validate_context(context_path: Path, *, repo_root: Path = ROOT) -> Path:
    run_root = repo_root / "bench/private/post_actuator_confirmation/runs"
    consumed_root = repo_root / "bench/private/post_actuator_confirmation/consumed"
    context_lexical = _lexical(context_path, "context", (run_root,))
    context_path = context_lexical.resolve(strict=True)
    context = _read(context_path)
    if (set(context) != CONTEXT_FIELDS or context.get("schema") != "post_actuator_child_context_v1"
            or context.get("condition") != "NONE"
            or context.get("body_entrypoint") != "bench.actuator_envelope_runtime.child_main"):
        raise ValueError("post child context differs")
    for name in ("operation_id", "scope_key", "checkpoint_sha256", "recovery_receipt_sha256",
                 "authority_sha256", "effect_key"):
        if not isinstance(context.get(name), str) or not context[name]:
            raise ValueError(f"context {name} differs")
    protocol_ref, protocol_path = _ref(context["outer_protocol"], "outer protocol", (repo_root,))
    intent_ref, intent_path = _ref(context["intent"], "intent", (run_root,))
    consumption_ref, consumption_path = _ref(context["consumption"], "consumption", (consumed_root,))
    manifest_ref, manifest_path = _ref(context["producer_manifest"], "producer manifest", (repo_root,))
    protocol = _read(protocol_path)
    if (protocol.get("schema") != "post_actuator_operational_protocol_v5"
            or protocol.get("status") != "registered" or protocol.get("condition") != "NONE"
            or protocol.get("max_sdk_launches") != 1
            or protocol.get("child_entrypoint") != "bench/post_actuator_child_v4.py"):
        raise ValueError("outer protocol child contract differs")
    intent, consumption, manifest = _read(intent_path), _read(consumption_path), _read(manifest_path)
    expected = {
        "operation_id": context["operation_id"], "scope_key": context["scope_key"],
        "condition": "NONE", "checkpoint_sha256": context["checkpoint_sha256"],
        "recovery_receipt_sha256": context["recovery_receipt_sha256"],
        "authority_sha256": context["authority_sha256"], "effect_key": context["effect_key"],
    }
    if (intent.get("schema") != "post_actuator_operation_intent_v2"
            or intent.get("status") != "reserved_pre_sdk"
            or any(intent.get(key) != value for key, value in expected.items())
            or intent.get("outer_protocol") != protocol_ref):
        raise ValueError("operation intent differs")
    if (consumption.get("schema") != "post_actuator_global_consumption_v2"
            or consumption.get("status") != "consumed_pre_sdk"
            or any(consumption.get(key) != value for key, value in expected.items())
            or consumption.get("intent") != intent_ref
            or consumption.get("outer_protocol") != protocol_ref):
        raise ValueError("global consumption differs")
    if (manifest.get("schema") != "actuator_envelope_runtime_manifest_v1"
            or manifest.get("condition") != "NONE"
            or manifest_ref["path"] != str(manifest_path)):
        raise ValueError("producer manifest is not NONE")
    if context.get("outer_protocol") != protocol_ref or context.get("intent") != intent_ref:
        raise ValueError("context reference normalization differs")
    if context.get("consumption") != consumption_ref or context.get("producer_manifest") != manifest_ref:
        raise ValueError("context evidence reference differs")
    return manifest_path


def run(context_path: Path, delegate: Callable[[Path], int] | None = None, *,
        repo_root: Path = ROOT) -> int:
    manifest = validate_context(context_path, repo_root=repo_root)
    if delegate is None:
        for folder in (repo_root, repo_root / "bench", repo_root / "src"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        from bench import actuator_envelope_runtime as body
        delegate = body.child_main
    return delegate(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        if not args.validate_only and args.repo_root != ROOT:
            raise ValueError("alternate repo root is validation-only")
        if args.validate_only:
            manifest = validate_context(args.context, repo_root=args.repo_root)
            forbidden = sorted(name for name in sys.modules
                               if name.split(".", 1)[0] in {"isaaclab", "omni", "torch"})
            print(json.dumps({"status": "validated", "manifest": str(manifest),
                              "sdk_app_started": False, "forbidden_imports": forbidden}, sort_keys=True))
            return 0
        return run(args.context)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

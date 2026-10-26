"""first-class record readiness 검증을 주입한 영상 선택 worker 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import vision_record_selection_inputs_v2 as INPUTS  # pyright: ignore[reportMissingImports]
import vision_selection_worker_v1 as WORKER  # pyright: ignore[reportMissingImports]


def prepare(
    protocol_path: Path,
    protocol_sha256_lf: str,
    source_anchor: str,
    input_locations: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    return WORKER.prepare(
        protocol_path,
        protocol_sha256_lf,
        source_anchor,
        input_locations,
        root=root,
        input_verifier=INPUTS.verify_inputs,
    )


def advance(
    protocol_path: Path,
    protocol_sha256_lf: str,
    source_anchor: str,
    input_locations: Path,
    call_id: str,
    *,
    root: Path = ROOT,
    provider_preflight: Callable[[], None] = WORKER._preflight_provider,
) -> tuple[dict[str, Any], int]:
    return WORKER.advance(
        protocol_path,
        protocol_sha256_lf,
        source_anchor,
        input_locations,
        call_id,
        root=root,
        input_verifier=INPUTS.verify_inputs,
        provider_preflight=provider_preflight,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "advance"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--input-locations", type=Path, required=True)
    parser.add_argument("--call-id")
    args = parser.parse_args(argv)
    if args.command == "verify":
        if args.call_id is not None:
            parser.error("verify에는 --call-id를 쓰지 않는다")
        result = prepare(
            args.protocol,
            args.protocol_sha256_lf,
            args.source_anchor,
            args.input_locations,
        )
        print(
            json.dumps(
                {"status": "verified", "plan": result["plan"]},
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.call_id is None:
        parser.error("advance에는 --call-id가 필요하다")
    result, code = advance(
        args.protocol,
        args.protocol_sha256_lf,
        args.source_anchor,
        args.input_locations,
        args.call_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

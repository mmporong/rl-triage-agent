"""annotated render worker를 preflight와 source/render controller에 주입한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_brev_preflight_v1 as PREFLIGHT
import run_vision_brev_render_v2 as RENDER
import vision_brev_render_backend_v3 as BACKEND  # pyright: ignore[reportMissingImports]


def _required(
    parser: argparse.ArgumentParser, values: dict[str, Any], purpose: str
) -> None:
    missing = [name for name, value in values.items() if value is None]
    if missing:
        parser.error(f"{purpose}에는 {', '.join(missing)} 인자가 필요하다")


def _render(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    common = (
        args.repo,
        args.protocol,
        args.protocol_sha256_lf,
        args.render_protocol,
        args.render_protocol_sha256_lf,
    )
    if args.command == "finalize":
        _required(parser, {"materials": args.materials}, "render finalize")
        return RENDER.finalize_materials(*common, args.materials)
    if args.command == "status":
        parser.error("render phase에는 status가 없다")
    if args.command == "export" or (
        args.command == "advance"
        and RENDER._all_capture_receipts(args.repo, args.protocol)
    ):
        _required(
            parser,
            {
                "source-anchor": args.source_anchor,
                "resource-proof": args.resource_proof,
                "resource-sha256": args.resource_sha256,
            },
            "render export",
        )
        return RENDER.export_materials(
            *common,
            args.source_anchor,
            args.resource_proof,
            args.resource_sha256,
            resource_locations_path=args.resource_locations,
        )
    _required(
        parser,
        {
            "source-anchor": args.source_anchor,
            "resource-proof": args.resource_proof,
            "resource-sha256": args.resource_sha256,
            "isaaclab": args.isaaclab,
            "python": args.python,
        },
        f"render {args.command}",
    )
    batch = RENDER.RenderBatch(
        *common,
        args.source_anchor,
        args.resource_proof,
        args.resource_sha256,
        args.isaaclab,
        args.python,
        backend_factory=BACKEND.AnnotatedVisionBrevBackend.from_runtime,
    )
    return batch.verify() if args.command == "verify" else batch.advance()


def _preflight(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    if args.command in {"export", "finalize"}:
        parser.error(f"preflight phase에는 {args.command}가 없다")
    _required(
        parser,
        {
            "preflight-protocol": args.preflight_protocol,
            "preflight-protocol-sha256-lf": args.preflight_protocol_sha256_lf,
            "source-anchor": args.source_anchor,
            "resource-proof": args.resource_proof,
            "resource-sha256": args.resource_sha256,
        },
        f"preflight {args.command}",
    )
    offline = (
        args.repo,
        args.protocol,
        args.protocol_sha256_lf,
        args.render_protocol,
        args.render_protocol_sha256_lf,
        args.preflight_protocol,
        args.preflight_protocol_sha256_lf,
        args.source_anchor,
        args.resource_proof,
        args.resource_sha256,
    )
    result: dict[str, Any] | None = None
    if args.command == "status":
        result = PREFLIGHT.inspect_preflight(*offline)
    elif args.command == "advance":
        result = PREFLIGHT.settle_preflight(*offline)
    if result is not None:
        return result
    _required(
        parser,
        {"isaaclab": args.isaaclab, "python": args.python},
        f"preflight {args.command}",
    )
    batch = PREFLIGHT.PreflightBatch(
        *offline,
        args.isaaclab,
        args.python,
        backend_factory=BACKEND.AnnotatedVisionBrevBackend.from_runtime,
    )
    return batch.verify() if args.command == "verify" else batch.advance()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "render"))
    parser.add_argument(
        "--command",
        required=True,
        choices=("verify", "advance", "status", "export", "finalize"),
    )
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--render-protocol", required=True, type=Path)
    parser.add_argument("--render-protocol-sha256-lf", required=True)
    parser.add_argument("--preflight-protocol", type=Path)
    parser.add_argument("--preflight-protocol-sha256-lf")
    parser.add_argument("--source-anchor")
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--resource-locations", type=Path)
    args = parser.parse_args(argv)
    args.repo = Path(args.repo).resolve(strict=True)
    result = (
        _preflight(args, parser) if args.phase == "preflight" else _render(args, parser)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())

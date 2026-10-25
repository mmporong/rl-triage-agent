"""Atomic Writer backend를 기존 offline settle 순서에 연결한다."""

from __future__ import annotations

import run_vision_brev_annotated_v2 as ENTRY
import vision_brev_render_backend_v5 as BACKEND  # pyright: ignore[reportMissingImports]


def main(argv: list[str] | None = None) -> int:
    return ENTRY.main(
        argv, backend_factory=BACKEND.AtomicVisionBrevBackend.from_runtime
    )


if __name__ == "__main__":
    raise SystemExit(main())

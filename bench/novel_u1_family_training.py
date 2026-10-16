"""신규 seed U1 함수군 교정 manifest를 기존 학습 하네스로 실행한다."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_correction_u1_family as BASE  # noqa: E402
import novel_u1_scope_confirmation as H  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    H.configure()
    BASE.H = H.BASE
    BASE.install_fault(args.manifest)
    BASE.RUNNER.execute(args.manifest, H.BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

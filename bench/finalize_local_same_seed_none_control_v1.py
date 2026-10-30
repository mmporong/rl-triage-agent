"""NONE same-seed control source commit을 등록 protocol로 봉인한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_same_seed_none_control_v1 import BASE, PROTOCOL, ROOT, SOURCE_FILES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    value = BASE.read(args.template)
    if (
        value.get("status") != "prepared_not_registered"
        or args.out.exists()
        or args.out.resolve() != PROTOCOL.resolve()
    ):
        raise ValueError("template 상태 또는 출력 경로가 잘못됐다")
    names = tuple(
        name for name in SOURCE_FILES if name != PROTOCOL.relative_to(ROOT).as_posix()
    )
    blobs = BASE.git_blob_map(ROOT, args.source_commit, names)
    current = {name: BASE.sha(ROOT / name, lf=True) for name in names}
    if blobs != current:
        raise ValueError("source commit blob과 현재 source가 다르다")
    value.update(
        {
            "status": "registered",
            "registration_anchor": args.source_commit,
            "source_sha256_lf": blobs,
        }
    )
    BASE.write_new(args.out, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""v3 source commit과 runtime/package pin을 registered protocol로 봉인한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_f1_restart_base_v3 import (
    ROOT,
    SOURCE_FILES,
    git_blob_map,
    read,
    sha,
    write_new,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    value = read(args.template)
    if value.get("status") != "prepared_not_registered" or args.out.exists():
        raise ValueError("template 상태 또는 출력 경로가 잘못됐다")
    names = tuple(
        name
        for name in SOURCE_FILES
        if name != "bench/protocols/local_f1_restart_s3221_v3.json"
    )
    blobs = git_blob_map(ROOT, args.source_commit, names)
    current = {name: sha(ROOT / name, lf=True) for name in names}
    if blobs != current:
        raise ValueError("source commit blob과 현재 source가 다르다")
    value["status"], value["registration_anchor"], value["source_sha256_lf"] = (
        "registered",
        args.source_commit,
        blobs,
    )
    write_new(args.out, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

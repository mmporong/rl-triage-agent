"""Render the failed v1 cohort without changing its frozen driver or raw results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def redact(value):
    if isinstance(value, str):
        value = value.replace(str(ROOT).replace("\\", "\\\\") + "\\\\", "")
        value = value.replace(str(ROOT) + "\\", "").replace(ROOT.as_posix() + "/", "")
        return re.sub(r"\\+", "/", value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def reject_absolute_paths(value):
    if isinstance(value, str) and re.search(r"[A-Za-z]:[\\/]|(?:^|[ '\"])/(?!/)[^ /]+(?:/|$)", value):
        raise ValueError("공개 감사에 호스트 절대 경로가 남아 있다")
    if isinstance(value, list):
        for item in value:
            reject_absolute_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            reject_absolute_paths(item)


def publish_new(outputs):
    if any(path.exists() for path in outputs):
        raise FileExistsError("공개 산출물을 덮어쓰지 않는다")
    created = []
    try:
        for path, content in outputs.items():
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                created.append(path)
                handle.write(content)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_audit = json.loads(args.audit.read_text(encoding="utf-8"))
    public_audit = redact(raw_audit)
    reject_absolute_paths(public_audit)
    public_text = json.dumps(public_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    spec = importlib.util.spec_from_file_location("failed_v1_renderer", ROOT / "evals/build_active_replay.py")
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    summary = renderer.build_summary(args.tag, args.audit)
    if any(case["proof"] for case in summary["cases"]):
        raise ValueError("이 어댑터는 판단 변화가 입증되지 않은 v1 기록만 표시한다")
    summary["claim"] = "판단 변화 입증 0건 · 실패와 HOLD를 보존한 실행 기록"
    audit_relative = args.audit.resolve().relative_to(ROOT).as_posix()
    public_relative = args.public_audit.resolve().relative_to(ROOT).as_posix()
    summary["audit"] = public_relative
    for source in summary["sources"]:
        if source["path"] == audit_relative:
            source.update(path=public_relative, sha256_lf=hashlib.sha256(public_text.encode()).hexdigest())
    for case in summary["cases"]:
        result = case.get("result") or {}
        stop = result.get("error") or result.get("stop_reason")
        if stop:
            case["reasons"].insert(0, "실행 중단: " + stop)
    summary["presentation_adapter"] = "evals/render_failed_active.py; failure wording and stop reasons only"
    rendered = renderer.render_html(summary)
    original = 'if(si===3){pair(dl,"갱신 1위"'
    updated = 'if(si===3){pair(dl,"중단 원인",r.error||r.stop_reason);pair(dl,"갱신 1위"'
    if rendered.count(original) != 1:
        raise ValueError("v1 렌더링 구조가 달라 중단 원인 표시를 적용할 수 없다")
    rendered = rendered.replace(original, updated, 1)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    publish_new({args.public_audit: public_text, args.out.with_suffix(".json"): summary_text, args.out: rendered})
    print(json.dumps({"html": str(args.out), "proof_changed": 0}))


if __name__ == "__main__":
    main()

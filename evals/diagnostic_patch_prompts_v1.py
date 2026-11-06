"""진단·패치 생성·검사 피드백 수정의 고정 메시지 계약."""
from __future__ import annotations

import json

import ppo_patch_cases_v1 as BASE


def messages(kind: str, context: dict) -> list[dict]:
    if kind == "diagnosis":
        return [{"role": "system", "content": (
            "Investigate the supplied PPO configuration adapter. Select the registered CPU check if "
            "configuration preservation needs verification, or hold if no check is justified. "
            "Do not claim that a fault is confirmed before a check result. Use only supplied source and facts. "
            "Return one JSON object with exactly action, probe, source_sha256, fact_ids, rationale. "
            "action is inspect or hold; probe is P_optimizer_updates_cpu_adapter for inspect and null for hold. "
            "Copy source_sha256 exactly. Cite at least one distinct existing fact ID. "
            "The proposal grants no execution permission.")},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)}]
    if kind not in {"patch", "revision"}:
        raise ValueError("등록되지 않은 모델 단계다")
    result = BASE.messages(context)
    result[0]["content"] += (
        " Preserve optional-argument omission and the installed constructor default; never invent a fallback "
        "that changes a missing configuration value. The host will check multiple configurations and "
        "unrelated parameters as well as actual PPO updates.")
    if kind == "revision":
        result[0]["content"] += (
            " This is the single registered revision after actual CPU validation. Read the supplied "
            "failed checks and propose a new replacement for the current candidate source. "
            "No further revision is available. The host does not supply replacement code.")
    if any(key.startswith("runtime.seed") for key in context.get("facts", {})):
        result[0]["content"] += (
            " Evidence IDs must exactly equal keys in the top-level facts mapping. Nested values "
            "are not separate evidence IDs. For feedback, inspect every failed_checks entry: "
            "passing counts under one configuration do not establish correctness under the others.")
    return result

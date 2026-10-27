"""인프라 재시도의 선행 비용과 촬영 원장을 결합한다."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA = "vision_capture_infrastructure_retry_v1"


def _amount(value: Any) -> Decimal:
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("비용은 유한한 0 이상의 수여야 한다") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("비용은 유한한 0 이상의 수여야 한다")
    return amount


def _reference(repo: Path, row: Any, *, committed: bool) -> Path:
    if not isinstance(row, dict) or set(row) != {"path", "sha256_lf"}:
        raise ValueError("참조 원문 형식이 다르다")
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("참조는 저장소 상대 경로여야 한다")
    path = repo / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("참조 원문이 일반 파일이 아니다")
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != row["sha256_lf"]:
        raise ValueError("참조 원문의 SHA가 다르다")
    if committed:
        found = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative.as_posix()}"],
            check=True, capture_output=True,
        ).stdout.replace(b"\r\n", b"\n")
        if found != raw:
            raise ValueError("참조 원문이 HEAD와 다르다")
    return path


def validate_retry(path: Path, *, repo: Path, committed: bool = True) -> dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    relative = path.relative_to(repo)
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if committed:
        version = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative.as_posix()}"],
            check=True, capture_output=True,
        ).stdout.replace(b"\r\n", b"\n")
        if raw != version:
            raise ValueError("재시도 사전등록이 HEAD와 다르다")
    retry = json.loads(raw)
    if (
        retry.get("schema") != SCHEMA
        or retry.get("registered_before_execution") is not True
        or retry.get("max_start_requests") != 1
        or retry.get("new_training_actions") != 0
        or retry.get("model_api_requests") != 0
        or retry.get("max_full_recordings") != 2
        or retry.get("retry_reason") != "infrastructure_readiness_timeout"
    ):
        raise ValueError("재시도 범위가 다르다")
    paths = {
        key: _reference(repo, retry[key], committed=committed)
        for key in ("record_protocol", "resource_authorization", "prior_result", "budget_guard")
    }
    try:
        registered = datetime.fromisoformat(retry["registered_at_utc"].replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("재시도 등록 시각이 다르다") from error
    if registered.tzinfo is None or registered > datetime.now(timezone.utc):  # noqa: UP017
        raise ValueError("재시도 등록 시각이 미래거나 시간대가 없다")
    prior = json.loads(paths["prior_result"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["record_protocol"].read_text(encoding="utf-8"))
    authorization = json.loads(paths["resource_authorization"].read_text(encoding="utf-8"))
    resource = retry.get("resource", {})
    original_resource = protocol.get("resources", {})
    limits = protocol.get("limits", {})
    cost = prior.get("cost", {})
    if (
        prior.get("status") != "HOLD"
        or prior.get("failure", {}).get("stage") != "instance_readiness"
        or prior.get("cleanup", {}).get("vm_status") != "STOPPED"
        or prior.get("actual_recordings") != 0
        or prior.get("new_training_actions") != 0
        or prior.get("model_api_requests") != 0
        or prior.get("protocol") != retry["record_protocol"]
        or protocol.get("protocol_id") != prior.get("protocol_id")
        or authorization.get("execution_protocol") != retry["record_protocol"]
        or resource.get("instance_id") != authorization.get("resource_id")
        or resource.get("instance_name") != authorization.get("resource_name")
        or resource.get("instance_id") != original_resource.get("owned_instance_id")
        or resource.get("instance_name") != original_resource.get("owned_instance_name")
        or limits.get("max_policies") != retry["max_full_recordings"]
        or limits.get("training_actions") != 0
        or limits.get("model_api_requests") != 0
        or protocol.get("checkpoints") != authorization.get("checkpoint_catalog")
        or retry["budget_guard"]["path"] != "bench/vision_capture_retry_budget_v1.py"
    ):
        raise ValueError("선행 실행이 촬영 전 인프라 실패가 아니다")
    prior_cost = _amount(cost.get("cumulative_estimate_including_upper_bound_usd"))
    remaining = _amount(cost.get("remaining_program_budget_lower_bound_usd"))
    cap = _amount(cost.get("program_cap_usd"))
    budget = retry.get("budget", {})
    if (
        _amount(budget.get("prior_upper_bound_usd")) != prior_cost
        or _amount(budget.get("remaining_lower_bound_usd")) != remaining
        or _amount(budget.get("program_cap_usd")) != cap
        or prior_cost + remaining != cap
        or _amount(budget.get("compute_quote_usd_per_h")) != Decimal("1.74")
        or _amount(budget.get("reserved_for_two_recordings_usd")) != Decimal("0.58")
        or remaining < Decimal("0.58")
    ):
        raise ValueError("선행 비용과 재시도 예산이 다르다")
    return retry


@contextmanager
def apply_budget(budget_module: Any, retry: Mapping[str, Any]) -> Iterator[None]:
    """기존 단계별 검사를 재사용하면서 선행 실패 비용을 포함한다."""
    budget = retry["budget"]
    prior = _amount(budget["prior_upper_bound_usd"])
    remaining = _amount(budget["remaining_lower_bound_usd"])
    cap = _amount(budget["program_cap_usd"])
    quote = _amount(budget["compute_quote_usd_per_h"])
    if prior + remaining != cap or budget_module.TOTAL != cap or budget_module.QUOTE != quote:
        raise ValueError("실행 예산 모듈과 재시도 사전등록이 다르다")
    old_prior, old_fresh = budget_module.PRIOR, budget_module.FRESH
    budget_module.PRIOR, budget_module.FRESH = prior, remaining
    try:
        yield
    finally:
        budget_module.PRIOR, budget_module.FRESH = old_prior, old_fresh

"""공용 GPU lease의 .NET/Python ISO 시각을 Python 3.10에서 읽는다."""
from __future__ import annotations

from datetime import datetime, timezone
import re

_STAMP = re.compile(
    r"(?P<second>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,7}))?(?P<zone>Z|[+-][0-9]{2}:[0-9]{2})\Z"
)


def parse_lease_timestamp(value: str) -> datetime:
    """시간대가 있는 lease 시각을 UTC로 반환한다. 7번째 소수 자리는 버린다.

    .NET round-trip 형식은 100 ns 단위이고 datetime은 1 us 단위다.
    소수부를 오른쪽 0으로 채우거나 여섯 자리에서 잘라 3.10과 호환한다.
    원문 metadata와 heartbeat 허용 시간은 변경하지 않는다.
    """
    match = _STAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("lease 시각은 시간대와 최대 7자리 소수 초가 있는 ISO 형식이어야 한다")
    zone = match["zone"]
    if zone == "Z":
        zone = "+00:00"
    elif int(zone[1:3]) > 23 or int(zone[4:6]) > 59:
        raise ValueError("lease 시각의 시간대 오프셋이 유효하지 않다")
    fraction = match["fraction"]
    normalized = match["second"] + ("." + fraction[:6].ljust(6, "0") if fraction else "") + zone
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)

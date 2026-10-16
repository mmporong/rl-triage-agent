"""lease 시각의 자릿수·시간대 계약을 고정한다."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from lease_timestamps import parse_lease_timestamp


@pytest.mark.parametrize("fraction,microseconds", [
    ("", 0), (".4", 400000), (".46", 460000), (".468", 468000),
    (".4686", 468600), (".46864", 468640), (".468649", 468649),
    (".4686492", 468649), (".9999999", 999999), (".0000001", 0),
])
@pytest.mark.parametrize("zone,hour", [("Z", 23), ("+00:00", 23), ("+09:00", 14)])
def test_fraction_is_padded_or_truncated_without_rounding(fraction, microseconds, zone, hour):
    assert parse_lease_timestamp(f"2026-10-16T23:02:45{fraction}{zone}") == datetime(
        2026, 10, 16, hour, 2, 45, microseconds, timezone.utc)


def test_negative_offset_rolls_date_and_keeps_utc():
    value = parse_lease_timestamp("2026-10-16T23:02:45.1234567-03:30")
    assert value == datetime(2026, 10, 17, 2, 32, 45, 123456, timezone.utc)
    assert value.tzinfo is timezone.utc


@pytest.mark.parametrize("value", [
    None, 123, b"2026-10-16T23:02:45Z", "", "2026-10-16", "2026-10-16T23:02:45",
    "2026-10-16 23:02:45Z", "2026-10-16T23:02:45.Z", "2026-10-16T23:02:45.12345678Z",
    "2026-10-16T23:02:45,123Z", "2026-10-16T23:02:45z", "2026-10-16T23:02:45+0000",
    "2026-10-16T23:02:45+24:00", "2026-10-16T23:02:45+00:60", "2026-10-16T23:02:45Z\n",
    " 2026-10-16T23:02:45Z", "2026-02-30T23:02:45Z", "2026-10-16T23:02:60Z",
    "2026-10-16T24:02:45Z", "２０２６-10-16T23:02:45Z",
])
def test_malformed_or_naive_input_is_rejected(value):
    with pytest.raises(ValueError):
        parse_lease_timestamp(value)

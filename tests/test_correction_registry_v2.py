from rl_triage.correction_registry import REGISTRY as LEGACY
from rl_triage.correction_registry_v2 import REGISTRY, correction_for, registry_digest


def test_legacy_entries_are_byte_value_preserved() -> None:
    assert {key: REGISTRY[key] for key in LEGACY} == LEGACY


def test_optimizer_recoveries_are_restart_only_and_distinct() -> None:
    count = correction_for("optimizer_update_count")
    gradient = correction_for("optimizer_surrogate")
    assert count["correction_id"] == "restore_configured_ppo_epochs"
    assert gradient["correction_id"] == "restore_clipped_ppo_surrogate"
    assert count["recovery_modes"] == gradient["recovery_modes"] == ["restart"]
    assert registry_digest() == registry_digest(REGISTRY)


def test_returned_entry_does_not_mutate_registry() -> None:
    row = correction_for("optimizer_update_count")
    row["recovery_modes"].append("resume")
    assert REGISTRY["optimizer_update_count"]["recovery_modes"] == ["restart"]

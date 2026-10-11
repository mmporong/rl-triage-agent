"""Brev probe backend의 transport·privacy·binding 경계를 실제 cloud 호출 없이 검증한다."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("brev_probe_backend_test", ROOT / "evals/brev_probe_backend.py")
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def protocol():
    return {
        "probe_backend": {"name": "brev", "instance": "marked-lavender-warbler",
                          "workspace_basename": "rl-triage-active-evidence-20261011",
                          "environment": dict(B.EXPECTED_ENVIRONMENT),
                          "owner_marker_sha256": "9" * 64},
        "probe_arguments": {"P_reward": {"num_envs": 256}},
        "probe_execution_budget_s": 180,
        "code_sha256_lf": {"evals/probes.py": "a" * 64},
    }


@pytest.mark.parametrize("field,value", [
    ("instance", "marked-lavender-warbler; touch /tmp/pwn"),
    ("instance", "$(id)"),
    ("workspace_basename", "../other-workspace"),
    ("workspace_basename", "space name"),
])
def test_config_rejects_command_and_path_injection(field, value):
    config = protocol()["probe_backend"] | {field: value}
    with pytest.raises(ValueError):
        B.validate_backend_config(config)


def test_native_config_remains_minimal_and_compatible():
    assert B.backend_config("native_windows") == {"name": "native_windows"}
    with pytest.raises(ValueError, match="미등록 필드"):
        B.validate_backend_config({"name": "native_windows", "instance": "unused"})


def test_marker_parser_accepts_one_envelope_and_ignores_brev_footer_only_as_text():
    marker = B._encoded_marker({"ready": True})
    assert B.parse_marker(f"remote chatter\n{marker}\nmarked-lavender-warbler\n") == {"ready": True}
    with pytest.raises(B.BrevBackendError, match="MarkerContractError"):
        B.parse_marker(marker + "\n" + marker)
    with pytest.raises(B.BrevBackendError, match="MarkerContractError"):
        B.parse_marker('{"ready": true}\nmarked-lavender-warbler')


def test_transport_failure_is_typed_and_does_not_publish_secret_or_command(tmp_path):
    def fail(*args, **kwargs):
        raise OSError(r"token=secret C:\Users\alice\private")

    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=fail)
    with pytest.raises(B.BrevBackendError) as raised:
        backend._exec("python3 -c harmless", "gpu snapshot")
    assert raised.value.error_type == "OSError"
    assert "secret" not in str(raised.value) and "alice" not in str(raised.value)


def test_cloud_busy_snapshot_is_fail_closed_before_approval_and_uses_default_exec(tmp_path):
    calls = []
    snapshot = {"gpu_utilization_pct": 17, "memory_mib": 4096, "memory_total_mib": 46068,
                "other_compute_pids": [321], "ready": False,
                "busy_reasons": ["other_gpu_compute", "unexplained_large_allocation"]}

    def runner(args, **kwargs):
        calls.append(args)
        if args[-2:] == ["printenv", "HOME"]:
            return SimpleNamespace(returncode=0, stdout="/home/local-user\n")
        return SimpleNamespace(returncode=0, stdout=B._encoded_marker(snapshot) + "\nmarked-lavender-warbler\n")

    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=runner)
    with pytest.raises(B.BrevGPUWaitError) as raised:
        backend.gpu_snapshot()
    assert raised.value.snapshot == snapshot
    assert calls[0] == ["wsl", "-d", "Ubuntu", "--", "printenv", "HOME"]
    assert calls[1][:6] == ["wsl", "-d", "Ubuntu", "--",
                            "/home/local-user/.local/bin/brev", "exec"]
    assert calls[1][6] == "marked-lavender-warbler"
    assert "--host" not in calls[1] and set(snapshot) == {
        "gpu_utilization_pct", "memory_mib", "memory_total_mib",
        "other_compute_pids", "ready", "busy_reasons"}


def test_transport_discovers_wsl_home_and_uses_brev_argv(tmp_path):
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        if args[-2:] == ["printenv", "HOME"]:
            return SimpleNamespace(returncode=0, stdout="/home/local-user\n")
        return SimpleNamespace(returncode=0, stdout=B._encoded_marker({"ok": True}))

    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=runner)
    assert backend._exec(B._remote_python("print('ignored')"), "test") == {"ok": True}
    assert calls[0] == ["wsl", "-d", "Ubuntu", "--", "printenv", "HOME"]
    assert calls[1][:6] == ["wsl", "-d", "Ubuntu", "--",
                            "/home/local-user/.local/bin/brev", "exec"]
    assert not any("/home/llroad" in argument for call in calls for argument in call)
    assert "--host" not in calls[1]


def test_transport_has_separate_bounded_timeout(tmp_path):
    timeouts = []

    def runner(args, **kwargs):
        timeouts.append(kwargs["timeout"])
        if args[-2:] == ["printenv", "HOME"]:
            return SimpleNamespace(returncode=0, stdout="/home/local-user\n")
        return SimpleNamespace(returncode=0, stdout=B._encoded_marker({"ok": True}))

    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=runner)
    backend._exec(B._remote_python("print('ignored')"), "test")
    assert timeouts == [B.TRANSPORT_TIMEOUT_S, B.TRANSPORT_TIMEOUT_S]


def test_environment_verification_uses_task_venv_and_exact_contract(tmp_path, monkeypatch):
    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=lambda *a, **k: pytest.fail("runner called"))
    workspace = "/home/cloud-user/rl-triage-active-evidence-20261011"
    calls = []
    actual = dict(B.EXPECTED_ENVIRONMENT)
    actual.update(ubuntu_version="22.04.5 LTS (Jammy Jellyfish)", owner_marker_sha256="9" * 64)
    monkeypatch.setattr(backend, "workspace", lambda *args: workspace)

    def fake_exec(command, stage, deadline=None):
        calls.append((command, stage, deadline))
        return actual

    monkeypatch.setattr(backend, "_exec", fake_exec)
    assert backend._verify_environment() == actual
    assert calls[0][0].startswith(f"{workspace}/venv/bin/python -c ")
    assert not calls[0][0].startswith("python3 ")
    assert B.EXPECTED_ENVIRONMENT["ubuntu"] == "22.04"
    assert B.EXPECTED_ENVIRONMENT["torch"] == "2.7.0+cu128"
    assert B.EXPECTED_ENVIRONMENT["isaac_sim"] == "4.5.0.0"


def test_environment_contract_rejects_patch_or_prefix_only_package_versions():
    actual = dict(B.EXPECTED_ENVIRONMENT)
    actual["ubuntu_version"] = "22.04.5 LTS (Jammy Jellyfish)"
    assert B.environment_matches(actual)
    for key, value in (("ubuntu", "22.04.5"), ("torch", "2.7"), ("isaac_sim", "4.5"),
                       ("cuda", "12.8"), ("cuda", "12.8.5")):
        changed = dict(actual)
        changed[key] = value
        assert not B.environment_matches(changed)


def test_cleanup_evidence_accepts_only_empty_owned_group_and_gpu():
    clean = {"cleanup_checked": True, "cleanup_check_error": False,
             "cleanup_group_alive": False, "cleanup_compute_pids": [],
             "cleanup_memory_mib": 0, "cleanup_wall_s": 0.2,
             "cleanup_confirmed": True}
    assert B.validate_cleanup_evidence(clean)["cleanup_memory_mib"] == 0


@pytest.mark.parametrize("change", [
    {"cleanup_group_alive": True},
    {"timed_out": True, "cleanup_group_alive": True},
    {"cleanup_compute_pids": [991]},
    {"cleanup_memory_mib": 4096},
    {"cleanup_checked": False, "cleanup_check_error": True},
])
def test_cleanup_evidence_rejects_child_gpu_or_inspection_residue(change):
    outcome = {"cleanup_checked": True, "cleanup_check_error": False,
               "cleanup_group_alive": False, "cleanup_compute_pids": [],
               "cleanup_memory_mib": 0, "cleanup_wall_s": 0.2,
               "cleanup_confirmed": True}
    outcome.update(change)
    with pytest.raises(B.BrevCleanupUnverified):
        B.validate_cleanup_evidence(outcome)


def test_preflight_refuses_existing_local_artifact_without_transport(tmp_path):
    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=lambda *a, **k: pytest.fail("transport called"))
    path = tmp_path / "evals/results/reused/probes/h01_s2027__P_reward.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(B.BrevBackendError, match="NoOverwriteError"):
        backend.preflight("h01_s2027", "P_reward", "reused")


def test_preflight_refuses_existing_public_meter_without_transport(tmp_path):
    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=lambda *a, **k: pytest.fail("transport called"))
    path = tmp_path / "evals/results/reused/meters/h01_s2027__P_reward.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(B.BrevBackendError, match="NoOverwriteError"):
        backend.preflight("h01_s2027", "P_reward", "reused")


def test_checkpoint_mismatch_stops_before_any_transport(tmp_path):
    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=lambda *a, **k: pytest.fail("transport called"))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"different")
    with pytest.raises(B.BrevBackendError, match="CheckpointDigestMismatch"):
        backend.execute("h01_s2027", "P_reward", "fresh", checkpoint, "0" * 64, 180)


def test_artifact_binding_rejects_checkpoint_report_and_jobs_mismatch():
    report = {"name": "h01_s2027", "checkpoint": {"sha256": "a" * 64}}
    row = {"probe": "P_reward", "checkpoint_sha256": "a" * 64, "completed": True}
    meter = {"completed": True, "measurements": [row], "jobs_sha256": "b" * 64,
             "probes_sha256_lf": "c" * 64}
    B.validate_probe_artifacts(report, meter, case="h01_s2027", probe="P_reward",
                               checkpoint_sha256="a" * 64, probes_sha256_lf="c" * 64,
                               jobs_sha256="b" * 64)
    for mutation in ("checkpoint", "report", "jobs"):
        changed_report = json.loads(json.dumps(report))
        changed_meter = json.loads(json.dumps(meter))
        kwargs = {"case": "h01_s2027", "probe": "P_reward", "checkpoint_sha256": "a" * 64,
                  "probes_sha256_lf": "c" * 64, "jobs_sha256": "b" * 64}
        if mutation == "checkpoint":
            changed_meter["measurements"][0]["checkpoint_sha256"] = "0" * 64
        elif mutation == "report":
            changed_report["name"] = "h02_s2027"
        else:
            changed_meter["jobs_sha256"] = "0" * 64
        with pytest.raises(B.BrevBackendError, match="ArtifactBindingMismatch"):
            B.validate_probe_artifacts(changed_report, changed_meter, **kwargs)


@pytest.mark.parametrize("private", [r"C:\Users\alice\secret.txt", "/home/shadeform/private/log.txt",
                                     "api_key=do-not-publish", "Bearer hidden-token"])
def test_public_report_privacy_filter_rejects_paths_and_credentials(private):
    assert B.contains_private_text({"probes": {"P_reward": {"error": private}}}) is True
    assert B.contains_private_text({"cpu_scope": private}) is True
    assert B.contains_private_text({"measurement": {"from": "model_0.pt"}}) is False


def test_prepare_execution_registers_private_jobs_and_public_hash_chain(tmp_path, monkeypatch):
    backend = B.BrevProbeBackend(tmp_path, protocol(), runner=lambda *a, **k: pytest.fail("raw runner called"))
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"last")
    dependency = tmp_path / "model_0.pt"
    dependency.write_bytes(b"first")
    ensured, copies = [], []
    monkeypatch.setattr(backend, "preflight", lambda *args: None)
    monkeypatch.setattr(backend, "_verify_sources", lambda *args: None)
    monkeypatch.setattr(backend, "_verify_environment", lambda: dict(B.EXPECTED_ENVIRONMENT))
    monkeypatch.setattr(backend, "workspace", lambda *args: "/home/shadeform/rl-triage-active-evidence-20261011")
    monkeypatch.setattr(backend, "_ensure_remote_checkpoint",
                        lambda local, remote, expected, deadline: ensured.append((local.name, expected)))
    monkeypatch.setattr(backend, "_fault", lambda case: ("F1", 2027))
    monkeypatch.setattr(backend, "_exec", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(backend, "_copy", lambda source, destination, stage, deadline: copies.append((source, destination)))
    owner = {"controller": "model", "protocol_sha256_lf": "d" * 64}
    plan = backend.prepare_execution(
        "h01_s2027", "P_value", "fresh", checkpoint,
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        {"P_value": {"file": "model_0.pt", "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}},
        owner)
    public = plan["public"]
    assert ensured == [("model_299.pt", hashlib.sha256(b"last").hexdigest()),
                       ("model_0.pt", hashlib.sha256(b"first").hexdigest())]
    assert len(copies) == 1 and Path(plan["jobs_path"]).is_file()
    assert public["jobs_sha256"] == hashlib.sha256(Path(plan["jobs_path"]).read_bytes()).hexdigest()
    assert public["source_set_digest"] == B.digest(protocol()["code_sha256_lf"])
    assert public["owner_binding_digest"] == B.digest(owner)
    assert public["checkpoint_set"][1]["role"] == "P_value:from"
    assert "fault" not in json.dumps(public).lower()
def test_transport_decodes_brev_utf8_independently_of_windows_locale(tmp_path):
    import sys

    protocol = {"probe_backend": B.backend_config("brev", owner_marker_sha256="a" * 64)}
    backend = B.BrevProbeBackend(tmp_path, protocol)
    result = backend._run([sys.executable, "-c", "import sys; sys.stdout.buffer.write('전송✓'.encode('utf-8'))"],
                          "utf8 transport")
    assert result.stdout == "전송✓"

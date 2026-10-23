# pyright: reportMissingImports=false
"""C4가 C3 의미 계약을 유지하고 새 인프라 재시도만 추가하는지 CPU에서 확인한다."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import ppo_ratio_repair_demo_v3 as V3  # noqa: E402
import ppo_ratio_repair_demo_v4 as D  # noqa: E402
from test_ppo_ratio_repair_demo_v3 import Backend, model_caller  # noqa: E402

C = D.C
TAG = "ppo_ratio_repair_s2165_20261023"
FROZEN_V3_LF_SHA = {
    "evals/ppo_ratio_repair_demo_v3.py": "cd85cd6a4d232e70918de6914ba8ea20c0e2cf1deff94106d36d8c0d30d31bd0",
    "tests/test_ppo_ratio_repair_demo_v3.py": "bbe41cebfa9903fd15048e586b6dd15286999d4d23b93ee203f003ce02e9c32e",
    "bench/protocols/ppo_ratio_repair_demo_v3.json": "194e0f551dd32f51124f938ed4f2df2cb7ad2768d87c04799790db560dee0864",
    "bench/run_ppo_ratio_repair_demo_v3.ps1": "9a4627b57e454a8113d447d25a70728dc16f590e92dfd0ec8e5f2dc07caf39c2",
    "evals/ppo_ratio_repair_execution_v3.py": "40bce19bb59eff05f761d4ddaaeee82b27fec233bff5db0c0dbc3790d77f207b",
    "evals/ppo_ratio_repair_model_call_v2.py": "c6d0d630c88b5395dbe70e93406c3a5bac17bce5360b70e096d08a0e30f2ac82",
    "evals/ppo_ratio_repair_contract_v2.py": "2551ea48a6a6f52d9de609890aad05e1f7d9af2b43975481da1fed52a3e5e0df",
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    original_sha = D.E.sha
    monkeypatch.setattr(D.E, "sha", lambda path, lf=False: "b" * 64 if path == D.PROTOCOL else original_sha(path, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda *args, **kwargs: {"status": "fresh", "seeds": [D.SEED, D.RESTART_SEED]}


def run(setup, backend=None, caller=None):
    return D.execute(
        TAG,
        Path("unused"),
        "operator",
        True,
        backend=backend or Backend(),
        caller=caller or model_caller(),
        seed_guard=setup,
    )


def test_c4_preserves_v3_semantics_and_records_infrastructure_retry():
    protocol = D.fixed_protocol()
    previous = V3.fixed_protocol()
    assert (protocol["seed"], protocol["restart_seed"]) == (2165, 3165)
    assert protocol["model"] == previous["model"]
    assert protocol["gpu"] == previous["gpu"]
    for key in (
        "question",
        "flow",
        "loop_registry",
        "correction_registry",
        "loop_rule",
        "inspection",
        "approval",
        "failure_policy",
        "prediction",
        "success_conditions",
        "outcome_wording",
        "interpretation",
    ):
        assert protocol[key] == previous[key]
    assert protocol["recovery_rule"] == {
        **previous["recovery_rule"],
        "reference": (
            "원 실패 seed의 NONE 정책을 기준으로 쓴다. source NONE/INVERSE_RATIO는 2165, "
            "교정 새 학습은 3165이다"
        ),
    }
    retry = protocol["c4_retry"]
    assert retry["previous_result_sha256"] == D.C3_RESULT["result_sha256"]
    assert retry["previous_protocol_sha256_lf"] == FROZEN_V3_LF_SHA["bench/protocols/ppo_ratio_repair_demo_v3.json"]
    assert retry["classification"] == "post_NONE_training_external_Kit_busy"
    assert retry["not_started"] == ["INVERSE_RATIO_training", "source_fixed_eval", "diagnosis", "probe_loop", "correction"]
    cost = retry["previous_cost"]
    assert cost["known_owned_child_wall_s_sum"] == 609.094
    assert sum(cost["components"].values()) == pytest.approx(cost["known_owned_child_wall_s_sum"])
    assert cost["components"] == {"training_parent_wall_s": 603.125, "checkpoint_parent_wall_s": 3.937,
                                  "telemetry_extract_parent_wall_s": 2.032}
    assert cost["api_requests"] == cost["fixed_evaluations"] == 0
    assert retry["api_ledger"] == {
        "v1": 1,
        "c2": 2,
        "c3": 0,
        "c4_cap": 4,
        "execution_attempts_cap": 7,
        "prior_diagnostic_requests_separate": 3,
    }
    assert protocol["model"]["infra_dialogue_attempts"] == 1
    assert protocol["model"]["feedback_retries"] == 1
    assert protocol["model"]["total_http_request_cap"] == 4


def test_c3_and_shared_backends_are_unchanged():
    assert {path: D.E.sha(ROOT / path, True) for path in FROZEN_V3_LF_SHA} == FROZEN_V3_LF_SHA


def test_complete_fixture_uses_new_seeds_and_v4_result_schema(setup):
    backend = Backend()
    result = run(setup, backend)
    assert result["schema"] == "ppo_ratio_repair_demo_result_v4"
    assert result["outcome"] == "recovered" and result["demo_success"]
    assert result["api_requests"] == 2
    assert ("train", "NONE", 2165) in backend.calls
    assert ("train", D.FAULT, 2165) in backend.calls
    assert ("train", "CORRECTED", 3165) in backend.calls
    assert result["design_coupling"]["c3_observation"] == D.C3_RESULT


def test_invalid_model_stops_after_single_dialogue_without_rule_fallback(setup):
    backend = Backend()
    result = run(setup, backend, model_caller(invalid=True))
    assert result["outcome"] == "model_invalid" and result["api_requests"] == 2
    assert not any(call[:2] == ("train", "CORRECTED") for call in backend.calls)
    assert not (D.PRIVATE / TAG / "model/diagnosis.feedback2.json").exists()


def test_seed_reuse_and_old_tag_stop_before_training(setup):
    backend = Backend()

    def reused(*args, **kwargs):
        raise ValueError("seed used")

    with pytest.raises(ValueError, match="seed used"):
        D.execute(TAG, Path("unused"), "operator", True, backend=backend, seed_guard=reused)
    with pytest.raises(ValueError, match="등록 tag"):
        D.execute(
            "ppo_ratio_repair_s2164_20261023",
            Path("unused"),
            "operator",
            True,
            backend=backend,
            seed_guard=setup,
        )
    assert backend.calls == [("preflight",)]
    assert not D.PRIVATE.exists()


def test_gpu_gate_recomputes_strict_receipt_and_rejects_busy():
    class Clear:
        @staticmethod
        def query_gpu():
            return {"busy": False, "reasons": []}

    class Busy:
        @staticmethod
        def query_gpu():
            return {"busy": True, "reasons": ["unapproved_active_engine"]}

    assert D.gpu_gate(Clear())["busy"] is False
    with pytest.raises(RuntimeError, match="strict GPU clear"):
        D.gpu_gate(Busy())


def test_wrapper_requires_factoryops_ack_gpu_gate_and_one_lease_command():
    text = (ROOT / "bench/run_ppo_ratio_repair_demo_v4.ps1").read_text(encoding="utf-8")
    assert "[switch]$FactoryOpsCompleteAck" in text
    assert "if (-not $FactoryOpsCompleteAck)" in text
    assert "gpu-gate" in text
    assert text.count("& $lease ") == 1
    assert "-Owner 'rl-triage-ppo-ratio-repair-v4'" in text
    assert "api_request_cap=4" in text


def test_prepare_payload_has_exact_source_bindings(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "PROTOCOL", protocol)
    monkeypatch.setattr(
        D,
        "dynamic",
        lambda _: {"source_sha256_lf": {"code": "a" * 64}, "evidence_sha256_lf": {}, "runtime_source_sha256_lf": {}},
    )
    assert D.prepare(Path("unused"))["status"] == "prepared_commit_required"
    assert json.loads(protocol.read_text(encoding="utf-8"))["seed"] == 2165
    with pytest.raises(FileExistsError):
        D.prepare(Path("unused"))

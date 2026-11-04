import json
from pathlib import Path

from evals import registered_recovery_resources_v1 as resources
from rl_triage.investigation_session import file_sha


def _write(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def test_measurement_starts_and_stops_sampler_and_preserves_partial_owned_costs(tmp_path: Path) -> None:
    events = []

    class Sampler:
        def __init__(self, output):
            self.output = Path(output)

        def start(self):
            events.append("start")
            return {"scope": "device_wide"}

        def stop(self):
            events.append("stop")
            return {"receipt_ref": _write(self.output / "stop.json", {"schema": "sampler-stop"})}

    admission = _write(tmp_path / "admission.json", {"schema": "admission"})
    output = tmp_path / "execution"
    measurement = resources.Measurement(tmp_path / "recovery", {"request_id": "r1", "effect_key": "e1"},
                                        admission, {"output_root": str(output)}, sampler_factory=Sampler)
    training = _write(output / "training/execution.json", {"wall_s": 10.0})
    checkpoint = _write(output / "training/checkpoint_execution.json", {"wall_s": 1.5})
    telemetry = _write(output / "training/extract_execution.json", {"wall_s": 0.8})
    ref = measurement.finish(raw=None, error=RuntimeError("fixed failed"))
    body = resources.verify(ref, request_id="r1", effect_key="e1")
    assert events == ["start", "stop"]
    assert body["status"] == "failed" and body["owned_gpu_s"] == 10.0
    assert body["simulator_sdk_launches"] == 1
    assert body["cpu_checkpoint_launches"] == 1 and body["total_owned_children"] == 3
    assert body["cpu_telemetry_launches"] == 1
    assert body["lease_await_s"] is None
    assert body["execution_refs"] == {"training": training, "fixed": None, "checkpoint": checkpoint,
                                     "telemetry": telemetry}


def test_measurement_preserves_sampler_close_failure_instead_of_claiming_success(tmp_path: Path) -> None:
    class Sampler:
        def __init__(self, output):
            self.output = Path(output)

        @staticmethod
        def start():
            return {"scope": "device_wide"}

        def stop(self):
            return {"receipt_ref": _write(self.output / "stop.json", {
                "schema": "periodic_device_sampling_stop_v1", "lease_matches": False})}

    admission = _write(tmp_path / "admission.json", {"schema": "admission"})
    measurement = resources.Measurement(tmp_path / "recovery", {"request_id": "r1", "effect_key": "e1"},
                                        admission, {"output_root": str(tmp_path / "execution")},
                                        sampler_factory=Sampler)
    body = resources.verify(measurement.finish(raw=None, error=None), request_id="r1", effect_key="e1")
    assert body["status"] == "failed" and body["sampler_ref"] is not None
    assert body["sampler_error_type"] == "LeaseMismatch"

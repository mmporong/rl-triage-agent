"""v1 input/receipt format with isolated bounded v2 NAT registrations.

This entry only registers hypotheses. It grants no inspection or GPU authority.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from threading import RLock

import yaml

from rl_triage import grounded_bounded_v2 as B
from rl_triage import triage_tools as T

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/triage_nemotron35_v2.yml"
FUNCTION_TYPES = {
    "grounded_evidence_group_index": {"_type": "triage_bounded_index_v2"},
    "grounded_evidence_group_page": {"_type": "triage_bounded_page_v2"},
    "grounded_evidence_details": {"_type": "triage_bounded_details_v2"},
    "grounded_active_registry": {"_type": "triage_bounded_registry_v2"},
    "write_grounded_active_assessment": {"_type": "triage_bounded_assessment_v2"},
}
_LOCK = RLock()


def entry():
    """Load a private module instance; leave the frozen v1 module untouched."""
    spec = importlib.util.spec_from_file_location("_nemotron35_entry_v2_base", Path(__file__).with_name("nemotron35_entry_v1.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("v1 entry source unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__["CONFIG"] = CONFIG
    old_config = module.check_config
    old_execute = module.execute
    old_prepare = module.prepare

    def check_config(path=CONFIG):
        from rl_triage import nat_grounded_bounded_v2  # noqa: F401

        report = old_config(path)
        document = yaml.safe_load(path.read_bytes())
        if document.get("functions") != FUNCTION_TYPES:
            report["problems"].append("bounded_function_types_changed")
        workflow = document.get("workflow", {})
        expected = {"max_iterations": 6, "max_history": 20, "handle_tool_errors": False,
                    "truncation_retry": {"max_retries": 0, "token_increment": 0}}
        if any(type(workflow.get(k)) is not type(v) or workflow.get(k) != v for k, v in expected.items()):
            report["problems"].append("bounded_workflow_limits_changed")
        if not module.WORKER.is_file():
            report["problems"].append("grounded_worker_not_packaged")
        report["ok"] = not report["problems"]
        report["path"] = "configs/triage_nemotron35_v2.yml"
        return report

    def prepare(run_dir, task_context, out, mode):
        if not check_config()["ok"]:
            raise ValueError("bounded v2 configuration or packaged worker invalid")
        return old_prepare(run_dir, task_context, out, mode)

    def execute(prepared):
        previous = T.WORKSPACE
        try:
            T.WORKSPACE = prepared["out"] / "workspace"
            session = B.BoundedSession(module.CASE_ID)
        finally:
            T.WORKSPACE = previous
        token = B.CURRENT_SESSION.set(session)
        try:
            return old_execute(prepared)
        finally:
            B.CURRENT_SESSION.reset(token)
            module._write_new(prepared["out"] / "bounded-session.json", session.receipt())

    module.__dict__.update(check_config=check_config, prepare=prepare, execute=execute)
    return module


def main(argv: list[str] | None = None) -> int:
    # Existing tools use a module-level workspace; do not run two workspaces in one process.
    with _LOCK:
        return entry().main(argv)


if __name__ == "__main__":
    sys.exit(main())

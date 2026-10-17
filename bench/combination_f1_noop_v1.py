"""설치 소스의 격리 함수에 F1 등록 교정을 적용하는 CPU NONE 대조."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType

import correction_adapter as A


def inspect(isaaclab: Path) -> dict:
    contract = A.source_contract(isaaclab)
    saved = {}
    try:
        for spec in A.TARGETS.values():
            source = isaaclab / spec["source"]
            module = ModuleType(spec["module"])
            module.__file__ = str(source.resolve())
            code = A._find_code(compile(source.read_bytes(), spec["source"], "exec", dont_inherit=True), spec["qualname"])
            function = FunctionType(code, {"__name__": spec["module"]}, code.co_name)
            function.__module__ = spec["module"]
            if "." in spec["qualname"]:
                owner, name = spec["qualname"].split(".")
                setattr(module, owner, type(owner, (), {name: function}))
            else:
                setattr(module, spec["qualname"], function)
            saved[spec["module"]] = sys.modules.get(spec["module"])
            sys.modules[spec["module"]] = module
        snapshot = A.capture(isaaclab, contract)
        readback = A.apply(snapshot, "restore_body_frame_reward")
        if (readback["before_fault"] != readback["after_fault"]
                or readback["after_fault"] != readback["after_correction"]
                or readback["changed_targets"] or readback["restored_targets"]):
            raise ValueError("NONE에 선택 교정을 적용했을 때 함수 지문이 바뀌었다")
        return {"schema": "combination_f1_none_noop_cpu_v1", "status": "passed",
            "scope": "설치 소스의 격리 함수 객체에 등록 교정 적용; GPU 환경·정책 평가는 하지 않음",
            "correction_id": "restore_body_frame_reward", "contract": contract,
            "before": readback["before_fault"], "after": readback["after_correction"],
            "changed_targets": readback["changed_targets"], "restored_targets": readback["restored_targets"],
            "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "gpu_executions": 0, "api_requests": 0}
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.isaaclab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "changed_targets": result["changed_targets"]}))


if __name__ == "__main__":
    main()

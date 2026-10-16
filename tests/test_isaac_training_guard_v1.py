from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import isaac_training_guard_v1 as G  # noqa: E402


def write_script(path: Path, main: str = "hooks.events.append('main')",
                 close: str = "hooks.events.append('close')", tail: str = "") -> None:
    path.write_text(
        "import guard_test_hooks as hooks\n"
        "class App:\n"
        "    def close(self):\n"
        f"        {close}\n"
        "simulation_app = App()\n"
        "def main():\n"
        f"    {main}\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "    simulation_app.close()\n"
        f"{tail}", encoding="utf-8", newline="\n")


def ready_callback(output: Path, manifest: str = "a" * 64, updates: int = 1):
    def callback(_app):
        G.durable_new(output / G.EVIDENCE,
                      {"manifest_sha256": manifest, "updates": updates, "message": "완료"})
        (output / G.RECORDS).write_text('{"update": 0}\n', encoding="utf-8", newline="\n")
        G.durable_new(output / G.CLOSE_READY, {"schema": G.CLOSE_SCHEMA,
            "manifest_sha256": manifest, "updates": updates,
            "evidence_sha256_lf": G._sha_lf(output / G.EVIDENCE),
            "records_sha256_lf": G._sha_lf(output / G.RECORDS)})
    return callback


@pytest.fixture
def hooks(monkeypatch):
    module = types.ModuleType("guard_test_hooks")
    setattr(module, "events", [])
    monkeypatch.setitem(sys.modules, "guard_test_hooks", module)
    return module


def test_durable_new_is_utf8_fsynced_and_exclusive(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(G.os, "fsync", lambda descriptor: calls.append(descriptor))
    path = tmp_path / "nested/result.json"
    G.durable_new(path, {"message": "학습 완료"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "학습 완료"}
    assert calls
    with pytest.raises(FileExistsError):
        G.durable_new(path, {"message": "overwrite"})


def test_record_failure_is_write_once_and_keeps_traceback(tmp_path):
    try:
        raise RuntimeError("첫 오류")
    except RuntimeError as error:
        first = G.record_failure(tmp_path, error, "training")
    try:
        raise ValueError("두 번째 오류")
    except ValueError as error:
        second = G.record_failure(tmp_path, error, "close")
    assert second == first
    assert first["status"] == "invalid" and first["type"] == "RuntimeError"
    assert first["message"] == "첫 오류" and "raise RuntimeError" in first["traceback"]
    assert first["stage"] == "training"


def test_run_upstream_executes_callback_then_real_close_without_source_edit(tmp_path, hooks):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    write_script(upstream)
    original = upstream.read_bytes()

    def callback(app):
        hooks.events.append("callback")
        assert type(app).__name__ == "App"
        ready_callback(output)(app)

    result = G.run_upstream(upstream, output, callback)
    assert result["status"] == "complete"
    assert hooks.events == ["main", "callback", "close"]
    assert upstream.read_bytes() == original and not (output / G.FAILURE).exists()
    G.validate_completion(output, "a" * 64)


def test_run_upstream_registers_isolated_actual_main_module_then_restores_it(tmp_path, hooks):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    write_script(
        upstream,
        main=("import __main__; "
              "assert __main__.__file__ == __file__; "
              "assert __main__.simulation_app is simulation_app; "
              "hooks.events.append('isolated-main')"),
    )
    previous_main = sys.modules["__main__"]
    G.run_upstream(upstream, output, ready_callback(output))
    assert hooks.events == ["isolated-main", "close"]
    assert sys.modules["__main__"] is previous_main


@pytest.mark.parametrize("deferred", [False, True])
def test_upstream_future_flags_are_preserved_without_guard_inheritance(tmp_path, hooks, deferred):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    expected = "'int'" if deferred else "int"
    write_script(upstream, main=f"assert main.__annotations__['argument'] == {expected}")
    source = upstream.read_text(encoding="utf-8").replace("def main():", "def main(argument: int = 0):")
    upstream.write_text(("from __future__ import annotations\n" if deferred else "") + source, encoding="utf-8")
    G.run_upstream(upstream, output, ready_callback(output))


def test_terminal_close_systemexit_zero_is_normal_only_after_ready_marker(tmp_path, hooks):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    write_script(upstream, close="hooks.events.append('close'); raise SystemExit(0)")
    result = G.run_upstream(upstream, output, ready_callback(output))
    assert result["close_completed"] is True and hooks.events == ["main", "close"]
    G.validate_completion(output, "a" * 64)


@pytest.mark.parametrize(("main", "callback_error", "error_type", "stage"), [
    ("raise SystemExit(0)", None, SystemExit, "upstream_script"),
    ("raise RuntimeError('script failed')", None, RuntimeError, "upstream_script"),
    ("hooks.events.append('main')", ValueError("callback failed"), ValueError, "before_close"),
])
def test_early_exit_script_and_callback_failures_record_then_reraise(
        tmp_path, hooks, capsys, main, callback_error, error_type, stage):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    write_script(upstream, main=main)

    def callback(app):
        if callback_error:
            raise callback_error
        ready_callback(output)(app)

    with pytest.raises(error_type):
        G.run_upstream(upstream, output, callback)
    failure = json.loads((output / G.FAILURE).read_text(encoding="utf-8"))
    assert failure["stage"] == stage and failure["type"] == error_type.__name__
    assert error_type.__name__ in capsys.readouterr().err
    assert "close" not in hooks.events


def test_missing_ready_marker_fails_before_close(tmp_path, hooks):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    write_script(upstream)
    with pytest.raises(RuntimeError, match="close marker"):
        G.run_upstream(upstream, output, lambda _app: None)
    assert hooks.events == ["main"]
    assert json.loads((output / G.FAILURE).read_text(encoding="utf-8"))["stage"] == "before_close"


@pytest.mark.parametrize("source", [
    "if __name__ == '__main__':\n    simulation_app.close()\nsimulation_app.close()\n",
    "if __name__ == '__main__':\n    simulation_app.close()\n    print('after')\n",
    "if __name__ == '__main__':\n    print('no close')\n",
])
def test_ast_gate_rejects_multiple_nonterminal_or_missing_close(tmp_path, source):
    upstream, output = tmp_path / "train.py", tmp_path / "out"
    upstream.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        G.run_upstream(upstream, output, lambda _app: None)
    assert json.loads((output / G.FAILURE).read_text(encoding="utf-8"))["stage"] == "upstream_ast"


def test_validate_completion_rejects_failure_missing_or_mismatched_evidence(tmp_path):
    manifest = "b" * 64
    ready_callback(tmp_path, manifest, 3)(object())
    G.validate_completion(tmp_path, manifest)

    marker_path = tmp_path / G.CLOSE_READY
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["updates"] = 2
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="결합"):
        G.validate_completion(tmp_path, manifest)

    marker["updates"] = 3
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    G.durable_new(tmp_path / G.FAILURE, {"status": "invalid"})
    with pytest.raises(ValueError, match="실패 기록"):
        G.validate_completion(tmp_path, manifest)


@pytest.mark.parametrize(("evidence_updates", "marker_updates"), [
    (0, 0), (-1, -1), (1, True), (True, 1),
])
def test_validate_completion_requires_positive_integer_updates(
        tmp_path, evidence_updates, marker_updates):
    manifest = "c" * 64
    G.durable_new(tmp_path / G.EVIDENCE,
                  {"manifest_sha256": manifest, "updates": evidence_updates})
    (tmp_path / G.RECORDS).write_text("{}\n", encoding="utf-8", newline="\n")
    G.durable_new(tmp_path / G.CLOSE_READY, {
        "schema": G.CLOSE_SCHEMA,
        "manifest_sha256": manifest,
        "updates": marker_updates,
        "evidence_sha256_lf": G._sha_lf(tmp_path / G.EVIDENCE),
        "records_sha256_lf": G._sha_lf(tmp_path / G.RECORDS),
    })
    with pytest.raises(ValueError, match="결합"):
        G.validate_completion(tmp_path, manifest)


def test_subprocess_early_exit_becomes_nonzero_and_hard_close_runs_after_marker(tmp_path):
    guard_path = ROOT / "bench/isaac_training_guard_v1.py"
    child = r'''import json,os,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import isaac_training_guard_v1 as G
upstream,output,mode=Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]
def before(app):
 G.durable_new(output/G.EVIDENCE,{'manifest_sha256':'a'*64,'updates':1})
 (output/G.RECORDS).write_text('{}\n',encoding='utf-8')
 G.durable_new(output/G.CLOSE_READY,{'schema':G.CLOSE_SCHEMA,'manifest_sha256':'a'*64,'updates':1,
  'evidence_sha256_lf':G._sha_lf(output/G.EVIDENCE),'records_sha256_lf':G._sha_lf(output/G.RECORDS)})
try:
 G.run_upstream(upstream,output,before)
except BaseException:
 os._exit(1)
'''
    early, early_out = tmp_path / "early.py", tmp_path / "early-out"
    early.write_text("raise SystemExit(0)\nif __name__ == '__main__':\n    simulation_app.close()\n",
                     encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", child, str(guard_path.parent), str(early),
                             str(early_out), "early"], cwd=ROOT, encoding="utf-8", errors="replace")
    assert result.returncode == 1 and (early_out / G.FAILURE).is_file()

    hard, hard_out = tmp_path / "hard.py", tmp_path / "hard-out"
    hard.write_text("import os\nfrom pathlib import Path\nclass App:\n"
                    "    def close(self):\n        Path(__file__).with_suffix('.closed').write_text('closed')\n"
                    "        os._exit(0)\nsimulation_app=App()\ndef main(): pass\n"
                    "if __name__ == '__main__':\n    main()\n    simulation_app.close()\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", child, str(guard_path.parent), str(hard),
                             str(hard_out), "hard"], cwd=ROOT, encoding="utf-8", errors="replace")
    assert result.returncode == 0 and hard.with_suffix(".closed").is_file()
    G.validate_completion(hard_out, "a" * 64)


def test_installed_simulation_app_close_reaches_shutdown_under_isaac_python(tmp_path):
    isaac_root = Path.home() / "IsaacLab" / "_isaac_sim"
    python = isaac_root / "kit/python/python.exe"
    source = isaac_root / "exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
    if not python.is_file() or not source.is_file():
        pytest.skip("이 호스트에 고정 Isaac Sim Python/source가 없다")
    marker, output = tmp_path / "shutdown.json", tmp_path / "out"
    upstream = tmp_path / "train.py"
    upstream.write_text(
        "import actual_close_runtime as runtime\n"
        "simulation_app = runtime.app\n"
        "def main(): pass\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "    simulation_app.close()\n",
        encoding="utf-8", newline="\n")
    script = r'''import ast,json,os,sys,types
from pathlib import Path
source,marker,guard_parent,upstream,output=map(Path,sys.argv[1:])
sys.path.insert(0,str(guard_parent)); import isaac_training_guard_v1 as G
tree=ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimulationApp')
close=next(n for n in klass.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='close')
module=ast.fix_missing_locations(ast.Module(body=[close],type_ignores=[]))
class Orchestrator:
 def get_status(self): return 'STOPPED'
 def stop(self): pass
 def wait_until_complete(self): pass
 def set_capture_on_play(self,value): pass
rep=types.ModuleType('omni.replicator.core'); rep.orchestrator=Orchestrator(); rep.orchestrator.Status=types.SimpleNamespace(STOPPED='STOPPED',STOPPING='STOPPING')
omni=types.ModuleType('omni'); omni.__path__=[]; replicator=types.ModuleType('omni.replicator'); replicator.__path__=[]; replicator.core=rep
omni.replicator=replicator; omni.usd=types.SimpleNamespace(get_context=lambda:types.SimpleNamespace(can_close_stage=lambda:False))
sys.modules.update({'omni':omni,'omni.replicator':replicator,'omni.replicator.core':rep})
utils=types.ModuleType('fakepkg.utils'); utils.is_stage_loading=lambda:False
pkg=types.ModuleType('fakepkg'); pkg.__path__=[]; sys.modules.update({'fakepkg':pkg,'fakepkg.utils':utils})
class App:
 def print_and_log(self,*args): pass
 def update(self): pass
 def shutdown(self): marker.write_text(json.dumps({'closed':True,'python':list(sys.version_info[:2])})); os._exit(0)
class Self:
 config={}; _exiting=False; _app=App(); _framework=types.SimpleNamespace(unload_all_plugins=lambda:None)
carb=types.SimpleNamespace(profiler=types.SimpleNamespace(acquire_profiler_interface=lambda **kw:None),logging=types.SimpleNamespace(acquire_logging=lambda:types.SimpleNamespace(set_log_enabled=lambda x:None)))
scope={'__package__':'fakepkg','time':__import__('time'),'omni':omni,'carb':carb}
exec(compile(module,str(source),'exec'),scope); Self.close=scope['close']
runtime=types.ModuleType('actual_close_runtime'); runtime.app=Self(); sys.modules['actual_close_runtime']=runtime
def before(_app):
 G.durable_new(output/G.EVIDENCE,{'manifest_sha256':'a'*64,'updates':1})
 (output/G.RECORDS).write_text('{}\n',encoding='utf-8')
 G.durable_new(output/G.CLOSE_READY,{'schema':G.CLOSE_SCHEMA,'manifest_sha256':'a'*64,'updates':1,
  'evidence_sha256_lf':G._sha_lf(output/G.EVIDENCE),'records_sha256_lf':G._sha_lf(output/G.RECORDS)})
G.run_upstream(upstream,output,before)
raise RuntimeError('shutdown did not exit')
'''
    result = subprocess.run([
        str(python), "-c", script, str(source), str(marker), str(ROOT / "bench"),
        str(upstream), str(output),
    ], cwd=ROOT, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert value == {"closed": True, "python": [3, 10]}
    G.validate_completion(output, "a" * 64)

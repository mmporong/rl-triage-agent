from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "bench" / "vision_brev_environment_v1.py"
SPEC = importlib.util.spec_from_file_location("vision_brev_environment_v1", MODULE_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _bundle_sha(bundle: Path) -> str:
    return hashlib.sha256((bundle / "bundle-manifest.json").read_bytes()).hexdigest()


def test_bundle_is_write_once_and_binds_only_explicit_cache_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "wheel.whl").write_bytes(b"wheel")
    (cache / "credential.json").write_text("secret", encoding="utf-8")
    output = tmp_path / "bundle"
    manifest = M.build_bundle(output, [{"logical_id": "wheels", "root": cache, "files": ["wheel.whl"]}])
    assert M.verify_bundle(output) == manifest
    cache_manifest = json.loads((output / "cache-manifest.json").read_text(encoding="utf-8"))
    assert cache_manifest["inputs"][0]["members"][0]["path"] == "wheel.whl"
    assert "credential" not in json.dumps(cache_manifest)
    assert manifest["generation_side_effects"] == {
        "gpu_query": False,
        "install": False,
        "network": False,
        "torch_import": False,
    }
    assert b"\r" not in (output / "install.sh").read_bytes()
    with pytest.raises(M.ContractError, match="새 경로"):
        M.build_bundle(output)


def test_cache_manifest_rejects_traversal_links_duplicates_and_unlisted_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "one").write_bytes(b"1")
    with pytest.raises(M.ContractError, match="안전하지 않은"):
        M.cache_manifest([{"logical_id": "x", "root": root, "files": ["../one"]}])
    with pytest.raises(M.ContractError, match="중복 캐시"):
        M.cache_manifest([{"logical_id": "x", "root": root, "files": ["one", "one"]}])
    hard = root / "hard"
    try:
        os.link(root / "one", hard)
    except OSError:
        pytest.fail("하드링크 회귀 fixture 생성 실패")
    with pytest.raises(M.ContractError, match="하드링크"):
        M.cache_manifest([{"logical_id": "x", "root": root, "files": ["hard"]}])
    hard.unlink()
    link = root / "link"
    link.write_bytes(b"link")
    original_lstat = Path.lstat

    def fake_lstat(path: Path):  # type: ignore[no-untyped-def]
        result = original_lstat(path)
        if path == link:
            values = list(result)
            values[0] = 0o120777
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(M.ContractError, match="일반 파일"):
        M.cache_manifest([{"logical_id": "x", "root": root, "files": ["link"]}])


def test_cache_hashing_streams_instead_of_reading_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    cache_file = root / "large.whl"
    cache_file.write_bytes(b"x" * (2 * 1024 * 1024 + 17))
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path == cache_file:
            raise AssertionError("cache read_bytes 호출")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = M.cache_manifest([{"logical_id": "wheel", "root": root, "files": ["large.whl"]}])
    assert result["inputs"][0]["members"][0]["bytes"] == 2 * 1024 * 1024 + 17


def test_bundle_verification_rejects_tamper_and_extra_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    (bundle / "pins.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(M.ContractError, match="검증 실패"):
        M.verify_bundle(bundle)
    other = tmp_path / "other"
    M.build_bundle(other)
    (other / "extra").write_bytes(b"x")
    with pytest.raises(M.ContractError, match="파일 집합"):
        M.verify_bundle(other)
    missing_row = tmp_path / "missing-row"
    M.build_bundle(missing_row)
    manifest_path = missing_row / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(M.ContractError, match="파일 행 수"):
        M.verify_bundle(missing_row)


def test_generated_verify_is_read_only_and_marks_gpu_gates_unverified(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    task_root = tmp_path / "workspace" / "rl-triage-vision-v1"
    task_root.mkdir(parents=True)
    completed = subprocess.run(
        [sys.executable, str(bundle / "verify_environment.py"), "--task-root", str(task_root)],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    assert completed.returncode == 3
    result = json.loads(completed.stdout)
    assert result["status"] == "mismatch"
    assert result["reuse_allowed"] is False
    assert result["os_image_digest"] is None
    assert set(result["gpu_gates"].values()) == {"unverified"}
    assert "torch" not in sys.modules


def test_install_recipe_keeps_proven_constraints_activation_and_reports(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    recipe = (bundle / "install_runtime.py").read_text(encoding="utf-8")
    assert '"--seed"' in recipe
    assert '"OMNI_KIT_ACCEPT_EULA"' in recipe
    assert '"--extra-index-url","https://pypi.nvidia.com"' in recipe
    assert 'os.environ["VIRTUAL_ENV"]' in recipe
    assert 'os.environ["PIP_CONSTRAINT"]' in recipe
    assert 'os.environ["PIP_BUILD_CONSTRAINT"]' in recipe
    assert '"--report",str(attempt/"torch-install.json")' in recipe
    assert '"--report",str(attempt/"isaacsim-install.json")' in recipe
    assert '"-m","pip","check"' in recipe
    assert '"-m","pip","freeze","--all"' in recipe
    assert 'check_stdout(verifier,expected_stdout)' in recipe


@pytest.mark.parametrize(
    "stdout, accepted",
    [
        ("uv 0.13.5\n", True),
        ("uv 0.13.5 (x86_64-unknown-linux-gnu)\n", True),
        ("uv 0.13.50\n", False),
        ("uv 0.13.4 (x86_64-unknown-linux-gnu)\n", False),
        ("uv 0.13.5 arbitrary\n", False),
    ],
)
def test_generated_uv_version_contract_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, accepted: bool
) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    spec = importlib.util.spec_from_file_location(
        f"generated_install_runtime_{hashlib.sha256(stdout.encode()).hexdigest()}",
        bundle / "install_runtime.py",
    )
    assert spec and spec.loader
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: stdout)
    assert runtime.uv_version_ok(["uv", "--version"]) is accepted


@pytest.mark.parametrize(
    "mutation",
    ["verifier", "installer", "constraints", "manifest", "self_consistent"],
)
def test_installer_rejects_bundle_tamper_before_task_or_subprocess(
    tmp_path: Path, mutation: str
) -> None:
    bundle = tmp_path / mutation
    M.build_bundle(bundle)
    expected_sha = _bundle_sha(bundle)
    if mutation == "manifest":
        path = bundle / "bundle-manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        target = {
            "verifier": bundle / "verify_environment.py",
            "installer": bundle / "install_runtime.py",
            "constraints": bundle / "constraints.txt",
            "self_consistent": bundle / "verify_environment.py",
        }[mutation]
        target.write_bytes(target.read_bytes() + b"\n# harmless mutation\n")
        if mutation == "self_consistent":
            manifest_path = bundle / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = target.read_bytes()
            for row in manifest["files"]:
                if row["path"] == target.name:
                    row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), sha256_lf=hashlib.sha256(data).hexdigest())
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    home = tmp_path / f"home-{mutation}"
    task = home / "workspace" / "rl-triage-vision-v1"
    env = os.environ.copy()
    env.update({"HOME": str(home), "USERPROFILE": str(home), "VISION_TASK_ROOT": str(task)})
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle / "install_runtime.py"),
            "--expected-manifest-sha256",
            expected_sha,
        ],
        env=env,
        check=False,
    )
    assert completed.returncode == 28
    assert not task.exists()


def _bash() -> str:
    preferred = Path(r"C:\Program Files\Git\bin\bash.exe")
    found = str(preferred) if preferred.exists() else shutil.which("bash")
    assert found is not None, "bash가 필요합니다"
    return found


def _git_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _posix_path(path: Path) -> str:
    if os.name != "nt":
        return str(path.resolve())
    return subprocess.check_output(
        ["wsl", "wslpath", "-a", "-u", str(path).replace("\\", "/")],
        text=True, encoding="utf-8", errors="replace",
    ).strip()


def _run_posix_driver(
    driver: Path, *args: str, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    command = ["bash", _posix_path(driver), *args]
    if os.name == "nt":
        command.insert(0, "wsl")
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_install_script_syntax_and_wrapper_exit_are_preserved(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    subprocess.run([_bash(), "-n", str(bundle / "install.sh")], check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\nexit 47\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_python.chmod(0o755)
    command = (
        f'PATH="{_git_path(fake_bin)}:$PATH" "{_git_path(bundle / "install.sh")}" '
        f'--expected-manifest-sha256 {_bundle_sha(bundle)}'
    )
    completed = subprocess.run([_bash(), "-lc", command], check=False)
    assert completed.returncode == 47


def test_unmarked_existing_task_root_is_rejected_before_install(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    home = tmp_path / "home"
    task = home / "workspace" / "rl-triage-vision-v1"
    task.mkdir(parents=True)
    (task / "foreign").write_bytes(b"x")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle / "install_runtime.py"),
            "--expected-manifest-sha256",
            _bundle_sha(bundle),
        ],
        env=env,
        check=False,
    )
    assert completed.returncode == 23
    assert not (task / ".rl-triage-vision-owner.json").exists()


def test_install_runtime_failure_resume_and_exact_reuse_with_fake_tools(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    driver = tmp_path / "driver.sh"
    driver.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail
bundle="$1"
bundle_sha="$2"
base="$(mktemp -d)"
export HOME="$base"
root="$HOME/workspace/rl-triage-vision-v1"
mkdir -p "$(dirname "$root")" "$HOME/fake-bin"
fake_bin="$HOME/fake-bin"
template="$HOME/fake-python"
cat > "$template" <<'PY'
#!/usr/bin/env bash
set -eu
root="${VISION_TASK_ROOT:?}"
printf '%s\n' "$*" >> "$root/python-calls.log"
if [[ "$*" == *verify_environment.py* ]]; then
  if [[ -f "$HOME/force-precheck" ]]; then
    rm "$HOME/force-precheck"
    echo '{"status":"mismatch","reuse_allowed":false}'
    exit 3
  fi
  if [[ -f "$root/.steps/isaaclab_packages.json" ]]; then
    echo '{"status":"match","reuse_allowed":true}'
    exit 0
  fi
  echo '{"status":"mismatch","reuse_allowed":false}'
  exit 3
fi
if [[ "$*" == *"-m pip install pip==26.2.1"* && -f "$HOME/fail-once" ]]; then
  rm "$HOME/fail-once"
  exit 47
fi
if [[ "$*" == *"-m pip install pip==26.2.1"* ]]; then touch "$root/fake-pip-installed"; exit 0; fi
if [[ "$*" == *"m.version('pip')"* && ! -f "$root/fake-pip-installed" ]]; then exit 1; fi
exit 0
PY
chmod +x "$template"
cat > "$fake_bin/uv" <<'UV'
#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == --version ]]; then echo 'uv 0.13.5'; exit 0; fi
target="${@: -1}"
mkdir -p "$target/bin"
cp "$FAKE_PY_TEMPLATE" "$target/bin/python"
chmod +x "$target/bin/python"
UV
cat > "$fake_bin/git" <<'GIT'
#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == clone ]]; then
  target="${@: -1}"
  mkdir -p "$target/.git"
  cat > "$target/isaaclab.sh" <<'LAB'
#!/usr/bin/env bash
exit 0
LAB
  chmod +x "$target/isaaclab.sh"
  exit 0
fi
if [[ "$*" == *"rev-parse HEAD"* ]]; then
  echo '90b79bb2d44feb8d833f260f2bf37da3487180ba'
  exit 0
fi
if [[ "$*" == *"remote get-url origin"* ]]; then echo 'https://github.com/isaac-sim/IsaacLab.git'; exit 0; fi
if [[ "$*" == *"status --porcelain"* ]]; then exit 0; fi
exit 2
GIT
chmod +x "$fake_bin/uv" "$fake_bin/git"
export VISION_TASK_ROOT="$root" FAKE_PY_TEMPLATE="$template" PATH="$fake_bin:$PATH"
echo x > "$HOME/fail-once"
set +e
"$bundle/install.sh" --expected-manifest-sha256 "$bundle_sha"
first_rc=$?
set -e
[[ "$first_rc" == 47 ]]
grep -q '"exit_code":47' "$root"/reports/install-attempt-*/result.json
"$bundle/install.sh" --expected-manifest-sha256 "$bundle_sha"
before="$(grep -c -- '-m pip install' "$root/python-calls.log")"
"$bundle/install.sh" --expected-manifest-sha256 "$bundle_sha"
after="$(grep -c -- '-m pip install' "$root/python-calls.log")"
[[ "$before" == "$after" ]]
grep -q '"stage":"reused"' "$root"/reports/install-attempt-*/result.json
rm "$root/.steps/pip.json"
echo x > "$HOME/force-precheck"
before="$(grep -c -- '-m pip install' "$root/python-calls.log")"
"$bundle/install.sh" --expected-manifest-sha256 "$bundle_sha"
after="$(grep -c -- '-m pip install' "$root/python-calls.log")"
[[ "$before" == "$after" ]]
test -f "$root/.steps/pip.json"
''',
        encoding="utf-8",
        newline="\n",
    )
    _run_posix_driver(driver, _posix_path(bundle), _bundle_sha(bundle))


def test_existing_environment_inspection_writes_mismatch_receipt_without_markers(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    existing = tmp_path / "existing"
    subprocess.run([sys.executable, "-m", "venv", str(existing / "venv")], check=True)
    receipt = tmp_path / "reuse.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle / "install_runtime.py"),
            "--expected-manifest-sha256",
            _bundle_sha(bundle),
            "--existing-environment-root",
            str(existing),
            "--reuse-receipt",
            str(receipt),
        ],
        check=False,
    )
    assert completed.returncode == 3
    record = json.loads(receipt.read_text(encoding="utf-8"))
    assert record["status"] == "mismatch"
    assert record["inspection_only"] is True
    assert "other_wheels" in record["unverified"]
    assert not (existing / ".rl-triage-vision-owner.json").exists()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                str(bundle / "install_runtime.py"),
                "--expected-manifest-sha256",
                _bundle_sha(bundle),
                "--existing-environment-root",
                str(existing),
                "--reuse-receipt",
                str(existing / "receipt.json"),
            ],
            check=True,
        )


def test_corrupt_owned_uv_cache_fails_without_download_or_venv(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    M.build_bundle(bundle)
    driver = tmp_path / "uv-cache-driver.sh"
    driver.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail
bundle="$1"; bundle_sha="$2"; base="$(mktemp -d)"; export HOME="$base"
root="$HOME/workspace/rl-triage-vision-v1"; mkdir -p "$root/tools" "$HOME/fake-bin"
printf '%s\n' '{"owner":"rl-triage-vision-v1","schema":"vision_brev_environment_owner_v1"}' > "$root/.rl-triage-vision-owner.json"
printf 'corrupt' > "$root/tools/uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
cat > "$HOME/fake-bin/uv" <<'UV'
#!/usr/bin/env bash
echo 'uv 9.9.9'
UV
chmod +x "$HOME/fake-bin/uv"
export PATH="$HOME/fake-bin:/usr/bin:/bin" VISION_TASK_ROOT="$root"
test_python="${VISION_TEST_PYTHON:-python3}"
set +e
"$test_python" "$bundle/install_runtime.py" --expected-manifest-sha256 "$bundle_sha"
rc=$?
set -e
[[ "$rc" == 1 ]]
test ! -e "$root/venv"
grep -q '"stage":"uv_bootstrap"' "$root"/reports/install-attempt-*/result.json
grep -q 'cached uv wheel mismatch' "$root"/reports/install-attempt-*/error.json
''',
        encoding="utf-8",
        newline="\n",
    )
    _run_posix_driver(driver, _posix_path(bundle), _bundle_sha(bundle))


@pytest.mark.parametrize(
    "mode, expected_error",
    [("partial", "partial IsaacLab checkout"), ("exact_dirty", "IsaacLab checkout dirty")],
)
def test_partial_or_exact_dirty_isaaclab_fails_before_package_install(
    tmp_path: Path, mode: str, expected_error: str
) -> None:
    bundle = tmp_path / f"bundle-{mode}"
    M.build_bundle(bundle)
    driver = tmp_path / f"lab-{mode}.sh"
    driver.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail
bundle="$1"; bundle_sha="$2"; mode="$3"; base="$(mktemp -d)"; export HOME="$base"
root="$HOME/workspace/rl-triage-vision-v1"; fake="$HOME/fake-bin"; mkdir -p "$root/IsaacLab" "$fake"
printf '%s\n' '{"owner":"rl-triage-vision-v1","schema":"vision_brev_environment_owner_v1"}' > "$root/.rl-triage-vision-owner.json"
if [[ "$mode" == exact_dirty ]]; then mkdir -p "$root/IsaacLab/.git"; fi
cat > "$HOME/fake-python" <<'PY'
#!/usr/bin/env bash
if [[ "$*" == *verify_environment.py* ]]; then echo '{"status":"mismatch"}'; exit 3; fi
exit 0
PY
cat > "$fake/uv" <<'UV'
#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == --version ]]; then echo 'uv 0.13.5'; exit 0; fi
target="${@: -1}"; mkdir -p "$target/bin"; cp "$FAKE_PY_TEMPLATE" "$target/bin/python"; chmod +x "$target/bin/python"
UV
cat > "$fake/git" <<'GIT'
#!/usr/bin/env bash
set -eu
root="${VISION_TASK_ROOT:?}"
printf '%s\n' "$*" >> "$root/git-calls.log"
if [[ "$*" == *"rev-parse HEAD"* ]]; then echo '90b79bb2d44feb8d833f260f2bf37da3487180ba'; exit 0; fi
if [[ "$*" == *"remote get-url origin"* ]]; then echo 'https://github.com/isaac-sim/IsaacLab.git'; exit 0; fi
if [[ "$*" == *"status --porcelain"* ]]; then echo ' M source/file.py'; exit 0; fi
if [[ "$*" == *"fetch"* || "$*" == *"checkout"* ]]; then touch "$root/unsafe-mutation"; exit 0; fi
exit 2
GIT
chmod +x "$HOME/fake-python" "$fake/uv" "$fake/git"
export PATH="$fake:/usr/bin:/bin" VISION_TASK_ROOT="$root" FAKE_PY_TEMPLATE="$HOME/fake-python"
test_python="${VISION_TEST_PYTHON:-python3}"
set +e
"$test_python" "$bundle/install_runtime.py" --expected-manifest-sha256 "$bundle_sha"
rc=$?
set -e
[[ "$rc" == 1 ]]
test ! -e "$root/unsafe-mutation"
test -d "$root/IsaacLab"
test ! -e "$root/venv"
if [[ "$mode" == partial ]]; then
  grep -q 'partial IsaacLab checkout' "$root"/reports/install-attempt-*/error.json
else
  grep -q 'IsaacLab checkout dirty' "$root"/reports/install-attempt-*/error.json
  ! grep -Eq 'fetch|checkout --detach' "$root/git-calls.log"
fi
''',
        encoding="utf-8",
        newline="\n",
    )
    completed = _run_posix_driver(
        driver, _posix_path(bundle), _bundle_sha(bundle), mode, capture_output=True
    )
    assert completed.returncode == 0
    assert expected_error in (bundle / "install_runtime.py").read_text(encoding="utf-8")


def test_clean_known_origin_wrong_head_recovers_before_install(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-recover"
    M.build_bundle(bundle)
    driver = tmp_path / "lab-recover.sh"
    driver.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail
bundle="$1"; bundle_sha="$2"; base="$(mktemp -d)"; export HOME="$base"
root="$HOME/workspace/rl-triage-vision-v1"; fake="$HOME/fake-bin"; mkdir -p "$root/IsaacLab/.git" "$fake"
printf '%s\n' '{"owner":"rl-triage-vision-v1","schema":"vision_brev_environment_owner_v1"}' > "$root/.rl-triage-vision-owner.json"
cat > "$root/IsaacLab/isaaclab.sh" <<'LAB'
#!/usr/bin/env bash
exit 0
LAB
cat > "$HOME/fake-python" <<'PY'
#!/usr/bin/env bash
root="${VISION_TASK_ROOT:?}"
printf '%s\n' "$*" >> "$root/python-calls.log"
if [[ "$*" == *verify_environment.py* ]]; then
  if [[ -f "$root/.steps/isaaclab_packages.json" ]]; then echo '{"status":"match"}'; exit 0; fi
  echo '{"status":"mismatch"}'; exit 3
fi
exit 0
PY
cat > "$fake/uv" <<'UV'
#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == --version ]]; then echo 'uv 0.13.5'; exit 0; fi
target="${@: -1}"; mkdir -p "$target/bin"; cp "$FAKE_PY_TEMPLATE" "$target/bin/python"; chmod +x "$target/bin/python"
UV
cat > "$fake/git" <<'GIT'
#!/usr/bin/env bash
set -eu
root="${VISION_TASK_ROOT:?}"; printf '%s\n' "$*" >> "$root/git-calls.log"
if [[ "$*" == *"remote get-url origin"* ]]; then echo 'https://github.com/isaac-sim/IsaacLab.git'; exit 0; fi
if [[ "$*" == *"status --porcelain"* ]]; then exit 0; fi
if [[ "$*" == *"rev-parse HEAD"* ]]; then
  if [[ -f "$root/recovered-head" ]]; then echo '90b79bb2d44feb8d833f260f2bf37da3487180ba'; else echo deadbeef; fi
  exit 0
fi
if [[ "$*" == *"fetch origin 90b79bb2d44feb8d833f260f2bf37da3487180ba"* ]]; then touch "$root/fetched"; exit 0; fi
if [[ "$*" == *"checkout --detach 90b79bb2d44feb8d833f260f2bf37da3487180ba"* ]]; then
  test -f "$root/fetched"; touch "$root/recovered-head"; exit 0
fi
exit 2
GIT
chmod +x "$root/IsaacLab/isaaclab.sh" "$HOME/fake-python" "$fake/uv" "$fake/git"
export PATH="$fake:/usr/bin:/bin" VISION_TASK_ROOT="$root" FAKE_PY_TEMPLATE="$HOME/fake-python"
test_python="${VISION_TEST_PYTHON:-python3}"
"$test_python" "$bundle/install_runtime.py" --expected-manifest-sha256 "$bundle_sha"
test -f "$root/fetched"; test -f "$root/recovered-head"
fetch_line="$(grep -n 'fetch origin' "$root/git-calls.log" | cut -d: -f1)"
checkout_line="$(grep -n 'checkout --detach' "$root/git-calls.log" | cut -d: -f1)"
[[ "$fetch_line" -lt "$checkout_line" ]]
grep -q '"status":"complete"' "$root"/reports/install-attempt-*/result.json
''',
        encoding="utf-8",
        newline="\n",
    )
    _run_posix_driver(driver, _posix_path(bundle), _bundle_sha(bundle))

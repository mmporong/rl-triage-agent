from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "vision_brev_environment_bundle_v1"
LAB_SHA = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
PINS: dict[str, Any] = {
    "schema": "vision_brev_environment_pins_v1",
    "python": "3.10.12",
    "pip": "26.2.1",
    "uv_bootstrap": "0.13.5",
    "uv_wheel_filename": "uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "uv_wheel_sha256": "fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946",
    "uv_wheel_url": "https://files.pythonhosted.org/packages/3a/ad/3c2a81d66d1839063662e5b6c0002ec31f8ee987dbc5295747627ca6160a/uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "isaacsim": "4.5.0.0",
    "isaaclab_commit": LAB_SHA,
    "isaaclab": "0.41.3",
    "isaaclab_rl": "0.1.8",
    "rsl_rl_lib": "2.3.3",
    "gymnasium": "1.2.0",
    "os_image_digest": None,
    "gpu_gates": {
        "cuda_driver": "unverified",
        "isaac_launch": "unverified",
        "one_frame": "unverified",
    },
}
GENERATED = (
    "pins.json",
    "constraints.txt",
    "build-constraints.txt",
    "cache-manifest.json",
    "verify_environment.py",
    "install_runtime.py",
    "install.sh",
)


class ContractError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ContractError(f"비유한 JSON 수치: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"중복 JSON 키: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError("캐시 경로는 비어 있지 않은 POSIX 상대 경로여야 합니다")
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ContractError(f"안전하지 않은 캐시 경로: {value!r}")
    return rel.as_posix()


def _regular_owned_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"캐시 파일 누락: {path.name}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"일반 파일만 허용합니다: {path.name}")
    if info.st_nlink != 1:
        raise ContractError(f"하드링크 캐시 파일은 허용하지 않습니다: {path.name}")
    return info


def _stream_digest(path: Path) -> tuple[int, str]:
    before = _regular_owned_file(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise ContractError("캐시 파일이 열기 전에 바뀌었습니다")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
        closed = os.fstat(handle.fileno())
    after = _regular_owned_file(path)
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink
    if identity(before) != identity(opened) or identity(opened) != identity(closed) or identity(closed) != identity(after):
        raise ContractError("캐시 파일이 해시 계산 중 바뀌었습니다")
    return size, digest.hexdigest()


def cache_manifest(cache_inputs: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    logical_seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for source in cache_inputs:
        if set(source) != {"logical_id", "root", "files"}:
            raise ContractError("캐시 입력 키는 logical_id, root, files로 고정됩니다")
        logical_id = source["logical_id"]
        root_raw = source["root"]
        files = source["files"]
        if not isinstance(logical_id, str) or not logical_id or logical_id in logical_seen:
            raise ContractError("logical_id는 고유한 문자열이어야 합니다")
        logical_seen.add(logical_id)
        if not isinstance(root_raw, (str, os.PathLike)) or not isinstance(files, list) or not files:
            raise ContractError("root와 하나 이상의 명시 파일이 필요합니다")
        root = Path(root_raw)
        if root.is_symlink() or not root.is_dir():
            raise ContractError("캐시 root는 symlink가 아닌 폴더여야 합니다")
        member_seen: set[str] = set()
        members: list[dict[str, Any]] = []
        for item in files:
            rel = _safe_relative(item)
            if rel in member_seen:
                raise ContractError(f"중복 캐시 경로: {rel}")
            member_seen.add(rel)
            candidate = root.joinpath(*PurePosixPath(rel).parts)
            cursor = root
            for part in PurePosixPath(rel).parts[:-1]:
                cursor /= part
                if cursor.is_symlink():
                    raise ContractError(f"symlink 캐시 경로는 허용하지 않습니다: {rel}")
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root.resolve(strict=True))
            except ValueError as exc:
                raise ContractError("캐시 경로가 root 밖을 가리킵니다") from exc
            size, digest = _stream_digest(candidate)
            members.append({"path": rel, "bytes": size, "sha256": digest})
        rows.append({"logical_id": logical_id, "members": sorted(members, key=lambda row: row["path"])})
    return {"schema": "vision_brev_cache_manifest_v1", "inputs": sorted(rows, key=lambda row: row["logical_id"])}


def _constraints() -> bytes:
    return b"gymnasium==1.2.0\nrsl-rl-lib==2.3.3\ntorch==2.7.0\ntorchvision==0.22.0\n"


def _build_constraints() -> bytes:
    return b"setuptools<81\n"


def _verify_script() -> bytes:
    text = f'''#!/usr/bin/env python3
import argparse, importlib.metadata as md, json, platform, subprocess, sys
from pathlib import Path

EXPECTED = {json.dumps(PINS, sort_keys=True)!r}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-root", required=True)
    ns = p.parse_args()
    expected = json.loads(EXPECTED)
    actual = {{"python": platform.python_version()}}
    for key, package in (("pip","pip"),("torch","torch"),("torchvision","torchvision"),
                         ("isaacsim","isaacsim"),("isaaclab","isaaclab"),
                         ("isaaclab_rl","isaaclab_rl"),("rsl_rl_lib","rsl-rl-lib"),
                         ("gymnasium","gymnasium")):
        try: actual[key] = md.version(package)
        except md.PackageNotFoundError: actual[key] = None
    lab = Path(ns.task_root) / "IsaacLab"
    try:
        actual["isaaclab_commit"] = subprocess.run(
            ["git", "-C", str(lab), "rev-parse", "HEAD"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
        actual["isaaclab_tracked_clean"] = subprocess.run(
            ["git", "-C", str(lab), "status", "--porcelain", "--untracked-files=all"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout == ""
        actual["isaaclab_origin"] = subprocess.run(
            ["git", "-C", str(lab), "remote", "get-url", "origin"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        actual["isaaclab_commit"] = None
        actual["isaaclab_tracked_clean"] = None
        actual["isaaclab_origin"] = None
    try:
        actual["pip_check"] = subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).returncode == 0
    except OSError:
        actual["pip_check"] = False
    keys = ("python","pip","torch","torchvision","isaacsim","isaaclab",
            "isaaclab_rl","rsl_rl_lib","gymnasium","isaaclab_commit")
    mismatches = [k for k in keys if actual.get(k) != expected[k]]
    if actual.get("isaaclab_tracked_clean") is not True: mismatches.append("isaaclab_tracked_clean")
    if actual.get("isaaclab_origin") != "https://github.com/isaac-sim/IsaacLab.git": mismatches.append("isaaclab_origin")
    if actual.get("pip_check") is not True: mismatches.append("pip_check")
    result = {{"schema":"vision_brev_environment_verification_v1",
              "status":"match" if not mismatches else "mismatch",
              "reuse_allowed":not mismatches,"actual":actual,"expected":{{k:expected[k] for k in keys}},
              "os_image_digest":None,"gpu_gates":expected["gpu_gates"],"mismatches":mismatches}}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if not mismatches else 3)
if __name__ == "__main__": main()
'''
    return text.encode()


def _install_runtime() -> bytes:
    text = f'''#!/usr/bin/env python3
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

LAB_SHA = "{LAB_SHA}"
OWNER = {{"schema":"vision_brev_environment_owner_v1","owner":"rl-triage-vision-v1"}}
EXPECTED_FILES = {json.dumps(list(GENERATED), sort_keys=True)!r}
UV_FILENAME = "{PINS['uv_wheel_filename']}"
UV_SHA256 = "{PINS['uv_wheel_sha256']}"
UV_URL = "{PINS['uv_wheel_url']}"
UV_VERSION_OUTPUTS = {{"uv 0.13.5", "uv 0.13.5 (x86_64-unknown-linux-gnu)"}}

def reject_constant(value):
    raise ValueError("invalid JSON constant: "+value)

def unique_pairs(pairs):
    result={{}}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate JSON key: "+key)
        result[key]=value
    return result

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=unique_pairs,parse_constant=reject_constant)

def regular_file(path):
    info=path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_nlink!=1: raise ValueError("bundle regular-file policy")
    return info

def verify_bundle(bundle, expected_sha):
    if len(expected_sha)!=64 or any(c not in "0123456789abcdef" for c in expected_sha): raise ValueError("expected bundle SHA format")
    manifest_path=bundle/"bundle-manifest.json"
    regular_file(manifest_path)
    manifest_bytes=manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=expected_sha: raise ValueError("bundle manifest SHA mismatch")
    manifest=load_json(manifest_path)
    expected=set(json.loads(EXPECTED_FILES)); all_expected=expected|{{"bundle-manifest.json"}}
    if {{item.name for item in bundle.iterdir()}}!=all_expected: raise ValueError("bundle file set mismatch")
    if set(manifest)!={{"schema","files","recipe_sha256_lf","pins_sha256","cache_manifest_sha256","generation_side_effects"}}: raise ValueError("bundle manifest keys")
    if manifest["schema"]!="{SCHEMA}" or not isinstance(manifest["files"],list) or len(manifest["files"])!=len(expected): raise ValueError("bundle manifest rows")
    seen=set()
    for row in manifest["files"]:
        if not isinstance(row,dict) or set(row)!={{"path","bytes","sha256","sha256_lf"}}: raise ValueError("bundle row shape")
        name=row["path"]
        if name not in expected or name in seen: raise ValueError("bundle row path")
        seen.add(name); path=bundle/name; info=regular_file(path); data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if b"\\r" in data or info.st_size!=row["bytes"] or digest!=row["sha256"] or digest!=row["sha256_lf"]: raise ValueError("bundle row bytes")
    if seen!=expected: raise ValueError("bundle row completeness")
    if manifest["pins_sha256"]!=hashlib.sha256((bundle/"pins.json").read_bytes()).hexdigest(): raise ValueError("pins reference")
    if manifest["cache_manifest_sha256"]!=hashlib.sha256((bundle/"cache-manifest.json").read_bytes()).hexdigest(): raise ValueError("cache reference")
    if manifest["recipe_sha256_lf"]!=hashlib.sha256((bundle/"install.sh").read_bytes()).hexdigest(): raise ValueError("recipe reference")
    pins=load_json(bundle/"pins.json"); cache=load_json(bundle/"cache-manifest.json")
    if pins.get("schema")!="vision_brev_environment_pins_v1" or cache.get("schema")!="vision_brev_cache_manifest_v1": raise ValueError("pins/cache schema")

def write_json(path, value, exclusive=False):
    data=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    mode="xb" if exclusive else "wb"
    with path.open(mode) as handle: handle.write(data)

def run(command, **kw):
    return subprocess.run(command, check=True, **kw)

def check(command):
    try: run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return True
    except (OSError, subprocess.CalledProcessError): return False

def check_stdout(command, expected):
    try: return subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL).strip()==expected
    except (OSError, subprocess.CalledProcessError): return False

def uv_version_ok(command):
    try: output=subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return False
    return output in UV_VERSION_OUTPUTS

def sha_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def recover_owned_link(path, pattern):
    if not path.exists() or path.stat().st_nlink==1: return
    matches=[item for item in path.parent.glob(pattern) if item.is_file() and item.stat().st_ino==path.stat().st_ino and item.stat().st_dev==path.stat().st_dev]
    if path.stat().st_nlink!=2 or len(matches)!=1: raise RuntimeError("owned cache hardlink mismatch")
    matches[0].unlink()

def write_atomic_json(path, value):
    data=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n").encode()
    partial=path.parent/(path.name+".partial-"+str(os.getpid()))
    with partial.open("xb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
    try: os.link(partial,path); partial.unlink()
    except FileExistsError: raise RuntimeError("owned cache receipt race")

def prepare_owned_uv(root):
    tools=root/"tools"; tools.mkdir(exist_ok=True)
    wheel=tools/UV_FILENAME; binary=tools/"uv"; receipt=tools/"uv-binary.json"
    recover_owned_link(wheel,UV_FILENAME+".partial-*")
    if wheel.exists():
        if not wheel.is_file() or wheel.is_symlink() or wheel.stat().st_nlink!=1 or sha_file(wheel)!=UV_SHA256: raise RuntimeError("cached uv wheel mismatch")
    else:
        partial=tools/(UV_FILENAME+".partial-"+str(os.getpid()))
        digest=hashlib.sha256()
        with urllib.request.urlopen(UV_URL,timeout=120) as response, partial.open("xb") as handle:
            for chunk in iter(lambda:response.read(1024*1024),b""):
                digest.update(chunk); handle.write(chunk)
            handle.flush(); os.fsync(handle.fileno())
        if digest.hexdigest()!=UV_SHA256: raise RuntimeError("downloaded uv wheel mismatch")
        try: os.link(partial,wheel); partial.unlink()
        except FileExistsError: raise RuntimeError("uv wheel cache race")
    with zipfile.ZipFile(wheel) as archive:
        names=[name for name in archive.namelist() if not name.endswith("/") and name.rsplit("/",1)[-1]=="uv"]
        if len(names)!=1: raise RuntimeError("uv wheel binary entry mismatch")
        data=archive.read(names[0])
    if data[:4]!=b"\\x7fELF": raise RuntimeError("uv binary format mismatch")
    binary_sha=hashlib.sha256(data).hexdigest()
    recover_owned_link(binary,"uv.partial-*")
    if binary.exists():
        if sha_file(binary)!=binary_sha: raise RuntimeError("uv binary hash mismatch")
        expected_proof={{"schema":"vision_brev_uv_binary_v1","wheel_sha256":UV_SHA256,"binary_sha256":binary_sha}}
        if receipt.exists():
            if load_json(receipt)!=expected_proof: raise RuntimeError("uv binary receipt mismatch")
        else: write_atomic_json(receipt,expected_proof)
    else:
        partial=tools/("uv.partial-"+str(os.getpid()))
        with partial.open("xb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        partial.chmod(0o755)
        try: os.link(partial,binary); partial.unlink()
        except FileExistsError: raise RuntimeError("uv binary cache race")
        write_atomic_json(receipt,{{"schema":"vision_brev_uv_binary_v1","wheel_sha256":UV_SHA256,"binary_sha256":binary_sha}})
    if not uv_version_ok([str(binary),"--version"]): raise RuntimeError("uv binary version mismatch")
    return str(binary)

def environment_python(root):
    posix=root/"venv"/"bin"/"python"
    windows=root/"venv"/"Scripts"/"python.exe"
    return windows if windows.exists() and not posix.exists() else posix

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-manifest-sha256",required=True)
    parser.add_argument("--existing-environment-root")
    parser.add_argument("--reuse-receipt")
    args=parser.parse_args()
    bundle=Path(__file__).resolve().parent
    try:
        verify_bundle(bundle,args.expected_manifest_sha256)
    except (OSError,ValueError,json.JSONDecodeError):
        return 28
    home=Path.home().resolve()
    os.environ["OMNI_KIT_ACCEPT_EULA"]="YES"
    if args.existing_environment_root:
        if not args.reuse_receipt: raise SystemExit(25)
        existing=Path(args.existing_environment_root).resolve(strict=True)
        receipt=Path(args.reuse_receipt).resolve()
        if receipt.exists() or existing == receipt or existing in receipt.parents: raise SystemExit(26)
        vpy=environment_python(existing)
        try:
            completed=subprocess.run([str(vpy),str(bundle/"verify_environment.py"),"--task-root",str(existing)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            existing_rc=completed.returncode
            try: verification=json.loads(completed.stdout)
            except json.JSONDecodeError: verification={{"status":"invalid_output"}}
        except OSError as exc:
            existing_rc=27
            verification={{"status":"missing_interpreter","error_type":type(exc).__name__}}
        write_json(receipt,{{"schema":"vision_brev_existing_environment_reuse_v1",
            "status":"reusable" if existing_rc==0 else "mismatch",
            "environment_root_sha256":hashlib.sha256(str(existing).encode()).hexdigest(),
            "verification":verification,"inspection_only":True,
            "fixed_scope":["python","pip","torch","torchvision","isaacsim","isaaclab","isaaclab_rl","rsl_rl_lib","gymnasium","isaaclab_commit","isaaclab_tracked_clean","isaaclab_origin","pip_check"],
            "unverified":["os_image_digest","cuda_driver","other_wheels","isaac_launch","one_frame"]}},exclusive=True)
        return existing_rc
    if args.reuse_receipt: raise SystemExit(25)
    root=Path(os.environ.get("VISION_TASK_ROOT", str(home/"workspace"/"rl-triage-vision-v1"))).resolve()
    workspace=(home/"workspace").resolve()
    if root == workspace or workspace not in root.parents: raise SystemExit(22)
    owner_path=root/".rl-triage-vision-owner.json"
    if root.exists() and not owner_path.exists() and any(root.iterdir()): raise SystemExit(23)
    root.mkdir(parents=True,exist_ok=True)
    if owner_path.exists():
        if json.loads(owner_path.read_text()) != OWNER: raise SystemExit(24)
    else: write_json(owner_path, OWNER, exclusive=True)
    reports=root/"reports"; reports.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt=reports/("install-attempt-"+stamp); attempt.mkdir()
    write_json(attempt/"attempt.json",{{"schema":"vision_brev_install_attempt_v1","status":"running"}},exclusive=True)
    rc=1; stage="precheck"
    try:
        vpy=environment_python(root)
        verify=[str(vpy),str(bundle/"verify_environment.py"),"--task-root",str(root)]
        if vpy.exists() and check(verify):
            stage="reused"; rc=0
            write_json(attempt/"metadata.json",json.loads(subprocess.check_output(verify,text=True)),exclusive=True)
            return 0
        lab=root/"IsaacLab"; origin="https://github.com/isaac-sim/IsaacLab.git"
        if lab.exists():
            stage="isaaclab_precheck"
            if not (lab/".git").is_dir(): raise RuntimeError("partial IsaacLab checkout")
            if not check_stdout(["git","-C",str(lab),"remote","get-url","origin"],origin): raise RuntimeError("IsaacLab origin mismatch")
            if not check_stdout(["git","-C",str(lab),"status","--porcelain","--untracked-files=all"],""): raise RuntimeError("IsaacLab checkout dirty")
            if not check_stdout(["git","-C",str(lab),"rev-parse","HEAD"],LAB_SHA):
                run(["git","-C",str(lab),"fetch","origin",LAB_SHA])
                run(["git","-C",str(lab),"checkout","--detach",LAB_SHA])
                if not check_stdout(["git","-C",str(lab),"rev-parse","HEAD"],LAB_SHA): raise RuntimeError("IsaacLab checkout recovery mismatch")
        manifest_sha=hashlib.sha256((bundle/"bundle-manifest.json").read_bytes()).hexdigest()
        steps=root/".steps"; steps.mkdir(exist_ok=True)
        def do(name, command, verifier, expected_stdout=None):
            nonlocal stage
            stage=name; marker=steps/(name+".json")
            verified=lambda: check_stdout(verifier,expected_stdout) if expected_stdout is not None else check(verifier)
            if marker.exists():
                try: marked=json.loads(marker.read_text())
                except json.JSONDecodeError: marked={{}}
                if marked=={{"schema":"vision_brev_install_step_v1","step":name,"bundle_manifest_sha256":manifest_sha}} and verified(): return
            if verified():
                write_json(marker,{{"schema":"vision_brev_install_step_v1","step":name,"bundle_manifest_sha256":manifest_sha}})
                return
            run(command)
            if not verified(): raise RuntimeError("단계 검증 실패: "+name)
            write_json(marker,{{"schema":"vision_brev_install_step_v1","step":name,"bundle_manifest_sha256":manifest_sha}})
        uv=shutil.which("uv")
        if uv is None or not uv_version_ok([uv,"--version"]):
            stage="uv_bootstrap"
            uv=prepare_owned_uv(root)
        do("venv",[uv,"venv","--python","3.10.12","--seed",str(root/"venv")],[str(vpy),"-c","import platform;raise SystemExit(platform.python_version()!='3.10.12')"])
        os.environ["VIRTUAL_ENV"]=str(root/"venv")
        os.environ["PATH"]=str(vpy.parent)+os.pathsep+os.environ.get("PATH","")
        do("pip",[str(vpy),"-m","pip","install","pip==26.2.1"],[str(vpy),"-c","import importlib.metadata as m;raise SystemExit(m.version('pip')!='26.2.1')"])
        do("torch",[str(vpy),"-m","pip","install","--report",str(attempt/"torch-install.json"),"-c",str(bundle/"constraints.txt"),"--index-url","https://download.pytorch.org/whl/cu128","torch==2.7.0","torchvision==0.22.0"],[str(vpy),"-c","import importlib.metadata as m;raise SystemExit(m.version('torch')!='2.7.0+cu128' or m.version('torchvision')!='0.22.0+cu128')"])
        os.environ["PIP_BUILD_CONSTRAINT"]=str(bundle/"build-constraints.txt")
        do("isaacsim",[str(vpy),"-m","pip","install","--report",str(attempt/"isaacsim-install.json"),"-c",str(bundle/"constraints.txt"),"isaacsim[all,extscache]==4.5.0","--extra-index-url","https://pypi.nvidia.com"],[str(vpy),"-c","import importlib.metadata as m;raise SystemExit(m.version('isaacsim')!='4.5.0.0')"])
        if not lab.exists(): run(["git","clone","--branch","v2.1.1","--depth","1",origin,str(lab)])
        if not (lab/".git").is_dir(): raise RuntimeError("partial IsaacLab checkout")
        do("isaaclab_source",["git","-C",str(lab),"rev-parse","HEAD"],["git","-C",str(lab),"rev-parse","HEAD"],LAB_SHA)
        os.environ["PIP_CONSTRAINT"]=str(bundle/"constraints.txt")
        os.environ["PIP_BUILD_CONSTRAINT"]=str(bundle/"build-constraints.txt")
        do("isaaclab_packages",[str(lab/"isaaclab.sh"),"-i","rsl_rl"],[str(vpy),"-c","import importlib.metadata as m;raise SystemExit(m.version('isaaclab')!='0.41.3' or m.version('isaaclab_rl')!='0.1.8' or m.version('rsl-rl-lib')!='2.3.3' or m.version('gymnasium')!='1.2.0')"])
        stage="pip_check"; run([str(vpy),"-m","pip","check"])
        stage="freeze"; freeze=run([str(vpy),"-m","pip","freeze","--all"],stdout=subprocess.PIPE,text=True)
        (attempt/"freeze.txt").write_text(freeze.stdout,encoding="utf-8",newline="\\n")
        stage="final_verify"
        completed=run(verify,stdout=subprocess.PIPE,text=True)
        write_json(attempt/"metadata.json",json.loads(completed.stdout),exclusive=True)
        rc=0; return 0
    except subprocess.CalledProcessError as exc:
        rc=exc.returncode or 1; return rc
    except Exception as exc:
        write_json(attempt/"error.json",{{"type":type(exc).__name__,"message":str(exc)}},exclusive=True)
        rc=1; return rc
    finally:
        write_json(attempt/"result.json",{{"schema":"vision_brev_install_result_v1","status":"complete" if rc==0 else "failed","exit_code":rc,"stage":stage}},exclusive=True)
if __name__ == "__main__": raise SystemExit(main())
'''
    return text.encode()


def _install_sh() -> bytes:
    return b'''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$bundle_dir/install_runtime.py" "$@"
'''


def _exclusive(path: Path, data: bytes, executable: bool = False) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    if executable:
        path.chmod(0o755)


def build_bundle(output: Path, cache_inputs: Sequence[Mapping[str, object]] = ()) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise ContractError("bundle 출력은 새 경로여야 합니다")
    cache = cache_manifest(cache_inputs)
    files = {
        "pins.json": _canonical(PINS),
        "constraints.txt": _constraints(),
        "build-constraints.txt": _build_constraints(),
        "cache-manifest.json": _canonical(cache),
        "verify_environment.py": _verify_script(),
        "install_runtime.py": _install_runtime(),
        "install.sh": _install_sh(),
    }
    if any(b"\r" in data for data in files.values()):
        raise ContractError("생성 recipe는 LF만 사용해야 합니다")
    output.mkdir(parents=False)
    for name, data in files.items():
        _exclusive(output / name, data, name.endswith((".py", ".sh")))
    entries = [{"path": name, "bytes": len(data), "sha256": _sha(data), "sha256_lf": _sha(data)} for name, data in sorted(files.items())]
    manifest = {
        "schema": SCHEMA,
        "files": entries,
        "recipe_sha256_lf": next(row["sha256_lf"] for row in entries if row["path"] == "install.sh"),
        "pins_sha256": _sha(files["pins.json"]),
        "cache_manifest_sha256": _sha(files["cache-manifest.json"]),
        "generation_side_effects": {"install": False, "network": False, "gpu_query": False, "torch_import": False},
    }
    _exclusive(output / "bundle-manifest.json", _canonical(manifest))
    return manifest


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = Path(bundle)
    _regular_owned_file(bundle / "bundle-manifest.json")
    manifest = _load_json(bundle / "bundle-manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ContractError("bundle manifest schema 불일치")
    expected = set(GENERATED) | {"bundle-manifest.json"}
    actual = {item.name for item in bundle.iterdir()}
    if actual != expected:
        raise ContractError("bundle 파일 집합 불일치")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != len(GENERATED):
        raise ContractError("bundle manifest 파일 행 수 불일치")
    row_paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(row_paths) != len(GENERATED) or set(row_paths) != set(GENERATED):
        raise ContractError("bundle manifest 파일 경로 불일치")
    for row in rows:
        if set(row) != {"path", "bytes", "sha256", "sha256_lf"}:
            raise ContractError("bundle 파일 행 형식 불일치")
        name = _safe_relative(row["path"])
        path = bundle / name
        info = _regular_owned_file(path)
        data = path.read_bytes()
        if b"\r" in data or info.st_size != row["bytes"] or _sha(data) != row["sha256"] or _sha(data) != row["sha256_lf"]:
            raise ContractError(f"bundle 파일 검증 실패: {name}")
    if _load_json(bundle / "pins.json") != PINS:
        raise ContractError("pins 불일치")
    if _sha((bundle / "pins.json").read_bytes()) != manifest.get("pins_sha256"):
        raise ContractError("pins SHA 결합 불일치")
    recipe = bundle / "install.sh"
    if _sha(recipe.read_bytes()) != manifest.get("recipe_sha256_lf"):
        raise ContractError("recipe SHA 결합 불일치")
    cache = _load_json(bundle / "cache-manifest.json")
    if _sha(_canonical(cache)) != manifest["cache_manifest_sha256"]:
        raise ContractError("cache manifest 결합 불일치")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache-spec", type=Path)
    verify = sub.add_parser("verify-bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        inputs: Sequence[Mapping[str, object]] = ()
        if args.cache_spec is not None:
            raw = _load_json(args.cache_spec)
            if not isinstance(raw, list):
                raise ContractError("cache spec은 배열이어야 합니다")
            inputs = raw
        result = build_bundle(args.output, inputs)
    else:
        result = verify_bundle(args.bundle)
    bundle = args.output if args.command == "build" else args.bundle
    payload = {
        "manifest": result,
        "bundle_reference": {
            "path": "bundle-manifest.json",
            "sha256": _sha((bundle / "bundle-manifest.json").read_bytes()),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

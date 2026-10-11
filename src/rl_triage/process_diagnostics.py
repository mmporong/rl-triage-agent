"""프로세스 출력에서 비밀 값 없이 실행 상태와 등록된 WSL 오류만 보존한다."""
from __future__ import annotations

import hashlib
import re


# 일반 문구, 경로, 예외 메시지를 로그에 복사하지 않는다.
# 오류 코드 추가는 코드 리뷰를 거치며 임의의 출력 토큰은 수집하지 않는다.
WSL_ERROR_CODES = (
    "Wsl/Service/0x80072747",
    "Wsl/Service/0x8007000e",
    "Wsl/Service/0x8007019e",
    "Wsl/Service/0x80370102",
)
_WSL_TOKEN = re.compile(r"(?<![A-Za-z0-9_./+-])Wsl/Service/[A-Za-z0-9_./+-]+")


def _stream(value: bytes | str | None) -> dict:
    if value is None:
        return {"available": False, "bytes": None, "sha256": None, "wsl_error_codes": []}
    raw = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
    if not isinstance(raw, bytes):
        raise TypeError("process output는 bytes, str 또는 None이어야 합니다")
    # WSL은 Windows에서 BOM 없는 UTF-16LE로 오류를 반환하기도 한다.
    views = (raw.decode("utf-8", errors="replace"),
             raw.decode("utf-16-le", errors="replace"),
             raw.decode("utf-16-be", errors="replace"))
    observed = {match.group() for view in views for match in _WSL_TOKEN.finditer(view)}
    return {"available": True, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "wsl_error_codes": [code for code in WSL_ERROR_CODES if code in observed]}


def launch_diagnostics(returncode: int | None, stdout: bytes | str | None = None,
                       stderr: bytes | str | None = None, *, timed_out: bool = False) -> dict:
    """raw 출력·명령·환경변수 없이 관측된 프로세스 종료와 출력 해시를 반환한다.

    str 출력은 transport가 이미 decode한 텍스트의 UTF-8 해시다. bytes 출력은
    프로세스가 반환한 원본 바이트 해시이며, 두 해시를 동일하게 해석하면 안 된다.
    """
    if returncode is not None and type(returncode) is not int:
        raise TypeError("returncode는 int 또는 None이어야 합니다")
    if type(timed_out) is not bool:
        raise TypeError("timed_out은 bool이어야 합니다")
    return {
        "schema": "process_launch_diagnostics_v1", "returncode": returncode,
        "timed_out": timed_out,
        "stdout": {**_stream(stdout), "hash_scope": "decoded_text_utf8" if isinstance(stdout, str) else "raw_bytes"},
        "stderr": {**_stream(stderr), "hash_scope": "decoded_text_utf8" if isinstance(stderr, str) else "raw_bytes"},
        "raw_output_retained": False,
    }

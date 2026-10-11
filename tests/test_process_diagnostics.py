"""실제 launcher 실패의 코드 보존과 비밀 출력 차단 경계를 검증한다."""
import hashlib
import json

import pytest

from rl_triage.process_diagnostics import launch_diagnostics


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_wsl_code_is_recovered_without_retaining_surrounding_secret(encoding):
    raw = "token=private-value C:/private/path\nWsl/Service/0x80072747\n".encode(encoding)
    result = launch_diagnostics(1, stdout=raw, stderr=b"")
    assert result["stdout"]["wsl_error_codes"] == ["Wsl/Service/0x80072747"]
    assert result["stdout"]["bytes"] == len(raw)
    assert result["stdout"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["returncode"] == 1 and result["timed_out"] is False
    assert "private" not in json.dumps(result)


def test_missing_stream_differs_from_empty_output_and_timeout_has_no_exit_code():
    result = launch_diagnostics(None, stdout=None, stderr=b"", timed_out=True)
    assert result["stdout"]["available"] is False and result["stdout"]["bytes"] is None
    assert result["stderr"]["available"] is True and result["stderr"]["bytes"] == 0
    assert result["returncode"] is None and result["timed_out"] is True


def test_arbitrary_provider_errors_and_commands_are_not_logged():
    result = launch_diagnostics(2, stderr="Bearer secret-token Wsl/Service/0xDEADBEEF")
    assert result["stderr"]["wsl_error_codes"] == []
    assert result["stderr"]["hash_scope"] == "decoded_text_utf8"
    assert "Bearer" not in json.dumps(result) and "secret-token" not in json.dumps(result)


@pytest.mark.parametrize("text", [
    "Wsl/Service/0x800727470", "prefixWsl/Service/0x80072747suffix",
    "Wsl/Service/0x80072747suffix", "Wsl/Service/0x80072747/other",
])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_nearby_tokens_are_not_attributed_to_an_allowlisted_code(text, encoding):
    assert launch_diagnostics(1, text.encode(encoding))["stdout"]["wsl_error_codes"] == []


def test_stdout_and_stderr_codes_remain_distinct():
    result = launch_diagnostics(-15, b"", b"Wsl/Service/0x8007019e")
    assert result["stdout"]["wsl_error_codes"] == []
    assert result["stderr"]["wsl_error_codes"] == ["Wsl/Service/0x8007019e"]
    assert result["returncode"] == -15


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_malformed_return_code_is_rejected(value):
    with pytest.raises(TypeError):
        launch_diagnostics(value)

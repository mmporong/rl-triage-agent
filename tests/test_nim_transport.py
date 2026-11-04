import json
from urllib.error import HTTPError, URLError

import pytest

from rl_triage import nim_transport as N

MESSAGES = [{"role": "system", "content": "Select a registered probe."},
            {"role": "user", "content": "{}"}]


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"model": "returned-alias", "choices": [{"message": {"content": "{}"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                                   "total_tokens": 12}}).encode()


def test_exact_endpoint_and_usage_without_key_in_returned_record():
    calls = []
    def opener(request, **kwargs):
        calls.append((request, kwargs))
        return Response()
    result = N.complete_local(MESSAGES, N.PARAMS, environ={"NVIDIA_API_KEY": "sentinel-secret"}, opener=opener)
    assert len(calls) == 1
    assert calls[0][0].full_url == N.ENDPOINT
    assert calls[0][0].get_header("Authorization") == "Bearer sentinel-secret"
    assert calls[0][1]["timeout"] == 300
    assert result["usage"]["total_tokens"] == 12
    assert result["actual_billing_usd"] is None
    assert "sentinel-secret" not in json.dumps(result)


@pytest.mark.parametrize("error", [HTTPError(N.ENDPOINT, 503, "sentinel-secret", None, None),
                                   URLError("sentinel-secret"), TimeoutError("sentinel-secret")])
def test_provider_failure_is_single_attempt_and_error_payload_is_not_leaked(error):
    calls = []
    def opener(*_args, **_kwargs):
        calls.append(1)
        raise error
    result = N.complete_local(MESSAGES, N.PARAMS, environ={"NVIDIA_API_KEY": "sentinel-secret"}, opener=opener)
    assert result["status"] == "error"
    assert result["automatic_retries"] == 0
    assert len(calls) == 1
    assert "sentinel-secret" not in json.dumps(result)


def test_missing_credential_and_unregistered_request_do_not_send():
    def forbidden(*_args, **_kwargs):
        pytest.fail("network not allowed")
    result = N.complete_local(MESSAGES, N.PARAMS, environ={}, opener=forbidden)
    assert result["error_type"] == "MissingCredential"
    with pytest.raises(ValueError):
        N.complete_local(MESSAGES, {**N.PARAMS, "model": "other"}, environ={}, opener=forbidden)
    with pytest.raises(ValueError):
        N.complete_local([{"role": "user", "content": [{"type": "image_url"}]}],
                         N.PARAMS, environ={}, opener=forbidden)


def test_windows_mount_mapping_is_lexical_and_refuses_unc():
    from pathlib import Path
    assert N._mounted(Path("C:/Users/name with space/request.json")) == "/mnt/c/Users/name with space/request.json"
    with pytest.raises(ValueError):
        N._mounted(Path(r"\\server\share\request.json"))

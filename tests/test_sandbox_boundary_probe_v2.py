import errno
import socket
import urllib.error

from evals.probe_sandbox_boundary_v2 import error_errno, observe


def test_nested_url_permission_denial_is_recognised():
    assert error_errno(urllib.error.URLError(PermissionError(errno.EACCES, "denied"))) == errno.EACCES


def test_connection_refusal_is_not_counted_as_policy_denial():
    def operation():
        raise ConnectionRefusedError(errno.ECONNREFUSED, "refused")
    assert observe(operation)["policy_eacces"] is False


def test_timeout_and_dns_failure_are_not_policy_denial():
    for error in [TimeoutError("timeout"), socket.gaierror(-2, "not known")]:
        def operation(exc=error):
            raise exc
        assert observe(operation)["policy_eacces"] is False


def test_successful_control_is_not_marked_denied():
    assert observe(lambda: None) == {"outcome": "allowed", "errno": None, "policy_eacces": False}

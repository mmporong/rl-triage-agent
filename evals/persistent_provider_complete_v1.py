"""등록된 RI97 provider endpoint만 호출하는 completion helper."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench import persistent_brev_bridge_v1 as BRIDGE  # noqa: E402

CONFIG_ENV = "RI97_PROVIDER_CLIENT_CONFIG"
CONFIG_SHA_ENV = "RI97_PROVIDER_CLIENT_CONFIG_SHA256"


def bridge_complete(messages: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    path_value, expected = os.environ.get(CONFIG_ENV), os.environ.get(CONFIG_SHA_ENV)
    if not path_value or not expected:
        raise BRIDGE.BridgeError("provider bridge config not bound")
    path = Path(path_value).resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise BRIDGE.BridgeError("provider bridge config changed")
    config = json.loads(raw)
    return BRIDGE.ProviderClient(config).complete(messages, params)


@contextmanager
def config_scope(path: Path) -> Iterator[None]:
    target = Path(path).resolve(strict=True)
    previous_path, previous_sha = os.environ.get(CONFIG_ENV), os.environ.get(CONFIG_SHA_ENV)
    os.environ[CONFIG_ENV], os.environ[CONFIG_SHA_ENV] = str(target), hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop(CONFIG_ENV, None)
        else:
            os.environ[CONFIG_ENV] = previous_path
        if previous_sha is None:
            os.environ.pop(CONFIG_SHA_ENV, None)
        else:
            os.environ[CONFIG_SHA_ENV] = previous_sha

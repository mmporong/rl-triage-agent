"""등록 frame의 무결성 검증 뒤 이미지 품질만 격리하는 CPU evidence adapter."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image, UnidentifiedImageError  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_investigation_evidence_v1 as EVIDENCE
import agentic_investigation_runtime_v1 as BASE
import agentic_runtime_input_guard_v1 as INPUT

SCHEMA = "agentic_frame_quality_guard_v1"
POLICY_SCHEMA = "agentic_frame_quality_policy_v1"
VISUAL_SCHEMA = "agentic_frame_quality_visual_context_v1"
METRIC = "luma_u8_4_neighbor_laplacian_population_variance"
UNIT = "intensity_squared"
REQUIRED_SOURCES = {
    "evals/agentic_frame_quality_guard_v1.py",
    "tests/test_agentic_frame_quality_guard_v1.py",
}
_ADAPTER_LOCK = threading.RLock()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read_ref(ref: dict[str, Any]) -> tuple[dict[str, Any], str]:
    _require(set(ref) == {"path", "sha256"} and isinstance(ref.get("path"), str)
             and isinstance(ref.get("sha256"), str), "quality policy ref")
    path = Path(cast(str, ref["path"]))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == ref["sha256"], "quality policy SHA changed")
    value = json.loads(raw)
    _require(isinstance(value, dict)
             and set(value) == {"schema", "metric", "unit", "min_laplacian_variance"}
             and value.get("schema") == POLICY_SCHEMA and value.get("metric") == METRIC
             and value.get("unit") == UNIT
             and type(value.get("min_laplacian_variance")) in {int, float}
             and math.isfinite(float(value["min_laplacian_variance"]))
             and float(value["min_laplacian_variance"]) >= 0.0, "quality policy contract")
    return cast(dict[str, Any], value), digest


def _image_bytes(content: list[dict[str, Any]]) -> bytes:
    _require(len(content) == 2 and content[0].get("type") == "text"
             and content[1].get("type") == "image_url", "registered frame content")
    image = content[1].get("image_url")
    _require(isinstance(image, dict) and isinstance(image.get("url"), str), "frame data URL")
    checked_image = cast(Mapping[str, Any], image)
    prefix, separator, encoded = cast(str, checked_image["url"]).partition(",")
    _require(separator == "," and prefix.startswith("data:image/") and ";base64" in prefix,
             "frame data URL encoding")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("frame data URL base64") from error


def laplacian_variance(raw: bytes) -> float:
    """luma U8 내부 화소의 (4C-N-S-W-E) population variance [intensity²]."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            luma = np.asarray(image.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("frame image decode unavailable") from error
    if luma.ndim != 2 or min(luma.shape) < 3:
        raise ValueError("frame image decode unavailable")
    values = luma.astype(np.float64)
    laplacian = (4.0 * values[1:-1, 1:-1] - values[:-2, 1:-1] - values[2:, 1:-1]
                 - values[1:-1, :-2] - values[1:-1, 2:])
    return float(np.var(laplacian, ddof=0))


class QualityEvidenceAdapter(EVIDENCE.EvidenceAdapter):
    """기존 SHA/계보 검증을 유지하고 frame alias의 이미지 블록만 격리한다."""

    def __init__(self, index: dict, roots: dict[str, Path], registration: dict, *,
                 expected_registration_sha256: str, policy_ref: dict[str, Any]):
        self.policy_ref = deepcopy(policy_ref)
        self.policy, self.policy_sha256 = _read_ref(self.policy_ref)
        super().__init__(index, roots, registration,
                         expected_registration_sha256=expected_registration_sha256)

    def _quality(self, alias: str) -> list[dict[str, Any]]:
        registered = self.registration["aliases"].get(alias)
        _require(isinstance(registered, dict) and registered.get("kind") == "frames",
                 "quality alias must be registered frames")
        receipts = []
        for frame in registered["frame_ids"]:
            view = EVIDENCE.E.frame_view(
                self.index, registered["evidence_id"], frame, self.roots,
                expected_sha256=self.registration["index_sha256"])
            content = view["messages"][0]["content"]
            raw = _image_bytes(content)
            metric: dict[str, Any] | None
            try:
                value = laplacian_variance(raw)
                metric = {"name": METRIC, "value": value, "unit": UNIT}
                outcome = ("available" if value >= float(self.policy["min_laplacian_variance"])
                           else "quality_below_registered_floor")
            except ValueError as error:
                if str(error) != "frame image decode unavailable":
                    raise
                metric, outcome = None, "quarantined_decode_unavailable"
            receipts.append({
                "schema": SCHEMA + "_receipt", "policy_sha256": self.policy_sha256,
                "registration_sha256": self.registration_sha256,
                "index_sha256": self.registration["index_sha256"],
                "frame_content_sha256": hashlib.sha256(raw).hexdigest(),
                "metric": metric, "outcome": outcome,
            })
        return receipts

    def _catalogue(self, loop: Any, cache: dict) -> dict:
        catalogue = super()._catalogue(loop, cache)
        for alias, kind in catalogue.items():
            if kind == "frames":
                self._quality(alias)
        return catalogue

    def _unwrap(self, visual_context: dict | None) -> tuple[dict | None, dict[str, list[dict]]]:
        if visual_context is None:
            return None, {}
        _require(set(visual_context) == {"schema", "policy_sha256", "registration_sha256",
                                        "index_sha256", "base", "quality"}
                 and visual_context.get("schema") == VISUAL_SCHEMA
                 and visual_context.get("policy_sha256") == self.policy_sha256
                 and visual_context.get("registration_sha256") == self.registration_sha256
                 and visual_context.get("index_sha256") == self.registration["index_sha256"]
                 and isinstance(visual_context.get("quality"), dict),
                 "quality visual context binding")
        quality = cast(dict[str, list[dict]], visual_context["quality"])
        for alias, receipts in quality.items():
            _require(receipts == self._quality(alias), "quality receipt changed")
        return cast(dict, visual_context["base"]), deepcopy(quality)

    def _wrap(self, base: dict, quality: dict[str, list[dict]]) -> dict:
        return {"schema": VISUAL_SCHEMA, "policy_sha256": self.policy_sha256,
                "registration_sha256": self.registration_sha256,
                "index_sha256": self.registration["index_sha256"],
                "base": deepcopy(base), "quality": deepcopy(quality)}

    @staticmethod
    def _available(receipts: list[dict[str, Any]]) -> bool:
        return bool(receipts) and all(row["outcome"] == "available" for row in receipts)

    def inspect_frames(self, loop: Any, alias: str, *, visual_context: dict | None = None) -> dict:
        receipts = self._quality(alias)  # 모델에 이미지 블록을 만들기 전에 판정한다.
        base_visual, quality = self._unwrap(visual_context)
        result = super().inspect_frames(loop, alias, visual_context=base_visual)
        quality[alias] = receipts
        available = self._available(receipts)
        result.update({"schema": SCHEMA + "_evidence_receipt",
                       "visual_context": self._wrap(result["visual_context"], quality),
                       "quality_receipts": deepcopy(receipts),
                       "visual_outcome": "available" if available else "unknown",
                       "frame_refs": result["frame_refs"] if available else [],
                       "image_count": result["image_count"] if available else 0})
        return result

    def read_evidence(self, loop: Any, alias: str, *, visual_context: dict | None = None) -> dict:
        base_visual, quality = self._unwrap(visual_context)
        result = super().read_evidence(loop, alias, visual_context=base_visual)
        result["visual_context"] = self._wrap(result["visual_context"], quality)
        return result

    def messages(self, loop: Any, *, visual_context: dict | None = None) -> dict:
        base_visual, quality = self._unwrap(visual_context)
        result = super().messages(loop, visual_context=base_visual)
        blocks: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        for alias in loop.revealed:
            registered = self.registration["aliases"][alias]
            if registered["kind"] != "frames":
                continue
            receipts = cast(list[dict[str, Any]], quality.get(alias))
            _require(receipts == self._quality(alias), "missing quality receipt")
            if not self._available(receipts):
                continue
            for frame in registered["frame_ids"]:
                view = EVIDENCE.E.frame_view(
                    self.index, registered["evidence_id"], frame, self.roots,
                    expected_sha256=self.registration["index_sha256"])
                content = view["messages"][0]["content"]
                blocks.extend(deepcopy(content))
                citations.append(next(row for row in result["frame_refs"] if row["frame_id"] == frame))
        text = deepcopy(result["messages"][1]["content"][0])
        result.update({"schema": SCHEMA + "_messages", "visual_context": self._wrap(
            cast(dict, result["visual_context"]), quality), "frame_refs": citations,
            "image_count": len(citations)})
        result["messages"][1]["content"] = [text, *blocks]
        result["messages_sha256"] = EVIDENCE.E.index_sha256({"messages": result["messages"]})
        return result


def quality_evidence_spec(adapter: QualityEvidenceAdapter) -> dict[str, Any]:
    value = BASE.evidence_spec(adapter)
    value["quality_policy"] = deepcopy(adapter.policy_ref)
    return value


def adapter_from_spec(spec: dict[str, Any]) -> QualityEvidenceAdapter:
    _require(set(spec) == {"index", "roots", "registration", "registration_sha256",
                           "quality_policy"}, "quality evidence spec")
    return QualityEvidenceAdapter(
        spec["index"], {key: Path(value) for key, value in spec["roots"].items()},
        spec["registration"], expected_registration_sha256=spec["registration_sha256"],
        policy_ref=spec["quality_policy"])


@contextmanager
def adapter_scope(spec: dict[str, Any]) -> Iterator[None]:
    """현재 thread의 정확한 spec만 교체하고 모든 종료 경로에서 factory를 복원한다."""
    owner = threading.get_ident()
    with _ADAPTER_LOCK:
        original = BASE._adapter

        def selected(candidate: dict | None) -> EVIDENCE.EvidenceAdapter | None:
            if threading.get_ident() != owner:
                return original(candidate)
            if candidate is None:
                return original(None)
            _require(candidate == spec, "quality evidence spec changed; new registration required")
            return adapter_from_spec(spec)

        BASE._adapter = selected
        try:
            yield
        finally:
            BASE._adapter = original


class Runtime(INPUT.Runtime):
    """현재 runtime thread의 정확한 evidence spec에만 quality adapter를 주입한다."""

    def __init__(self, folder: Path, registration_ref: dict, protocol_ref: dict,
                 source_anchor: str, *, repo: Path = INPUT.ROOT):
        protocol = BASE.read_ref(protocol_ref)
        evidence = BASE.read_ref(registration_ref).get("evidence")
        _require(isinstance(protocol.get("sources_lf"), dict)
                 and REQUIRED_SOURCES <= protocol["sources_lf"].keys(),
                 "quality guard/test sources are not registered")
        _require(isinstance(evidence, dict) and isinstance(evidence.get("quality_policy"), dict)
                 and protocol.get("quality_policy_sha256")
                 == evidence["quality_policy"].get("sha256"), "quality policy protocol binding")
        super().__init__(folder, registration_ref, protocol_ref, source_anchor, repo=repo)

    def advance(self, *, complete: Any = None, host: Any = None, now_utc: Any = None) -> dict:
        spec = self.reg["evidence"]
        with adapter_scope(spec):
            return super().advance(complete=complete, host=host, now_utc=now_utc)

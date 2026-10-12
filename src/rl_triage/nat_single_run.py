"""정상 기준이 없는 단일 실행 진단 도구의 NeMo Agent Toolkit 등록."""
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, ConfigDict

from rl_triage import single_run as S
from rl_triage import grounded_evidence as G
from rl_triage import grounded_active_profile as A
from rl_triage.nat_functions import _dump, _trace


class SingleRunContextConfig(FunctionBaseConfig, name="triage_single_run_context"):
    pass


@register_function(config_type=SingleRunContextConfig)
async def single_run_context_fn(config: SingleRunContextConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("single_run_context", {"case_id": case_id}, _dump(S.get_context(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "단일 실행의 텔레메트리와 해시로 연결된 설정 근거를 읽는다. 선택된 보상·관측·센서 등의 설정 값과 "
        "근거 ID를 반환한다. 설정은 행동이나 원인 확정의 증거가 아니다. 미제공은 설정 부재를 뜻하지 않는다."))


class SingleRunOverviewConfig(FunctionBaseConfig, name="triage_single_run_overview"):
    pass


@register_function(config_type=SingleRunOverviewConfig)
async def single_run_overview_fn(config: SingleRunOverviewConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("single_run_overview", {"case_id": case_id}, _dump(S.telemetry_overview(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "정상 기준이 없는 단일 학습 실행의 모든 스칼라 요약을 보여준다. 각 지표의 표본 수, 처음·마지막 값, "
        "최솟값·최댓값, 앞·뒤 20% 평균을 반환하며 정상 기준 비율이나 정상성 판정은 만들지 않는다."))


class SingleRunSeriesConfig(FunctionBaseConfig, name="triage_single_run_series"):
    pass


@register_function(config_type=SingleRunSeriesConfig)
async def single_run_series_fn(config: SingleRunSeriesConfig, builder: Builder):
    async def _run(case_id: str, tag: str) -> str:
        return _trace("single_run_series", {"case_id": case_id, "tag": tag}, _dump(S.get_series(case_id, tag)))
    yield FunctionInfo.from_fn(_run, description=(
        "단일 실행에서 지표 하나의 학습 곡선을 20개 이하 지점으로 샘플링하고 전체 표본 수를 반환한다. "
        "정상 기준 곡선은 제공하지 않는다."))


class SingleRunAnalysisConfig(FunctionBaseConfig, name="triage_single_run_analysis"):
    timeout_s: int = 30


@register_function(config_type=SingleRunAnalysisConfig)
async def single_run_analysis_fn(config: SingleRunAnalysisConfig, builder: Builder):
    async def _run(case_id: str, code: str) -> str:
        result = S.run_analysis(code, case_id, timeout_s=config.timeout_s)
        return _trace("single_run_analysis", {"case_id": case_id, "code": code}, _dump(result))
    yield FunctionInfo.from_fn(_run, description=(
        "격리된 파이썬 분석 코드로 단일 실행 곡선을 계산한다. run은 지표명에서 값 목록으로 이어지는 객체이며 "
        "ref는 정상 기준이 없음을 나타내는 빈 객체다. 결과는 print로 출력한다."))


class WriteSingleAssessmentConfig(FunctionBaseConfig, name="triage_write_single_assessment"):
    pass


@register_function(config_type=WriteSingleAssessmentConfig)
async def write_single_assessment_fn(config: WriteSingleAssessmentConfig, builder: Builder):
    async def _run(case_id: str, hypotheses: list[str], evidence: str,
                   limitations: str, next_check: str, fact_claims: list[dict] | None = None) -> str:
        args = {"case_id": case_id, "hypotheses": hypotheses, "evidence": evidence,
                "limitations": limitations, "next_check": next_check, "fact_claims": fact_claims}
        return _trace("write_single_assessment", args, _dump(S.write_assessment(**args)))
    yield FunctionInfo.from_fn(_run, description=(
        "단일 실행의 가설·수치 근거·한계·다음 계측 하나를 저장한다. 설정 근거가 있으면 fact_claims에 "
        "인용한 id와 value를 하나 이상 넣는다. 설정 해시와 인용 값이 맞아야 저장되며 자유 문장 전체의 "
        "정확성을 판정하지는 않는다. status는 hypotheses_only이고 기존 평가는 덮어쓰지 않는다."))


class GroundedEvidenceCatalogConfig(FunctionBaseConfig, name="triage_grounded_evidence_catalog"):
    pass


@register_function(config_type=GroundedEvidenceCatalogConfig)
async def grounded_evidence_catalog_fn(config: GroundedEvidenceCatalogConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("grounded_evidence_catalog", {"case_id": case_id}, _dump(G.evidence_catalog(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "전체 단일 실행 telemetry의 원본 SHA와 고정 summary 필드, 선택된 configuration fact만 근거 ID로 반환한다. "
        "missing과 null을 구분하며 샘플 곡선이나 정상성·인과 판단을 만들지 않는다."))


class GroundedEvidenceGroupIndexConfig(FunctionBaseConfig, name="triage_grounded_evidence_group_index"):
    pass


@register_function(config_type=GroundedEvidenceGroupIndexConfig)
async def grounded_evidence_group_index_fn(config: GroundedEvidenceGroupIndexConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace(
            "grounded_evidence_group_index", {"case_id": case_id},
            _dump(G.evidence_catalog_group_index(case_id)),
        )
    yield FunctionInfo.from_fn(_run, description=(
        "근거 값을 싣지 않고 telemetry tag와 configuration 경로로 나눈 결정적 그룹 인덱스를 반환한다. "
        "모든 근거 ID의 그룹 범위와 원본 SHA를 제공하지만 포괄 진단이나 원인 순위를 만들지 않는다."))


class GroundedEvidenceGroupPageConfig(FunctionBaseConfig, name="triage_grounded_evidence_group_page"):
    pass


@register_function(config_type=GroundedEvidenceGroupPageConfig)
async def grounded_evidence_group_page_fn(config: GroundedEvidenceGroupPageConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, group_id: str,
                   offset: int = 0, limit: int = G.GROUP_PAGE_LIMIT) -> str:
        args = {"case_id": case_id, "catalog_sha256": catalog_sha256,
                "group_id": group_id, "offset": offset, "limit": limit}
        return _trace("grounded_evidence_group_page", args, _dump(G.evidence_catalog_group_page(**args)))
    yield FunctionInfo.from_fn(_run, description=(
        "선택한 중립 경로 그룹의 근거 ID·경로·상태·단위만 제한된 페이지로 반환한다. 값은 포함하지 않으며 "
        "미조회 그룹과 페이지 수를 남겨 부분 조회를 포괄 진단으로 취급하지 않게 한다."))


class GroundedEvidenceDetailsConfig(FunctionBaseConfig, name="triage_grounded_evidence_details"):
    pass


@register_function(config_type=GroundedEvidenceDetailsConfig)
async def grounded_evidence_details_fn(config: GroundedEvidenceDetailsConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, evidence_ids: list[str]) -> str:
        args = {"case_id": case_id, "catalog_sha256": catalog_sha256,
                "evidence_ids": evidence_ids}
        return _trace("grounded_evidence_details", args, _dump(G.evidence_catalog_details(**args)))
    yield FunctionInfo.from_fn(_run, description=(
        "그룹 페이지에서 고른 최대 16개 근거 ID의 원문 행을 현재 catalog SHA에 묶어 반환한다. "
        "missing·null·0을 그대로 보존하며 근거 선택이나 원인 판단을 대신하지 않는다."))


class WriteGroundedAssessmentConfig(FunctionBaseConfig, name="triage_write_grounded_assessment"):
    pass


@register_function(config_type=WriteGroundedAssessmentConfig)
async def write_grounded_assessment_fn(config: WriteGroundedAssessmentConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, selection_binding: dict,
                   hypotheses: list[dict], next_check: dict) -> str:
        args = {"case_id": case_id, "catalog_sha256": catalog_sha256,
                "selection_binding": selection_binding,
                "hypotheses": hypotheses, "next_check": next_check}
        return _trace("write_grounded_assessment", args, _dump(G.write_grounded_assessment(**args)))
    yield FunctionInfo.from_fn(_run, description=(
        "host가 발급한 selection_binding에 포함된 근거 ID의 값 관계만 검증해 복수 가설과 read-only 다음 측정을 "
        "저장한다. 자유 서술과 사전 예측의 "
        "과학적 타당성은 검증하지 않으며, 측정은 실행하지 않고 approval_pending으로 등록한다."))


class GroundedActiveRegistryConfig(FunctionBaseConfig, name="triage_grounded_active_registry"):
    pass


class GroundedActiveRegistryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_function(config_type=GroundedActiveRegistryConfig)
async def grounded_active_registry_fn(config: GroundedActiveRegistryConfig, builder: Builder):
    async def _run(request: GroundedActiveRegistryInput) -> str:
        return _trace("grounded_active_registry", {}, _dump(A.active_registry_profile()))
    yield FunctionInfo.from_fn(_run, input_schema=GroundedActiveRegistryInput, description=(
        "기존 read-only 3개와 승인된 simulation reset 검사 하나의 고정 registry를 반환한다. "
        "도구는 승인·실행·GPU 접근을 수행하지 않는다."))


class WriteGroundedActiveAssessmentConfig(
        FunctionBaseConfig, name="triage_write_grounded_active_assessment"):
    pass


@register_function(config_type=WriteGroundedActiveAssessmentConfig)
async def write_grounded_active_assessment_fn(
        config: WriteGroundedActiveAssessmentConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, selection_binding: dict,
                   hypotheses: list[dict], next_check: dict) -> str:
        args = {"case_id": case_id, "catalog_sha256": catalog_sha256,
                "selection_binding": selection_binding, "hypotheses": hypotheses,
                "next_check": next_check}
        return _trace("write_grounded_active_assessment", args, _dump(
            A.write_grounded_active_assessment(**args)))
    yield FunctionInfo.from_fn(_run, description=(
        "host selection 근거에 묶인 초기 가설과 사전 prediction을 active profile로 write-once 저장한다. "
        "승인·소비·실행·receipt 생성은 모델 도구로 제공하지 않는다."))

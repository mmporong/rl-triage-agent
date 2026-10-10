"""정상 기준이 없는 단일 실행 진단 도구의 NeMo Agent Toolkit 등록."""
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from rl_triage import single_run as S
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

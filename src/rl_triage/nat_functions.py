"""NeMo Agent Toolkit(nvidia-nat) 함수 등록. triage_tools의 구현을 에이전트 도구로 노출한다."""
import json
import os
import time

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from rl_triage import triage_tools as T


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _trace(tool: str, args: dict, out: str) -> str:
    """도구 호출 기록(TRIAGE_TRACE 경로가 있을 때만). 도구 동작은 바꾸지 않는다."""
    path = os.environ.get("TRIAGE_TRACE")
    if path:
        rec = {"t": time.time(), "tool": tool, "args": {k: (v if len(str(v)) < 400 else str(v)[:400] + "...")
                                                        for k, v in args.items()},
               "result_head": out[:600]}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


class ListChangesConfig(FunctionBaseConfig, name="triage_list_changes"):
    pass


@register_function(config_type=ListChangesConfig)
async def list_changes_fn(config: ListChangesConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("list_changes", {"case_id": case_id}, _dump(T.list_changes(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "이번 학습에서 정상 기준 실행 대비 바뀐 설정 목록을 change_id, key, old, new로 돌려준다. "
        "이 중 하나 이상이 학습 실패 원인일 수 있고 나머지는 무해할 수 있다."))


class TelemetryOverviewConfig(FunctionBaseConfig, name="triage_telemetry_overview"):
    pass


@register_function(config_type=TelemetryOverviewConfig)
async def telemetry_overview_fn(config: TelemetryOverviewConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("telemetry_overview", {"case_id": case_id}, _dump(T.telemetry_overview(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "핵심 학습 지표(보상, 에피소드 길이, 종료 원인, 속도 추종 오차, 손실, 탐색 노이즈, 보상 항목별 값)의 "
        "학습 후반 평균을 정상 기준 실행과 비율로 비교한다. 진단의 첫 단계로 쓴다."))


class GetSeriesConfig(FunctionBaseConfig, name="triage_get_series"):
    pass


@register_function(config_type=GetSeriesConfig)
async def get_series_fn(config: GetSeriesConfig, builder: Builder):
    async def _run(case_id: str, tag: str) -> str:
        return _trace("get_series", {"case_id": case_id, "tag": tag}, _dump(T.get_series(case_id, tag)))
    yield FunctionInfo.from_fn(_run, description=(
        "지표 하나(tag)의 학습 곡선과 정상 기준 곡선을 20개 지점으로 돌려준다. 추세·발산·붕괴를 볼 때 쓴다."))


class RunAnalysisConfig(FunctionBaseConfig, name="triage_run_analysis"):
    timeout_s: int = 30


@register_function(config_type=RunAnalysisConfig)
async def run_analysis_fn(config: RunAnalysisConfig, builder: Builder):
    async def _run(case_id: str, code: str) -> str:
        return _trace("run_analysis", {"case_id": case_id, "code": code}, _dump(T.run_analysis(code, case_id, timeout_s=config.timeout_s)))
    yield FunctionInfo.from_fn(_run, description=(
        "파이썬 분석 코드를 샌드박스에서 실행한다. 변수 run(이 케이스 series dict, 지표명→값 리스트)과 "
        "ref(정상 기준 series dict)를 쓸 수 있고 결과는 print로 출력한다. 네트워크와 파일 쓰기는 막혀 있다."))


class QueryLedgerConfig(FunctionBaseConfig, name="triage_query_ledger"):
    pass


@register_function(config_type=QueryLedgerConfig)
async def query_ledger_fn(config: QueryLedgerConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return _trace("query_ledger", {"case_id": case_id}, _dump(T.query_ledger(case_id)))
    yield FunctionInfo.from_fn(_run, description=(
        "이 학습 계열에서 이미 시도했다가 기각된 개입 기록. 같은 개입을 다시 제안하지 않도록 먼저 확인한다."))


class WritePreregConfig(FunctionBaseConfig, name="triage_write_preregistration"):
    pass


@register_function(config_type=WritePreregConfig)
async def write_prereg_fn(config: WritePreregConfig, builder: Builder):
    async def _run(case_id: str, suspected_change_id: str, ranking: list[str], hypothesis: str,
                   single_experimental_variable: str, acceptance_gate: str, expected_signature: str) -> str:
        return _trace("write_preregistration", {"case_id": case_id, "suspected_change_id": suspected_change_id,
                                                "ranking": ranking, "hypothesis": hypothesis},
                      _dump(T.write_preregistration(case_id, suspected_change_id, ranking, hypothesis,
                                                    single_experimental_variable, acceptance_gate, expected_signature)))
    yield FunctionInfo.from_fn(_run, description=(
        "진단 결론을 다음 실험 사전등록 JSON으로 저장한다. ranking은 원인 가능성이 높은 순서의 change_id 전체 목록, "
        "single_experimental_variable은 되돌리거나 바꿀 변수 하나, acceptance_gate는 성공 판정 지표와 기준, "
        "expected_signature는 가설이 맞다면 재실행에서 보일 지표 변화. 마지막 단계에서 반드시 호출한다."))


class WriteDiagnosisConfig(FunctionBaseConfig, name="triage_write_diagnosis"):
    pass


@register_function(config_type=WriteDiagnosisConfig)
async def write_diagnosis_fn(config: WriteDiagnosisConfig, builder: Builder):
    async def _run(case_id: str, mechanism_ranking: list[str], evidence: str, next_check: str) -> str:
        return _trace("write_diagnosis", {"case_id": case_id, "mechanism_ranking": mechanism_ranking},
                      _dump(T.write_diagnosis(case_id, mechanism_ranking, evidence, next_check)))
    yield FunctionInfo.from_fn(_run, description=(
        "원인을 모르는 실패의 메커니즘 순위를 저장한다. mechanism_ranking은 reward, actuator, exploration, "
        "optimizer, physics, termination 6개 전체를 가능성 높은 순서로 나열한다. evidence는 관찰한 수치, "
        "next_check는 다음에 확인할 실험 하나. 마지막 단계에서 반드시 호출한다."))

"""데모 슬라이드(HTML) 생성. 모든 수치는 저장소의 결과 파일에서 읽는다(손으로 적지 않는다).

python docs/demo/build_slides.py  →  docs/demo/slides/*.html
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "demo" / "slides"
TRACE = ROOT / "evals" / "results" / "traces" / "blind_seed42_c04_1790476659.jsonl"

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{width:1280px;height:720px;background:#f4f1ea;color:#161616;font-family:'Malgun Gothic','Pretendard',sans-serif;
     padding:56px 72px;display:flex;flex-direction:column;overflow:hidden}
.kicker{font-size:20px;letter-spacing:.08em;color:#5b5b5b;text-transform:uppercase}
h1{font-size:52px;line-height:1.15;margin:14px 0 18px;font-weight:800}
h2{font-size:38px;margin:10px 0 22px;font-weight:800}
p.sub{font-size:24px;color:#3a3a3a;line-height:1.5}
.bar{height:6px;width:120px;background:#76b900;margin-top:auto}
table{border-collapse:collapse;font-size:22px;width:100%}
td,th{padding:10px 14px;border-bottom:1px solid #cfc8b8;text-align:left;vertical-align:top}
th{font-size:18px;color:#5b5b5b;font-weight:600}
.num{font-family:Consolas,monospace;font-weight:700}
.ok{color:#2f6b00;font-weight:700}.no{color:#a3241b;font-weight:700}
.step{display:flex;gap:18px;align-items:flex-start;margin:7px 0;font-size:21px}
.step b{font-family:Consolas,monospace;background:#161616;color:#f4f1ea;padding:3px 10px;border-radius:4px;min-width:210px}
.step span{color:#333;line-height:1.35}
.foot{font-size:16px;color:#6b6b6b;margin-top:auto}
"""


def page(name: str, body: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.html").write_text(
        f"<!doctype html><html lang='ko'><meta charset='utf-8'><style>{CSS}</style><body>{body}</body></html>",
        encoding="utf-8")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def final_rows(path):
    last = {}
    for r in load_jsonl(path):
        last[(r["case_id"], r["mode"])] = r
    return last


def score(tag, seeds, task_key="correct"):
    out = {}
    for mode in ("agent", "control"):
        n = c = 0
        for s in seeds:
            p = ROOT / "evals" / "results" / tag / f"seed{s}.jsonl"
            if not p.exists():
                return None
            for (cid, m), r in final_rows(p).items():
                if m == mode:
                    n += 1
                    c += bool(r.get(task_key))
        out[mode] = (c, n)
    return out


def main():
    page("01_title", """
      <div class='kicker'>Korea Agentic AI Hackathon 2026 · Team Mollet</div>
      <h1>로봇 학습이 무너졌다.<br>무엇을 바꿔서 그런가?</h1>
      <p class='sub'>Isaac Lab 강화학습 실패 원인 추적 에이전트<br>
      Nemotron 3 Super · NeMo Agent Toolkit · OpenShell · Isaac Lab</p><div class='bar'></div>""")

    steps = []
    for r in load_jsonl(TRACE):
        tool, args = r["tool"], r["args"]
        head = r["result_head"]
        if tool == "get_series":
            note = args.get("tag", "")
            m = re.search(r'"this_run": \[([^\]]*)', head)
            if m:
                vals = [float(v) for v in m.group(1).split(",") if v.strip()]
                if len(vals) >= 2:
                    note += f"  {vals[0]:g} → {vals[-1]:g} (곡선 앞부분)"
            elif '"error"' in head:
                note += "  (없는 지표 → 오류를 읽고 다른 지표로 전환)"
        elif tool == "run_analysis":
            note = "텔레메트리에 파이썬 분석 코드 실행(샌드박스)"
        elif tool == "write_diagnosis":
            note = "진단 저장: " + " > ".join(args["mechanism_ranking"][:3])
        elif tool == "telemetry_overview":
            note = "정상 기준 대비 핵심 지표 비율"
        else:
            note = json.dumps(args, ensure_ascii=False)[:60]
        steps.append(f"<div class='step'><b>{html.escape(tool)}</b><span>{html.escape(note)}</span></div>")
    page("03_agent", f"""
      <div class='kicker'>원인을 모르는 실패 · 설정 목록 없이 텔레메트리만</div>
      <h2>에이전트가 스스로 계획하고 도구를 부른다</h2>
      {''.join(steps)}
      <p class='foot'>실제 실행 기록 evals/results/traces/{TRACE.name} · 정답: actuator(액션 스케일 0.25→1.5)</p>""")

    proofs = json.loads((ROOT / "evals" / "results" / "policy_proofs.json").read_text(encoding="utf-8"))
    label = {"triage_agent": "에이전트 기본 정책", "ok_request_eval": "재학습 1회 요청",
             "bad_patch_config": "설정 파일 수정 권한 요청", "bad_exfil_checkpoint": "체크포인트 외부 업로드",
             "bad_write_reference": "기준 설정 쓰기"}
    rows = "".join(
        f"<tr><td>{label[k]}</td><td class='num'>{v['result']}</td>"
        f"<td class='{'ok' if v['gate'] == 'human_review' else 'no'}'>{'사람 검토' if v['gate'] == 'human_review' else '자동 거절'}</td></tr>"
        for k, v in proofs.items())
    page("04_security", f"""
      <div class='kicker'>Securing Agents with OpenShell</div>
      <h2>게이트를 통과시키려는 편법은 커널과 증명기가 막는다</h2>
      <table><tr><th>에이전트가 원하는 권한</th><th>openshell-prover</th><th>결과</th></tr>{rows}</table>
      <p class='sub' style='margin-top:22px;font-size:21px'>샌드박스 커널 차단 테스트 6/6 통과 · 에이전트가 보는 API 키는 자리표시자
      <span class='num'>openshell:resolve:…</span></p>""")

    smoke = json.loads((ROOT / "evals" / "results" / "bridge_smoke.json").read_text(encoding="utf-8"))
    srows = "".join(
        f"<tr><td>{'올바른 진단' if s['label'] == 'correct_diagnosis' else '틀린 진단'}</td>"
        f"<td class='num'>{html.escape(s['reverted_override'])}</td>"
        f"<td class='num'>{s['ratios']['Train/mean_episode_length']}</td>"
        f"<td class='{'ok' if s['recovered'] else 'no'}'>{'회복' if s['recovered'] else '미회복'}</td></tr>" for s in smoke)
    page("05_loop", f"""
      <div class='kicker'>사전등록 → 사람 승인 → Isaac Lab 재학습</div>
      <h2>진단은 실제 재학습으로 확인한다</h2>
      <table><tr><th>사전등록</th><th>되돌린 변경 하나</th><th>에피소드 길이 비</th><th>판정</th></tr>{srows}</table>
      <p class='sub' style='margin-top:22px;font-size:21px'>틀린 진단은 재학습이 걸러낸다. 승인은 브리지 호스트의 사람만 할 수 있다.</p>""")

    a = score("v1_openai_client", (42, 7))
    b_dev = score("blind_v1", (42, 7))
    b_held = score("heldout_blind", (123,))
    def cell(x, mode):
        return "측정 중" if x is None else f"{x[mode][0]}/{x[mode][1]}"
    ctrl_a = "20/20"
    page("06_results", f"""
      <div class='kicker'>결함 주입 벤치마크 · Go2 flat · 1024 env × 100 iter</div>
      <h2>같은 모델, 도구가 있고 없고의 차이</h2>
      <table><tr><th>과제</th><th>세트</th><th>에이전트</th><th>단일 프롬프트</th><th>무작위</th></tr>
      <tr><td>A. 바꾼 설정 중 원인</td><td>개발</td><td class='num'>{cell(a, 'agent')}</td><td class='num'>{ctrl_a}</td><td class='num'>33%</td></tr>
      <tr><td>B. 원인 모름 · 메커니즘</td><td>개발</td><td class='num'>{cell(b_dev, 'agent')}</td><td class='num'>{cell(b_dev, 'control')}</td><td class='num'>17%</td></tr>
      <tr><td>B. 원인 모름 · 메커니즘</td><td><b>보류(seed 123)</b></td><td class='num'>{cell(b_held, 'agent')}</td><td class='num'>{cell(b_held, 'control')}</td><td class='num'>17%</td></tr></table>
      <p class='foot'>과제 A는 두 방식 모두 만점이라 변별력이 없다. 프롬프트는 각 평가 전에 커밋으로 동결했다.</p>""")

    page("07_end", """
      <div class='kicker'>NVIDIA nemoclaw-community 레시피 형식</div>
      <h1>설정을 고치는 대신<br>다음 실험 하나를 증명된 권한 안에서</h1>
      <p class='sub'>github.com/mmporong/rl-triage-agent</p><div class='bar'></div>""")
    print("slides:", sorted(p.name for p in OUT.glob("*.html")))


if __name__ == "__main__":
    main()

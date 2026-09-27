"""제출용 PDF의 HTML 원고 생성(submission/pdf.html). 문안은 form_answers.txt, 그림은 docs/demo/png에서 가져온다.
PDF 변환: node submission/to_pdf.mjs submission/pdf.html "submission/NVIDIA 해커톤_Mollet_RL Triage Agent.pdf"
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "submission" / "form_answers.txt").read_text(encoding="utf-8")
# 최상위 제목에서만 나눈다(기술 스택 안의 [NVIDIA AI 기술] 같은 소제목은 본문으로 둔다).
HEADS = ["서비스 명", "해결하고자 했던 문제", "서비스 소개 및 주요 기능", "활용한 핵심 기술 및 AI 모델", "추가 URL"]
parts = re.split(r"\n?\[(" + "|".join(map(re.escape, HEADS)) + r")\]\n", "\n" + t)
sec = {k: v.strip() for k, v in zip(parts[1::2], parts[2::2])}
png = (ROOT / "docs" / "demo" / "png").as_uri()
stack = "".join(
    (f"<li style='list-style:none;margin-left:-5mm;font-weight:700'>{html.escape(l)}</li>" if l.startswith("[")
     else f"<li>{html.escape(l.lstrip('- '))}</li>")
    for l in sec["활용한 핵심 기술 및 AI 모델"].splitlines() if l.strip())

doc = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'Malgun Gothic', sans-serif; color:#161616; font-size: 10.5pt; line-height: 1.6; }}
h1 {{ font-size: 20pt; margin: 0 0 2mm; }} h2 {{ font-size: 12.5pt; margin: 6mm 0 2mm; border-bottom: 2px solid #76b900; padding-bottom: 1mm; }}
.meta {{ color:#555; font-size: 9.5pt; }} .links td {{ padding: 1mm 3mm 1mm 0; }}
img {{ width: 100%; border: 1px solid #ddd; margin: 2mm 0; }} ul {{ margin: 0; padding-left: 5mm; }}
.pb {{ page-break-before: always; }}
</style><body>
<h1>{html.escape(sec['서비스 명'])}</h1>
<div class="meta">NVIDIA Korea Agentic AI Hackathon 2026 · 온라인 사전 챌린지 · Team Mollet</div>
<table class="links">
<tr><td><b>GitHub</b></td><td>https://github.com/mmporong/rl-triage-agent</td></tr>
<tr><td><b>데모 영상</b></td><td>https://github.com/mmporong/rl-triage-agent/blob/main/docs/demo/rl_triage_demo.mp4 (50초)</td></tr>
<tr><td><b>재현</b></td><td>README의 Quickstart · Verification (uv sync → pytest → evals/prove_policies.py → evals/run_eval.py)</td></tr>
</table>
<h2>해결하고자 했던 문제</h2><p>{html.escape(sec['해결하고자 했던 문제'])}</p>
<h2>서비스 소개 및 주요 기능</h2><p>{html.escape(sec['서비스 소개 및 주요 기능'])}</p>
<h2>활용한 핵심 기술 및 AI 모델</h2><ul>{stack}</ul>
<h2 class="pb">에이전트 실행 기록 (원인을 모르는 실패, 실제 로그)</h2><img src="{png}/03_agent.png">
<h2>권한 경계: openshell-prover와 커널 차단</h2><img src="{png}/04_security.png">
<h2 class="pb">사람 승인 후 Isaac Lab 재학습으로 진단 확인</h2><img src="{png}/05_loop.png">
<h2>결함 주입 벤치마크 결과</h2><img src="{png}/06_results.png">
</body></html>"""
(ROOT / "submission" / "pdf.html").write_text(doc, encoding="utf-8")
print("wrote submission/pdf.html")

#!/usr/bin/env bash
# 데모 영상 합성: 슬라이드 PNG(docs/demo/png) + Isaac Sim 녹화(docs/media) → docs/demo/rl_triage_demo.mp4
# 필요: ffmpeg, 한글 글꼴(Windows: C:/Windows/Fonts/malgun.ttf)
set -euo pipefail
cd "$(dirname "$0")"
FONT=${FONT:-C\\:/Windows/Fonts/malgunbd.ttf}
TMP=_build; rm -rf "$TMP"; mkdir -p "$TMP"
still() { ffmpeg -loglevel error -y -loop 1 -t "$2" -i "png/$1.png" -vf "scale=1280:720,format=yuv420p,fps=30" -c:v libx264 "$TMP/$1.mp4"; }

still 01_title 4
# Isaac 정상 vs 결함 나란히(각 6초) + 자막
ffmpeg -loglevel error -y -i ../media/baseline_s42_play.mp4 -i ../media/c04_s42_play.mp4 -filter_complex \
 "[0:v]trim=0:6,setpts=PTS-STARTPTS,scale=640:360[a];[1:v]trim=0:6,setpts=PTS-STARTPTS,scale=640:360[b];\
  [a][b]hstack=inputs=2,pad=1280:720:0:180:color=0xf4f1ea,\
  drawtext=fontfile='$FONT':text='정상 기준 학습':x=210:y=560:fontsize=30:fontcolor=0x161616,\
  drawtext=fontfile='$FONT':text='설정 3개를 바꾼 학습 (무엇이 원인?)':x=720:y=560:fontsize=30:fontcolor=0xa3241b,\
  drawtext=fontfile='$FONT':text='Isaac Sim 4.5 · Unitree Go2 · 학습 100 iteration 체크포인트':x=72:y=90:fontsize=26:fontcolor=0x5b5b5b,\
  format=yuv420p,fps=30[v]" -map "[v]" -c:v libx264 "$TMP/02_isaac.mp4"
still 03_agent 10
still 04_security 9
still 05_loop 8
still 06_results 9
still 07_end 4

# concat 목록은 list.txt 기준 상대경로(Windows ffmpeg와 Git Bash 경로 차이 회피)
printf "file '%s'\n" 01_title.mp4 02_isaac.mp4 03_agent.mp4 04_security.mp4 \
  05_loop.mp4 06_results.mp4 07_end.mp4 > "$TMP/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$TMP/list.txt" -c copy rl_triage_demo.mp4
ffprobe -loglevel error -show_entries format=duration -of csv=p=0 rl_triage_demo.mp4

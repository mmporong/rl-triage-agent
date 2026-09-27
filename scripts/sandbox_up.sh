#!/usr/bin/env bash
# OpenShell 샌드박스에서 RL Triage Agent 실행 준비: provider → 작업공간 이미지 → 정책 적용 샌드박스
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEED=${1:-42}
NAME=${SANDBOX_NAME:-rl-triage}
. ~/.config/nvidia/env   # NVIDIA_API_KEY (사용자 로컬, 저장소 밖)

openshell provider get nvidia-triage >/dev/null 2>&1 || \
  openshell provider create --name nvidia-triage --type nvidia --credential NVIDIA_API_KEY

CTX=$(mktemp -d)
cp -r "$REPO/workspace/seed$SEED" "$CTX/workspace"
rm -rf "$CTX/workspace/preregistrations" "$CTX/workspace/scratch"
docker build -q -f "$REPO/docker/Dockerfile.workspace" -t "rl-triage-ws:seed$SEED" "$CTX"

openshell sandbox delete "$NAME" >/dev/null 2>&1 || true
openshell sandbox create --name "$NAME" --from "rl-triage-ws:seed$SEED" \
  --policy "$REPO/policies/triage_agent.yaml" --provider nvidia-triage
openshell sandbox exec -n "$NAME" -- bash -c 'id; ls -la /sandbox/workspace; ls /sandbox/workspace/cases | wc -l'

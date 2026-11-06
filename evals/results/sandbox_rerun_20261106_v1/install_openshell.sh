#!/usr/bin/env bash
# OpenShell 0.1.1 설치(WSL Ubuntu). sudo가 필요한 단계만 모았다. 비밀번호는 처음 한 번만 묻는다.
# - 공식 릴리스 v0.1.1의 Debian 패키지를 받아 공식 체크섬과 대조한 뒤 설치한다.
#   이 패키지에 openshell, openshell-gateway, openshell-prover(/usr/bin)가 함께 들어 있다.
#   (snap은 스토어 채널만 따르고 버전 고정을 지원하지 않아 쓰지 않는다: 공식 install.sh v0.1.1 usage)
# - docker 그룹 추가: Docker Desktop WSL 통합 소켓을 sudo 없이 쓰기 위해.
# - linger: 게이트웨이 사용자 서비스가 터미널을 닫아도 계속 돌게 하기 위해(공식 설치 문서).
# 게이트웨이 시작·등록(systemctl --user, openshell gateway add)은 sudo가 필요 없어 README 따라 하기 3단계에서 사용자 권한으로 한다.
set -euo pipefail
V=0.1.1
DEB="openshell_${V}-1_amd64.deb"
URL="https://github.com/NVIDIA/OpenShell/releases/download/v${V}"
DIR="$HOME/.cache/openshell-v${V}"

mkdir -p "$DIR" && cd "$DIR"
[ -f "$DEB" ] || curl -fsSLO "$URL/$DEB"
curl -fsSLO "$URL/openshell-checksums-sha256.txt"
grep "  ${DEB}\$" openshell-checksums-sha256.txt | sha256sum -c -

sudo dpkg -i "$DEB"
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
sudo loginctl enable-linger "$USER"

dpkg-query -W -f='${Package} ${Version} ${Status}\n' openshell
ls -l /usr/bin/openshell /usr/bin/openshell-gateway /usr/bin/openshell-prover
echo "완료. 이 터미널을 닫고 새 WSL 터미널을 열어야 docker 그룹이 적용됩니다."

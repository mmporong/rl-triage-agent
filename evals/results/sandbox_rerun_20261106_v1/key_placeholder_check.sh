# 샌드박스 안 NVIDIA_API_KEY가 OpenShell 자리표시자인지 확인한다. 실제 키면 값은 숨기고 길이만 출력한다.
# 사용(WSL): openshell sandbox exec -n rl-triage -- bash -c "$(cat key_placeholder_check.sh)" </dev/null
v="$NVIDIA_API_KEY"
case "$v" in
  openshell:*) printf "placeholder prefix=%s len=%s\n" "$(printf %s "$v" | cut -c1-12)" "${#v}" ;;
  "") echo "NVIDIA_API_KEY unset" ;;
  *) printf "NOT a placeholder (value hidden) len=%s\n" "${#v}" ;;
esac

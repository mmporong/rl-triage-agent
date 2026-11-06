"""a6cc301과 수정 커밋 사이에 바뀐 재생 데이터 해시가 모두 CRLF 변환 해시인지 git blob으로 확인한다.

사용법: clone 루트에서 `uv run --no-sync python <이 파일> a6cc301 db59134`
"""
import hashlib, re, subprocess, sys
BEFORE, AFTER = sys.argv[1], sys.argv[2]
def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, check=True).stdout
for f in ["docs/replay/data.js", "docs/replay/additional-checks-v2-data.js", "docs/replay/ppo-gradient-data.js"]:
    old, new = show(BEFORE, f).decode().splitlines(), show(AFTER, f).decode().splitlines()
    assert len(old) == len(new), f
    changed = crlf = other = 0
    for i, (a, b) in enumerate(zip(old, new)):
        if a == b:
            continue
        changed += 1
        if '"sha256"' not in a or '"sha256"' not in b:
            other += 1; continue
        path = next(m.group(1) for j in range(i, -1, -1) if (m := re.search(r'"(?:path|id)": "([^"]+)"', new[j])))
        blob = show(AFTER, path)
        ok = hashlib.sha256(blob).hexdigest() in b and hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest() in a and b"\r" not in blob
        crlf += ok; other += not ok
    print(f"{f}: changed_lines={changed} crlf_hash_explained={crlf} other={other}")

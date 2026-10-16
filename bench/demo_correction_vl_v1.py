"""VL 보조 시연의 등록 context에서 기존 v4 교정 학습 진입점을 실행한다."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import vl_aux_demo_v1 as V


def main(argv=None):
    with V.configured():
        original = V.E.verify_protocol
        V.E.verify_protocol = lambda variant, lab, experiment=None: (
            V.E.read(V.PROTOCOL), V.check_active(lab, experiment)["commit"])
        try:
            return V.DC.main(argv)
        finally:
            V.E.verify_protocol = original


if __name__ == "__main__":
    raise SystemExit(main())

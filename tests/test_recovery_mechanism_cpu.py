"""CPU 감사의 통계 정의를 독립 작은 예로 검증한다."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("recovery_audit", Path(__file__).resolve().parents[1] / "bench/analyze_recovery_mechanism.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class StatisticsTest(unittest.TestCase):
    def test_effective_rank_extremes(self):
        self.assertAlmostEqual(audit.effective_rank([1, 1, 1]), 3)
        self.assertEqual(audit.effective_rank([5, 0, 0]), 1)
        self.assertEqual(audit.effective_rank([0, 0]), 0)
        with self.assertRaises(ValueError):
            audit.effective_rank([float("nan")])

    def test_summary_uses_separate_60_windows(self):
        result = audit.summary([1.] * 60 + [3.] * 60)
        self.assertEqual(result["first_60_mean"], 1)
        self.assertEqual(result["last_60_mean"], 3)
        with self.assertRaises(ValueError):
            audit.summary([])

    def test_bound_input_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "checkpoint_fixture.bin"
            manifest = Path(folder) / "inputs.json"
            data.write_bytes(b"original")
            with patch.object(audit, "input_files", return_value=[data]):
                audit.prepare_commitment(manifest)
                audit.verify_commitment(manifest)
                data.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    audit.verify_commitment(manifest)


if __name__ == "__main__":
    unittest.main()

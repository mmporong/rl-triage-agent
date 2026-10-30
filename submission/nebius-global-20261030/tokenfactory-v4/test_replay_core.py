"""CPU tests for the public recorded-request replay; no inference or GPU."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import replay_core as R

BUNDLE = Path(__file__).resolve().parent
REPO_ROOT = None
SCRATCH_ROOT = None


class PublicReplayTests(unittest.TestCase):
    def setUp(self):
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH_ROOT)
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        if not self.root.is_relative_to(SCRATCH_ROOT.resolve()):
            raise ValueError("temporary cleanup outside declared scratch root")
        self.temporary.cleanup()

    def clone_bundle(self, name):
        target = self.root / name
        index = json.loads((BUNDLE / "PUBLIC_INDEX.json").read_text(encoding="utf-8"))
        for relative in [*index["files"], "PUBLIC_INDEX.json"]:
            parts = PurePosixPath(relative)
            self.assertFalse(parts.is_absolute())
            self.assertNotIn("..", parts.parts)
            source, destination = BUNDLE / relative, target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return target

    def rehash_file(self, bundle, name):
        index_path = bundle / "PUBLIC_INDEX.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        raw = (bundle / name).read_bytes()
        index["files"][name] = {"bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}
        index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    def test_eight_actual_public_requests_match_reconstructed_core_inputs(self):
        report = R.replay(BUNDLE, REPO_ROOT, self.root / "core")
        self.assertEqual(report["records_replayed"], 8)
        self.assertEqual(report["normal_control_holds"], 4)
        self.assertEqual(report["pending_inspections"], 4)
        self.assertTrue(all(row["same_public_request"] for row in report["episodes"]))
        self.assertTrue(all(row["original_core_decisions"] == 1 for row in report["episodes"]))
        self.assertEqual(report["new_HTTP_or_GPU_calls"], 0)
        self.assertFalse(report["execution_allowed"])
        self.assertFalse(report["original_private_execution_source_reauthenticated"])

    def test_fresh_process_replay_is_identical_without_new_provider_use(self):
        output = self.root / "repeat"
        command = [sys.executable,"-B",str(BUNDLE / "replay_core.py"),"--repo-root",str(REPO_ROOT),"--output",str(output)]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        before = (output / "CORE_REPLAY.json").read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(before, (output / "CORE_REPLAY.json").read_bytes())

    def test_rehashed_goal_and_params_changes_rejected_after_basic_bundle_checks_pass(self):
        for mode in ("goal", "params"):
            with self.subTest(mode=mode):
                bundle = self.clone_bundle(mode)
                name = "evidence/ep01.request.json"
                request = json.loads((bundle / name).read_text(encoding="utf-8"))
                if mode == "params":
                    request["max_tokens"] += 1
                else:
                    view = json.loads(request["messages"][1]["content"])
                    view["goal"] += " modified"
                    request["messages"][1]["content"] = json.dumps(view, ensure_ascii=False, sort_keys=True)
                (bundle / name).write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
                self.rehash_file(bundle, name)
                R.load_bundle(bundle)
                with self.assertRaisesRegex(ValueError,"model-facing request differs"):
                    R.replay(bundle, REPO_ROOT, self.root / ("out-" + mode))

    def test_episode_path_escape_rejected_before_writing_outside_run(self):
        bundle = self.clone_bundle("escape")
        path = bundle / "PUBLIC_INDEX.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        index["core_records"][0]["episode_id"] = "../escape"
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"episode ID"):
            R.replay(bundle, REPO_ROOT, self.root / "out")
        self.assertFalse((self.root / "escape/result.json").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=BUNDLE / "test_runs")
    args = parser.parse_args()
    REPO_ROOT = args.repo_root.resolve()
    SCRATCH_ROOT = args.scratch_root.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)

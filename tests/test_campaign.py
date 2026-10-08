import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from laya_research.campaign import campaign_lock, child, resolve_run
from laya_research.io import write_jsonl
from laya_research.selection import compare, paired_metrics
from tests.helpers import cases


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "seeds": [0, 1],
            "contract_hash": "fixed",
            "objective": "macro_ce",
            "direction": "min",
            "profile": "distribution",
            "min_delta": 0.01,
            "task_regression": 0.1,
            "task_weights": {"task": 1},
        }

    def pair(self, value, task=None, recipe="a"):
        return [
            {
                "seed": seed,
                "recipe_hash": recipe,
                "contract_hash": "fixed",
                "status": "evaluated",
                "metrics": {
                    "macro_ce": value,
                    "macro_hit_at_k": value,
                    "by_task": {"task": {"objective_ce": value if task is None else task}},
                },
            }
            for seed in (0, 1)
        ]

    def test_keep_discard_and_exact_tie(self):
        baseline = self.pair(1)
        self.assertEqual(
            compare(self.pair(0.8), baseline, baseline, self.manifest)["status"], "keep"
        )
        self.assertEqual(
            compare(self.pair(1), baseline, baseline, self.manifest)["status"], "discard"
        )
        self.assertEqual(
            compare(self.pair(1.1), baseline, baseline, self.manifest)["status"], "discard"
        )

    def test_baseline_gate_prevents_accumulated_regression(self):
        result = compare(self.pair(0.7, 1.15), self.pair(0.8, 1.08), self.pair(1, 1), self.manifest)
        self.assertEqual(result["status"], "discard")
        self.assertEqual(result["task_gate_failures"], ["task"])

    def test_seed_recipe_contract_and_status_must_match(self):
        for key, value in (
            ("seed", 5),
            ("recipe_hash", "other"),
            ("contract_hash", "changed"),
            ("status", "crash"),
        ):
            rows = self.pair(1)
            rows[1][key] = value
            with self.assertRaises(ValueError):
                paired_metrics(rows, self.manifest)

    def test_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with campaign_lock(path):
                with self.assertRaises(ValueError):
                    with campaign_lock(path):
                        pass
            self.assertFalse((path / ".writer.lock").exists())

    def test_child_exit_timeout_and_missing_prediction_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            child([sys.executable, "-c", "print('ok')"], log, 10)
            self.assertIn("ok", log.read_text())
            with self.assertRaises(TimeoutError):
                child([sys.executable, "-c", "import time; time.sleep(10)"], log, 0.2)
            with self.assertRaises(ValueError):
                child([sys.executable, "-c", "raise ValueError('bad logits')"], log, 10)

    def test_run_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            resolve_run(Path("."), "../../private")

    def test_cli_end_to_end_data_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, cases())
            script = str(Path(__file__).resolve().parents[1] / "prepare.py")
            for command in (
                ["prepare", "--data", str(source), "--campaign", str(root / "campaign")],
                ["references", "--campaign", str(root / "campaign")],
                ["status", "--campaign", str(root / "campaign")],
            ):
                result = subprocess.run(
                    [sys.executable, script, *command], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "campaign" / "references.json").exists())
            self.assertFalse((root / "campaign" / "runs").exists())

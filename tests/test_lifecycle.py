"""Runner integration with a deterministic fake child, NOT a Laya quality benchmark."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from laya_research import campaign, contract
from laya_research.io import read_json, tree_hashes, write_json, write_jsonl
from laya_research.metrics import evaluate
from tests.helpers import cases


class LifecycleTests(unittest.TestCase):
    def test_baseline_keep_discard_failure_and_test_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, cases())
            path = root / "campaign"
            manifest = contract.prepare(path, source)
            manifest.update(
                status="frozen",
                seeds=[0, 1],
                max_trials=20,
                compute_limit_seconds=10000,
                timeout_seconds=60,
                budget_seconds=1,
                overshoot_seconds=5,
                base_checkpoint=str(root / "base"),
                max_len=64,
                head_max_len=32,
                min_delta=0.01,
                task_regression=0.1,
                max_vram_mb=None,
            )
            recipe = root / "recipe.py"
            test_accesses = []

            def fake_child(command, log, timeout, **kwargs):
                if "laya_research.worker" not in command:
                    context = read_json(Path(command[-1]))
                    out = Path(context["output"])
                    (out / "model.safetensors").write_bytes(b"fake test weights")
                    write_json(
                        out / "training.json",
                        {
                            "train_seconds": 1,
                            "updates": 1,
                            "seed": context["seed"],
                            "peak_vram_mb": 0,
                        },
                    )
                    return 0.001
                run = Path(command[command.index("--run") + 1])
                if "--final" in command:
                    test_accesses.append(run.name)
                    write_json(
                        run / "test-evaluation.json", {"split": "test", "metrics": {"fake": True}}
                    )
                    return 0.001
                rows = contract.records(path, "dev")
                good = "good" in (run / "recipe.py").read_text()
                logits = {
                    r["id"]: [
                        (4.0 if good and i == r["hard"] else 0.0) for i in range(len(r["target"]))
                    ]
                    for r in rows
                }
                metrics, predictions = evaluate(rows, logits, manifest["task_weights"])
                artifact = run / "artifact"
                artifact.mkdir()
                (artifact / "model.safetensors").write_bytes(b"fake exported weights")
                write_json(
                    run / "evaluation.json",
                    {
                        "contract_hash": manifest["contract_hash"],
                        "metrics": metrics,
                        "timing": {"peak_vram_mb": 0},
                        "artifact_hashes": tree_hashes(artifact),
                    },
                )
                write_jsonl(run / "dev-predictions.jsonl", predictions)
                return 0.001

            with (
                patch("laya_research.campaign.load_contract", return_value=manifest),
                patch("laya_research.campaign.child", side_effect=fake_child),
            ):
                recipe.write_text("# baseline", encoding="utf-8")
                baseline = [campaign.run(path, recipe, s, "Baseline") for s in (0, 1)]
                ids = [r["run_id"] for r in baseline]
                self.assertTrue(all(r["status"] == "evaluated" for r in baseline))
                self.assertEqual(campaign.select(path, ids, baseline=True)["status"], "baseline")
                recipe.write_text("# good", encoding="utf-8")
                better = [campaign.run(path, recipe, s, "Better fake") for s in (0, 1)]
                better_ids = [r["run_id"] for r in better]
                self.assertEqual(campaign.select(path, better_ids)["status"], "keep")
                self.assertEqual(campaign.select(path, ids)["status"], "discard")
                self.assertEqual(test_accesses, [])
                with patch(
                    "laya_research.campaign.child", side_effect=ValueError("bad prediction")
                ):
                    failure = campaign.run(path, recipe, 0, "Intentional failure")
                self.assertEqual(failure["status"], "invalid")
                self.assertIsNone(failure["objective_value"])
                self.assertFalse((path / ".writer.lock").exists())
                final = campaign.finalize(path)
                self.assertEqual(set(test_accesses), {ids[0], better_ids[0]})
                self.assertIn(better_ids[0], final["artifact"])
                self.assertEqual(campaign.finalize(path), final)
                with self.assertRaises(ValueError):
                    campaign.run(path, recipe, 0, "Cannot resume after test")
                with self.assertRaises(ValueError):
                    campaign.select(path, ids)
                (path / "runs" / better_ids[0] / "artifact" / "model.safetensors").write_bytes(
                    b"tampered"
                )
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    campaign.checked_report(path, better_ids[0])

    def test_recovery_does_not_resume_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "runs" / "abc" / "run.json", {"run_id": "abc", "status": "running"})
            with self.assertRaises(ValueError):
                campaign.recover(path)
            with patch(
                "laya_research.campaign.load_contract", return_value={"timeout_seconds": 60}
            ):
                self.assertEqual(campaign.recover(path, True), {"recovered": ["abc"]})
            report = read_json(path / "runs" / "abc" / "run.json")
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["total_seconds"], 60)

    def test_recovery_replays_committed_selection_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expected = {"baseline": ["a", "b"], "incumbent": ["c", "d"]}
            write_jsonl(path / "events.jsonl", [{"event": "selection", "state": expected}])
            write_json(path / "state.json", {"incumbent": ["stale"]})
            campaign.recover(path, True)
            self.assertEqual(read_json(path / "state.json"), expected)

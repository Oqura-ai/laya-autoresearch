"""Opt-in REAL integration. Requires installed Laya, local checkpoint and CUDA.

Set LAYA_TEST_CHECKPOINT to the checkpoint directory to enable. This consumes GPU
compute and writes artifacts only inside a temporary directory, removed after test.
"""

import os
import tempfile
import unittest
from pathlib import Path

from laya_research import campaign, contract
from laya_research.io import write_jsonl
from tests.helpers import cases


@unittest.skipUnless(
    os.environ.get("LAYA_TEST_CHECKPOINT"), "real Laya/CUDA checkpoint not configured"
)
class GPUSmokeTests(unittest.TestCase):
    def test_real_train_calibrate_export_and_sdk_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, cases(100))
            path = root / "campaign"
            contract.prepare(path, source)
            contract.freeze(
                path,
                Path(os.environ["LAYA_TEST_CHECKPOINT"]),
                budget=2,
                overshoot=60,
                timeout=600,
                max_len=128,
                head_max_len=96,
                min_delta=0,
                task_regression=0.2,
            )
            recipe = root / "recipe.py"
            recipe.write_text(
                "import argparse\nfrom laya_research.training import Recipe, train\n"
                'p=argparse.ArgumentParser(); p.add_argument("--context",required=True)\n'
                'train(p.parse_args().context, Recipe(loss="soft-ce",micro_batch=2,grad_accum=2))\n',
                encoding="utf-8",
            )
            report = campaign.run(path, recipe, 0, "Real GPU integration smoke")
            self.assertEqual(report["status"], "evaluated", report)
            self.assertTrue(
                (path / "runs" / report["run_id"] / "artifact" / "model.safetensors").exists()
            )

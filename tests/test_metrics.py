import math
import unittest

from laya_research.calibration import fit_temperatures
from laya_research.metrics import (
    aggregate,
    bootstrap_delta,
    ce,
    evaluate,
    log_softmax,
    score_rows,
    validate_predictions,
)
from tests.helpers import records


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.records = records()
        self.logits = {r["id"]: [0.0] * len(r["target"]) for r in self.records}
        tasks = {r["task"] for r in self.records}
        self.weights = {t: 1 / len(tasks) for t in tasks}

    def test_cross_entropy_entropy_kl_identity(self):
        r = records(1, "gold")[0]
        rows = score_rows([r], {r["id"]: [math.log(0.7), math.log(0.3)]})
        self.assertAlmostEqual(rows[0]["kl"], 0)
        self.assertAlmostEqual(rows[0]["ce"], -0.7 * math.log(0.7) - 0.3 * math.log(0.3))

    def test_log_softmax_stability(self):
        self.assertTrue(math.isfinite(ce([10000, -10000], [0, 1])))
        self.assertAlmostEqual(ce([10000, 10000], [1, 0]), math.log(2))

    def test_missing_extra_nonfinite_and_width_rejected(self):
        for mutate in (
            lambda p: p.pop(next(iter(p))),
            lambda p: p.update(extra=[0, 0]),
            lambda p: p.update({self.records[0]["id"]: [float("nan"), 0]}),
            lambda p: p.update({self.records[0]["id"]: [0]}),
        ):
            p = dict(self.logits)
            mutate(p)
            with self.assertRaises(ValueError):
                validate_predictions(self.records, p)

    def test_all_types_distribution_error(self):
        report, rows = evaluate(self.records, self.logits, self.weights)
        self.assertEqual(report["micro"]["metric_support"]["distribution_l2"], 120)
        binary = next(r for r in rows if r["type"] == "noul")
        self.assertAlmostEqual(binary["distribution_l2"], 0.5)
        self.assertAlmostEqual(report["macro_ce"], (2 * math.log(2) + math.log(3)) / 3)

    def test_score_rps_respects_distance(self):
        r = next(r for r in self.records if r["type"] == "score" and r["scalar"] == 0)
        near = score_rows([r], {r["id"]: [-100, 0, -100]})[0]
        far = score_rows([r], {r["id"]: [-100, -100, 0]})[0]
        self.assertLess(near["score_rps"], far["score_rps"])
        self.assertAlmostEqual(far["score_mae"], 2)

    def test_ties_are_canonical_but_noul_threshold_is_explicit(self):
        r = next(r for r in self.records if r["type"] == "choice" and r["hard"] == 0)
        row = score_rows([r], {r["id"]: [0, 0]})[0]
        self.assertEqual(row["hit_at_k"], 1)
        r = next(r for r in self.records if r["type"] == "noul" and r["hard"] == 1)
        row = score_rows([r], {r["id"]: [0, 0]})[0]
        self.assertEqual(row["correct"], 1)

    def test_soft_accuracy_is_not_proper_objective(self):
        q = [0.7, 0.3]
        self.assertLess(sum(p * t for p, t in zip(q, q, strict=True)), 0.7)
        self.assertLess(ce([math.log(0.7), math.log(0.3)], q), ce([0, -100], q))

    def test_group_weighting(self):
        r = self.records[0]
        rows = [dict(score_rows([r], {r["id"]: [0, 0]})[0], ce=0, group_id="A") for _ in range(10)]
        rows += [dict(rows[0], ce=10, group_id="B")]
        self.assertAlmostEqual(aggregate(rows, {r["task"]: 1}, "group")["macro_ce"], 5)
        self.assertAlmostEqual(aggregate(rows, {r["task"]: 1})["macro_ce"], 10 / 11)

    def test_calibration_reduces_ce_and_preserves_ranking(self):
        items = [r for r in records(40, "gold") if r["type"] == "choice"]
        logits = {r["id"]: [5, 0] for r in items}
        temps, report = fit_temperatures(items, logits)
        self.assertGreater(temps["choice"], 1)
        self.assertLess(ce([5, 0], [0.7, 0.3], temps["choice"]), ce([5, 0], [0.7, 0.3]))
        self.assertGreater(
            log_softmax([5, 0], temps["choice"])[0], log_softmax([5, 0], temps["choice"])[1]
        )

    def test_calibration_support_and_flat_logits(self):
        temps, report = fit_temperatures(
            self.records[:3], {r["id"]: self.logits[r["id"]] for r in self.records[:3]}
        )
        self.assertEqual(temps, {"choice": 1, "score": 1, "noul": 1})
        self.assertIn("not_fitted", report["choice"]["status"])
        temps, _ = fit_temperatures(self.records, self.logits)
        self.assertEqual(temps, {"choice": 1, "score": 1, "noul": 1})

    def test_bootstrap_paired_zero(self):
        _, rows = evaluate(self.records, self.logits, self.weights)
        interval = bootstrap_delta(rows, rows, self.weights, samples=100)
        self.assertEqual(interval["interval"], [0, 0])

    def test_missing_hard_labels_are_null_not_teacher_truth(self):
        rs = records(2, "gold")
        for r in rs:
            r["hard"], r["scalar"] = None, None
        report, _ = evaluate(rs, {r["id"]: self.logits[r["id"]] for r in rs}, self.weights)
        self.assertIsNone(report["micro"]["ece_15"]["value"])
        self.assertIsNotNone(report["micro"]["teacher_agreement"])

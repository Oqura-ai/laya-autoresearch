import tempfile
import unittest
from pathlib import Path

from laya_research import data
from laya_research.contract import load_contract, prepare
from laya_research.io import loads, write_jsonl
from tests.helpers import QUESTIONS, cases


class DataTests(unittest.TestCase):
    def test_all_types_and_sources(self):
        rows = data.normalize(cases())
        splits = data.split_cases(rows)
        items = data.expand(splits["train"], "gold")
        self.assertEqual({r["type"] for r in items}, {"choice", "noul", "score"})
        self.assertEqual(items[0]["target"], [0.7, 0.3])

    def test_bad_soft_targets(self):
        for values in (
            {"a": 0, "b": 0},
            {"a": 0.7},
            {"a": -0.1, "b": 1.1},
            {"a": float("nan"), "b": 1},
            {"a": True, "b": 0},
            {"a": 0.8, "b": 0.3},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                data.soft_target(QUESTIONS["team"], {"probabilities": values})

    def test_rounding_only_normalization(self):
        q = data.soft_target(QUESTIONS["team"], {"probabilities": {"a": 0.70001, "b": 0.3}})
        self.assertAlmostEqual(sum(q), 1)

    def test_noul_is_strict_boolean(self):
        for value in ("false", 0, 1, None):
            with self.assertRaises(ValueError):
                data.hard_target(QUESTIONS["flag"], value)
        self.assertEqual(data.hard_target(QUESTIONS["flag"], False)[0], [1, 0])

    def test_score_interpolation_opt_in(self):
        with self.assertRaises(ValueError):
            data.hard_target(QUESTIONS["level"], 1.5)
        self.assertEqual(
            data.hard_target(QUESTIONS["level"], 1.5, True), ([0, 0.5, 0.5], None, 1.5)
        )
        for value in (True, -1, 3, float("inf")):
            with self.assertRaises(ValueError):
                data.hard_target(QUESTIONS["level"], value, True)

    def test_missing_is_not_negative(self):
        rows = cases(4)
        del rows[0]["expected"]["flag"]
        normalized = data.normalize(rows)
        for r in normalized:
            r["split_group"] = r["group_id"]
        items = data.expand(normalized)
        self.assertEqual(len(items), 11)
        self.assertEqual(len(data.expand(normalized, "gold")), 12)

    def test_duplicate_ids_and_unknown_questions(self):
        rows = cases()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            data.normalize(rows)
        rows = cases()
        rows[0]["expected"]["typo"] = True
        with self.assertRaises(ValueError):
            data.normalize(rows)

    def test_group_split_deterministic_no_overlap(self):
        a = data.split_cases(data.normalize(cases()))
        b = data.split_cases(data.normalize(list(reversed(cases()))))
        self.assertEqual(a, b)
        seen = set()
        for split in data.SPLITS:
            groups = {r["split_group"] for r in a[split]}
            self.assertTrue(groups)
            self.assertFalse(groups & seen)
            seen |= groups

    def test_duplicate_state_unions_transitive_groups(self):
        raw = cases()
        raw[1]["state"] = raw[0]["state"]
        raw[2]["group_id"] = raw[1]["group_id"]
        raw[3]["state"] = raw[2]["state"]
        split = data.split_cases(data.normalize(raw))
        mapping = {r["id"]: key for key, rows in split.items() for r in rows}
        self.assertEqual(len({mapping[r["id"]] for r in raw[:4]}), 1)

    def test_official_test_overlap_rejected(self):
        raw, test = cases(), cases(1)
        test[0]["id"], test[0]["group_id"] = "test", "test"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            data.split_cases(
                data.normalize(raw), (0.7, 0.1, 0.2, 0), external_test=data.normalize(test)
            )

    def test_small_group_population_rejected(self):
        raw = cases()
        for r in raw:
            r["group_id"] = "one"
        with self.assertRaises(ValueError):
            data.split_cases(data.normalize(raw))

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            loads('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            loads('{"x":NaN}')

    def test_csv_requires_declared_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "cases.csv"
            file.write_text("id,text,label\nx,Hello,a\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                data.read_data(file)
            rows = data.read_data(file, {"team": QUESTIONS["team"]})
            self.assertEqual(rows[0]["expected"], {"team": "a"})

    def test_prepare_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, cases())
            manifest = prepare(root / "campaign", source)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(load_contract(root / "campaign"), manifest)
            with self.assertRaises(ValueError):
                load_contract(root / "campaign", frozen=True)
            with self.assertRaises(ValueError):
                prepare(root / "campaign", source)
            with (root / "campaign" / "data" / "dev.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "file changed"):
                load_contract(root / "campaign")

    def test_shortlist_requires_hard_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, cases())
            with self.assertRaises(ValueError):
                prepare(root / "campaign", source, profile="shortlist", k=2)

    def test_repository_quickstart_fixture(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(
                Path(directory) / "campaign",
                root / "examples/support.jsonl",
                questions=root / "examples/questions.json",
            )
            self.assertEqual(len(manifest["task_weights"]), 3)

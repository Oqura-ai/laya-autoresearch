"""Validate typed decisions and split related cases before expanding questions."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import canonical, digest, read_jsonl

SPLITS = ("train", "calibration", "dev", "test")
TYPES = ("choice", "score", "noul")


def options(question: dict) -> list[str]:
    if not isinstance(question, dict) or question.get("type") not in TYPES:
        raise ValueError("question type must be choice, noul or score")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("question instructions must be nonempty text")
    kind, criteria = question["type"], question.get("criteria")
    if "option_order" in question:
        raise ValueError("option_order is training augmentation, not part of v1 schemas")
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) != {"false", "true"}
        ):
            raise ValueError("noul criteria must describe false and true")
        labels = question.get("labels", {"false": "false", "true": "true"})
        if (
            not isinstance(labels, dict)
            or set(labels) != {"false", "true"}
            or any(not isinstance(v, str) or not v.strip() for v in labels.values())
            or labels["false"].strip() == labels["true"].strip()
        ):
            raise ValueError("noul labels must be distinct false/true descriptions")
        return ["false", "true"]
    if "labels" in question:
        raise ValueError("labels is supported only for noul")
    if kind == "choice":
        if isinstance(criteria, dict):
            keys = list(criteria)
        elif isinstance(criteria, list):
            keys = criteria
        else:
            raise ValueError("choice criteria must be a mapping or list of label IDs")
        if (
            len(keys) < 2
            or any(not isinstance(k, str) or not k.strip() for k in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("choice needs at least two distinct nonempty string label IDs")
        return list(keys)
    if not isinstance(criteria, list) or len(criteria) < 2:
        raise ValueError("score criteria must be an ordered list with at least two levels")
    return [str(i) for i in range(len(criteria))]


def hard_target(question: dict, value, fractional: bool = False):
    labels = options(question)
    kind = question["type"]
    scalar = None
    if kind == "choice":
        if not isinstance(value, str) or value not in labels:
            raise ValueError(f"choice label {value!r} not in {labels}")
        index = labels.index(value)
    elif kind == "noul":
        if type(value) is not bool:
            raise ValueError("noul expected must be a JSON boolean, not a string/number")
        index = int(value)
    else:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= len(labels) - 1
        ):
            raise ValueError("score expected must be a finite in-range numeric level")
        scalar = float(value)
        if scalar != int(scalar):
            if not fractional:
                raise ValueError("fractional score requires explicit adjacent interpolation policy")
            low = int(scalar)
            target = [0.0] * len(labels)
            target[low], target[low + 1] = 1 - (scalar - low), scalar - low
            return target, None, scalar
        index = int(scalar)
    return [float(i == index) for i in range(len(labels))], index, scalar


def soft_target(question: dict, value: dict) -> list[float]:
    labels = options(question)
    probs = value.get("probabilities") if isinstance(value, dict) else None
    if not isinstance(probs, dict) or set(probs) != set(labels):
        raise ValueError(f"probabilities must contain exactly {labels}")
    values = [probs[key] for key in labels]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in values
    ):
        raise ValueError("probabilities must be finite nonnegative numbers")
    total = sum(values)
    if total <= 0 or abs(total - 1) > 1e-4:
        raise ValueError(f"probabilities sum to {total}; expected 1 within 1e-4")
    return [v / total for v in values]


def normalize(rows: list[dict], questions: dict | None = None, *, fractional=False) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError(f"case IDs must be unique nonempty strings: {identity!r}")
        seen.add(identity)
        group = row.get("group_id", identity)
        if not isinstance(group, str) or not group:
            raise ValueError(f"{identity}: group_id must be nonempty text")
        state = row.get("state")
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError(f"{identity}: state must be nonempty text, object or list")
        if isinstance(state, str) and not state.strip():
            raise ValueError(f"{identity}: empty state")
        schema = row.get("questions", questions)
        if not isinstance(schema, dict) or not schema:
            raise ValueError(f"{identity}: no questions")
        expected, gold = row.get("expected", {}), row.get("gold", {})
        if not isinstance(expected, dict) or not isinstance(gold, dict):
            raise ValueError(f"{identity}: expected and gold must be mappings")
        if (set(expected) | set(gold)) - set(schema):
            raise ValueError(f"{identity}: labels reference unknown questions")
        workflow = row.get("workflow", "default")
        if not isinstance(workflow, str) or not workflow:
            raise ValueError(f"{identity}: workflow must be nonempty text")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{identity}: metadata must be an object")
        decisions = []
        for qid, question in schema.items():
            if not isinstance(qid, str) or not qid:
                raise ValueError("question IDs must be nonempty strings")
            labels = options(question)
            target_hard, hard, scalar = None, None, None
            if qid in expected:
                target_hard, hard, scalar = hard_target(question, expected[qid], fractional)
            target_soft = soft_target(question, gold[qid]) if qid in gold else None
            decisions.append(
                {
                    "qid": qid,
                    "task": canonical([workflow, qid]),
                    "type": question["type"],
                    "question": question,
                    "labels": labels,
                    "expected_target": target_hard,
                    "gold_target": target_soft,
                    "hard": hard,
                    "scalar": scalar,
                }
            )
        result.append(
            {
                "id": identity,
                "group_id": group,
                "workflow": workflow,
                "state": state,
                "metadata": metadata,
                "decisions": decisions,
            }
        )
    if not result:
        raise ValueError("no cases")
    return result


def read_data(path: Path, questions=None, text_column="text", label_column="label"):
    if path.suffix.lower() != ".csv":
        return read_jsonl(path)
    if not questions or len(questions) != 1:
        raise ValueError(
            "CSV requires an explicit one-choice questions.json; no test vocabulary inference"
        )
    qid, question = next(iter(questions.items()))
    if question.get("type") != "choice":
        raise ValueError("CSV adapter supports one choice task")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", text_column, label_column} <= set(reader.fieldnames):
            raise ValueError("CSV requires id, text and label columns (or declared column names)")
        return [
            {
                "id": r["id"],
                "group_id": r.get("group_id") or r["id"],
                "state": r[text_column],
                "questions": questions,
                "expected": {qid: r[label_column]},
            }
            for r in reader
        ]


def split_cases(cases, ratios=(0.65, 0.1, 0.15, 0.1), seed=42, external_test=None):
    """Union source groups AND duplicate states, then seeded greedy whole-group allocation."""
    if (
        len(ratios) != 4
        or any(not math.isfinite(x) or x < 0 for x in ratios)
        or abs(sum(ratios) - 1) > 1e-9
    ):
        raise ValueError("four nonnegative split ratios must sum to one")
    if any(ratios[i] <= 0 for i in range(3)) or (not external_test and ratios[3] <= 0):
        raise ValueError("train/calibration/dev/test must be nonempty")
    all_cases = cases + (external_test or [])
    if len({c["id"] for c in all_cases}) != len(all_cases):
        raise ValueError("duplicate case IDs across input files")
    parent = {c["group_id"]: c["group_id"] for c in all_cases}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    states = {}
    for case in all_cases:
        key = digest(case["state"])
        if key in states:
            a, b = find(case["group_id"]), find(states[key])
            parent[max(a, b)] = min(a, b)
        states[key] = case["group_id"]
    for case in all_cases:
        case["split_group"] = find(case["group_id"])
    if external_test:
        if {c["split_group"] for c in cases} & {c["split_group"] for c in external_test}:
            raise ValueError("official test overlaps training groups or exact states")
    groups = defaultdict(list)
    for case in cases:
        groups[case["split_group"]].append(case)
    active = [i for i, r in enumerate(ratios) if r > 0 and not (external_test and i == 3)]
    if len(groups) < len(active):
        raise ValueError("too few independent groups to make all partitions")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda g: -len(groups[g]))  # stable randomized ties
    denom = sum(ratios[i] for i in active)
    targets = {i: len(cases) * ratios[i] / denom for i in active}
    splits = {name: [] for name in SPLITS}
    for number, group in enumerate(ordered):
        empty = [i for i in active if not splits[SPLITS[i]]]
        pool = empty if len(ordered) - number == len(empty) else active
        selected = max(pool, key=lambda i: (targets[i] - len(splits[SPLITS[i]])) / targets[i])
        splits[SPLITS[selected]].extend(groups[group])
    if external_test:
        splits["test"] = external_test
    return {name: sorted(rows, key=lambda r: r["id"]) for name, rows in splits.items()}


def expand(cases, source="expected"):
    if source not in ("expected", "gold"):
        raise ValueError("target source must explicitly be expected or gold")
    records = []
    for case in cases:
        for decision in case["decisions"]:
            target = decision[source + "_target"]
            if target is not None:
                records.append(
                    {
                        **decision,
                        "id": canonical([case["id"], decision["qid"]]),
                        "case_id": case["id"],
                        "group_id": case["split_group"],
                        "workflow": case["workflow"],
                        "state": case["state"],
                        "language": case["metadata"].get("language", "unknown"),
                        "target": target,
                        "target_source": source,
                    }
                )
    return records


def describe(cases, records):
    tasks = defaultdict(list)
    for record in records:
        tasks[record["task"]].append(record)
    return {
        "cases": len(cases),
        "groups": len({c["split_group"] for c in cases}),
        "decisions": len(records),
        "missing_labels": sum(len(c["decisions"]) for c in cases) - len(records),
        "tasks": {
            task: {
                "decisions": len(items),
                "groups": len({r["group_id"] for r in items}),
                "hard_labels": dict(
                    Counter(str(r["hard"]) for r in items if r["hard"] is not None)
                ),
            }
            for task, items in sorted(tasks.items())
        },
    }

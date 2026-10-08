"""Fixed full-precision scoring. No model, training code or third-party dependencies."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import fmean


def log_softmax(logits, temperature=1.0):
    if (
        len(logits) < 2
        or any(not math.isfinite(x) for x in logits)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid logits or temperature")
    z = [float(x) / temperature for x in logits]
    maximum = max(z)
    z = [x - maximum for x in z]
    normalizer = math.log(sum(math.exp(x) for x in z))
    return [x - normalizer for x in z]


def ce(logits, target, temperature=1.0):
    if len(logits) != len(target):
        raise ValueError("logit/target width mismatch")
    return -sum(q * lp for q, lp in zip(target, log_softmax(logits, temperature), strict=True))


def validate_predictions(records, predictions):
    expected = {r["id"] for r in records}
    if not records or len(expected) != len(records) or set(predictions) != expected:
        raise ValueError("prediction IDs must exactly cover the fixed evaluation population")
    for record in records:
        logits = predictions[record["id"]]
        if len(logits) != len(record["target"]):
            raise ValueError(f"{record['id']}: wrong number of logits")
        log_softmax(logits)


def score_rows(records, predictions, temperatures=None, k=1):
    validate_predictions(records, predictions)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    temperatures = temperatures or {}
    rows = []
    for r in records:
        logits, q = predictions[r["id"]], r["target"]
        lp = log_softmax(logits, temperatures.get(r["type"], 1.0))
        p = [math.exp(x) for x in lp]
        # Deterministic ties: original canonical option order, never rounded SDK output.
        ranking = sorted(range(len(p)), key=lambda i: (-logits[i], i))
        pred, hard = ranking[0], r["hard"]
        entropy = -sum(x * math.log(x) for x in q if x > 0)
        loss = -sum(x * y for x, y in zip(q, lp, strict=True))
        item = {
            "id": r["id"],
            "task": r["task"],
            "type": r["type"],
            "group_id": r["group_id"],
            "case_id": r["case_id"],
            "workflow": r["workflow"],
            "language": r.get("language", "unknown"),
            "options": len(p),
            "probabilities": p,
            "target": q,
            "ce": loss,
            "entropy": entropy,
            "kl": max(0.0, loss - entropy),
            "ce_raw": ce(logits, q),
            "distribution_l2": sum((a - b) ** 2 for a, b in zip(p, q, strict=True)),
            "pred_label": r["labels"][pred],
            "hard_label": r["labels"][hard] if hard is not None else None,
            "correct": float(pred == hard) if hard is not None else None,
            "confidence": p[pred],
            "labels": r["labels"],
            "teacher_agreement": float(pred == max(range(len(q)), key=q.__getitem__))
            if r["target_source"] == "gold"
            else None,
            "hit_at_k": float(hard in ranking[:k])
            if r["type"] == "choice" and hard is not None
            else None,
            "mrr": 1 / (ranking.index(hard) + 1)
            if r["type"] == "choice" and hard is not None
            else None,
        }
        if r["type"] == "noul" and hard is not None:
            # Explicit >= 0.5 rule (including ties); choice tie order is unrelated.
            predicted_true = p[1] >= 0.5
            item.update(
                correct=float(predicted_true == bool(hard)),
                pred_label="true" if predicted_true else "false",
                confidence=p[int(predicted_true)],
            )
        if r["type"] == "score":
            item["score_rps"] = sum(
                (sum(p[: j + 1]) - sum(q[: j + 1])) ** 2 for j in range(len(p) - 1)
            ) / (len(p) - 1)
            if r["scalar"] is not None:
                error = abs(sum(i * x for i, x in enumerate(p)) - r["scalar"])
                item.update(
                    score_mae=error,
                    score_normalized_mae=error / (len(p) - 1),
                    score_within_1=float(error <= 1),
                )
        rows.append(item)
    return rows


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return fmean(values) if values else None


def ece(rows, bins=15):
    eligible = [r for r in rows if r.get("correct") is not None]
    buckets = [[] for _ in range(bins)]
    for r in eligible:
        buckets[min(bins - 1, int(r["confidence"] * bins))].append(r)
    details = [
        {"n": len(b), "confidence": average(b, "confidence"), "accuracy": average(b, "correct")}
        for b in buckets
    ]
    value = (
        (
            sum(
                len(b) * abs(average(b, "confidence") - average(b, "correct")) for b in buckets if b
            )
            / len(eligible)
        )
        if eligible
        else None
    )
    return {
        "value": value,
        "n": len(eligible),
        "bins": details,
        "event": "hard-label argmax correctness; noul uses P(true)>=0.5",
    }


def classification(rows):
    rows = [r for r in rows if r.get("hard_label") is not None]
    labels = sorted({label for r in rows for label in r["labels"]})
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for r in rows:
        confusion[r["hard_label"]][r["pred_label"]] += 1
    classes = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[x][label] for x in labels)
        classes[label] = {
            "support": support,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else None,
        }
    defined = [v["f1"] for v in classes.values() if v["f1"] is not None]
    return {
        "n": len(rows),
        "accuracy": average(rows, "correct"),
        "macro_f1": fmean(defined) if defined else None,
        "classes": classes,
        "confusion": confusion,
        "undefined_policy": "null; macro-F1 excludes classes absent in truth AND predictions",
    }


def summarize(rows):
    names = (
        "ce",
        "ce_raw",
        "entropy",
        "kl",
        "distribution_l2",
        "score_rps",
        "score_mae",
        "score_normalized_mae",
        "score_within_1",
        "teacher_agreement",
        "hit_at_k",
        "mrr",
    )
    return {
        "n": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        **{key: average(rows, key) for key in names},
        "ece_15": ece(rows),
        "metric_support": {key: sum(r.get(key) is not None for r in rows) for key in names},
    }


def aggregate(rows, weights, averaging="decision"):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    if (
        set(by_task) != set(weights)
        or any(not math.isfinite(w) or w <= 0 for w in weights.values())
        or abs(sum(weights.values()) - 1) > 1e-8
    ):
        raise ValueError("task coverage/weights differ from the campaign contract")
    if averaging not in ("decision", "group"):
        raise ValueError("averaging must be decision or group")

    def task_mean(items, key):
        if averaging == "decision":
            return average(items, key)
        groups = defaultdict(list)
        for r in items:
            groups[r["group_id"]].append(r)
        values = [average(g, key) for g in groups.values()]
        return (
            fmean(v for v in values if v is not None)
            if any(v is not None for v in values)
            else None
        )

    report = {"micro": summarize(rows), "by_task": {}, "averaging": averaging}
    for key in ("ce", "ce_raw", "hit_at_k"):
        values = {task: task_mean(items, key) for task, items in by_task.items()}
        report["macro_" + key] = (
            sum(weights[t] * values[t] for t in weights)
            if all(v is not None for v in values.values())
            else None
        )
    for task, items in by_task.items():
        report["by_task"][task] = {
            **summarize(items),
            "objective_ce": task_mean(items, "ce"),
            "classification": classification(items),
        }
    for name in ("type", "workflow", "language", "options"):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[name])].append(r)
        report["by_" + name] = {key: summarize(items) for key, items in groups.items()}
    return report


def evaluate(records, predictions, weights, temperatures=None, k=1, averaging="decision"):
    rows = score_rows(records, predictions, temperatures, k)
    return aggregate(rows, weights, averaging), rows


def bootstrap_delta(candidate, incumbent, weights, key="ce", seed=42, samples=500):
    """Descriptive paired group bootstrap, NOT a multiple-comparison correction."""
    if {r["id"] for r in candidate} != {r["id"] for r in incumbent}:
        raise ValueError("paired uncertainty requires identical evaluation decisions")
    other = {r["id"]: r for r in incumbent}
    groups = defaultdict(list)
    for r in candidate:
        groups[r["group_id"]].append((r["task"], r[key] - other[r["id"]][key]))
    if len(groups) < 2:
        return {"status": "insufficient_groups", "interval": None}
    rng, keys, deltas = random.Random(seed), sorted(groups), []
    for _ in range(samples):
        values = defaultdict(list)
        for group in rng.choices(keys, k=len(keys)):
            for task, delta in groups[group]:
                values[task].append(delta)
        if set(values) == set(weights):
            deltas.append(sum(weights[t] * fmean(values[t]) for t in weights))
    if len(deltas) < samples // 2:
        return {"status": "insufficient_task_support", "interval": None}
    deltas.sort()
    return {
        "status": "descriptive_only",
        "successful_samples": len(deltas),
        "interval": [
            deltas[int(0.025 * (len(deltas) - 1))],
            deltas[int(0.975 * (len(deltas) - 1))],
        ],
        "direction": "candidate minus incumbent",
    }

"""Cheap references from training labels only; never fit a prior on dev."""

import math
from collections import defaultdict

from .metrics import evaluate


def reference_predictions(train, evaluation, mode):
    priors = defaultdict(list)
    for r in train:
        priors[(r["task"], tuple(r["labels"]))].append(r["target"])
    result = {}
    for r in evaluation:
        targets = priors.get((r["task"], tuple(r["labels"])), [])
        if mode == "uniform" or not targets:
            logits = [0.0] * len(r["labels"])
        elif mode == "prior":
            counts = [1 + sum(q[i] for q in targets) for i in range(len(r["labels"]))]
            logits = [math.log(c / sum(counts)) for c in counts]
        else:
            raise ValueError("unknown reference")
        result[r["id"]] = logits
    return result


def references(train, dev, manifest):
    return {
        name: evaluate(
            dev,
            reference_predictions(train, dev, name),
            manifest["task_weights"],
            k=manifest["k"],
            averaging=manifest["averaging"],
        )[0]
        for name in ("uniform", "prior")
    }

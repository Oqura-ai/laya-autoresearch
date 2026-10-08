"""Pure matched-seed comparison and baseline-relative task guardrails."""

import math
from statistics import fmean


def paired_metrics(reports, manifest):
    if len(reports) != 2 or [r["seed"] for r in reports] != manifest["seeds"]:
        raise ValueError("comparison requires exactly the two frozen seeds, in order")
    if len({r["recipe_hash"] for r in reports}) != 1:
        raise ValueError("confirmation runs must use the same recipe snapshot")
    if any(
        r["contract_hash"] != manifest["contract_hash"] or r["status"] != "evaluated"
        for r in reports
    ):
        raise ValueError("invalid run status or campaign mismatch")
    return {
        "objective": fmean(r["metrics"][manifest["objective"]] for r in reports),
        "ce": fmean(r["metrics"]["macro_ce"] for r in reports),
        "tasks": {
            t: fmean(r["metrics"]["by_task"][t]["objective_ce"] for r in reports)
            for t in manifest["task_weights"]
        },
    }


def compare(candidate, incumbent, baseline, manifest):
    c = paired_metrics(candidate, manifest)
    i = paired_metrics(incumbent, manifest)
    b = paired_metrics(baseline, manifest)
    if any(
        not math.isfinite(x)
        for d in (c, i, b)
        for x in [d["objective"], d["ce"], *d["tasks"].values()]
    ):
        raise ValueError("nonfinite comparison metric")
    violations = [
        t
        for t in c["tasks"]
        if any(c["tasks"][t] > ref["tasks"][t] + manifest["task_regression"] for ref in (b, i))
    ]
    improvement = (
        (i["objective"] - c["objective"])
        if manifest["direction"] == "min"
        else (c["objective"] - i["objective"])
    )
    better = improvement > manifest["min_delta"]
    if manifest["profile"] == "shortlist" and c["objective"] == i["objective"]:
        better = i["ce"] - c["ce"] > manifest["min_delta"]
    return {
        "status": "keep" if better and not violations else "discard",
        "improvement": improvement,
        "candidate": c,
        "incumbent": i,
        "baseline": b,
        "task_gate_failures": violations,
        "reason": "paired improvement with task gates"
        if better and not violations
        else "no qualifying improvement or task regression",
    }

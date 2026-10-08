"""Deterministic bounded temperature fitting, independent of candidate training."""

import math
from collections import defaultdict

from .metrics import ce, log_softmax, validate_predictions


def fit_temperatures(records, predictions, min_groups=10, min_items=10):
    validate_predictions(records, predictions)
    groups = defaultdict(list)
    for record in records:
        groups[record["type"]].append(record)
    temperatures, report = {}, {}
    for kind in ("choice", "score", "noul"):
        items = groups[kind]
        n_groups = len({r["group_id"] for r in items})
        t, status = 1.0, "not_fitted_insufficient_support"
        if len(items) >= min_items and n_groups >= min_groups:
            # CE is convex in inverse temperature beta. Solve its monotone derivative.
            def derivative(beta, items=items):
                total = 0.0
                for r in items:
                    logits = predictions[r["id"]]
                    probs = [math.exp(x) for x in log_softmax(logits, 1 / beta)]
                    total += sum(
                        (p - q) * z for p, q, z in zip(probs, r["target"], logits, strict=True)
                    )
                return total

            lo, hi = 0.2, 2.0  # T in [0.5, 5.0]
            if derivative(lo) >= 0:
                beta = lo
            elif derivative(hi) <= 0:
                beta = hi
            else:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if derivative(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                beta = (lo + hi) / 2
            # Flat calibration logits should retain T=1, not an arbitrary boundary.
            baseline = sum(ce(predictions[r["id"]], r["target"]) for r in items)
            fitted = sum(ce(predictions[r["id"]], r["target"], 1 / beta) for r in items)
            t = 1 / beta if fitted < baseline - 1e-12 else 1.0
            status = "fitted" if t != 1 else "identity_no_improvement"
        temperatures[kind] = t
        report[kind] = {
            "temperature": t,
            "items": len(items),
            "groups": n_groups,
            "status": status,
            "boundary": abs(t - 0.5) < 1e-8 or abs(t - 5) < 1e-8,
            "warning": "Small calibration sample" if len(items) < 50 else None,
        }
    return temperatures, report

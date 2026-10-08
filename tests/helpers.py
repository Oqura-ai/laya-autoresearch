import copy

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Route the request.",
        "criteria": {"a": "Accounts", "b": "Support"},
    },
    "flag": {"type": "noul", "instructions": "Escalate this request?"},
    "level": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Low", "Medium", "High"],
    },
}


def cases(n=40):
    return [
        {
            "id": f"case-{i:03}",
            "group_id": f"group-{i:03}",
            "state": f"Request {i}",
            "questions": copy.deepcopy(QUESTIONS),
            "expected": {"team": "a" if i % 2 else "b", "flag": bool(i % 2), "level": i % 3},
            "gold": {
                "team": {"probabilities": {"a": 0.7, "b": 0.3}},
                "flag": {"probabilities": {"false": 0.4, "true": 0.6}},
                "level": {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}},
            },
        }
        for i in range(n)
    ]


def records(n=40, source="expected"):
    from laya_research.data import expand, normalize

    normalized = normalize(cases(n))
    for row in normalized:
        row["split_group"] = row["group_id"]
    return expand(normalized, source)

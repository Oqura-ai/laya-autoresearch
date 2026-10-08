"""Campaign preparation and tamper-evident contracts; no implicit mutable defaults."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from .data import SPLITS, describe, expand, normalize, read_data, split_cases
from .io import digest, file_hash, read_json, read_jsonl, tree_hashes, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def infrastructure():
    files = list((ROOT / "laya_research").glob("*.py"))
    files += [ROOT / name for name in ("prepare.py", "program.md", "pyproject.toml")]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files if p.is_file()}


def environment():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = {
                "version": dist.version,
                "source": dist.read_text("direct_url.json"),
            }
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
    }
    try:
        import laya

        result["laya_source"] = tree_hashes(Path(laya.__file__).parent)
    except ImportError:
        result["laya_source"] = None
    return result


def seal(path, manifest):
    manifest.pop("contract_hash", None)
    manifest["contract_hash"] = digest(manifest)
    write_json(path / "manifest.json", manifest)


def load_contract(path: Path, *, frozen=False, runtime=False):
    path = path.resolve()
    manifest = read_json(path / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "contract_hash"}
    if digest(payload) != manifest.get("contract_hash"):
        raise ValueError("manifest fingerprint changed; create a new campaign")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"campaign file changed: {name}")
    if frozen:
        if manifest["status"] != "frozen":
            raise ValueError("campaign is data-only; run freeze on the intended GPU first")
        if infrastructure() != manifest["infrastructure"]:
            raise ValueError("fixed infrastructure changed; start a new campaign")
        if tree_hashes(Path(manifest["base_checkpoint"])) != manifest["base_files"]:
            raise ValueError("base checkpoint changed")
        if runtime and environment() != read_json(path / "environment.lock.json"):
            raise ValueError("Python/dependency/source environment changed")
    return manifest


def records(path, split):
    if split not in SPLITS:
        raise ValueError("unknown partition")
    return read_jsonl(path / "data" / f"{split}.jsonl")


def prepare(
    path: Path,
    data: Path,
    *,
    questions=None,
    test=None,
    seed=42,
    train_source="expected",
    eval_source="expected",
    ratios=None,
    fractional=False,
    profile="distribution",
    k=1,
    averaging="decision",
    text_column="text",
    label_column="label",
):
    if path.exists():
        raise ValueError(f"refusing to overwrite campaign: {path}")
    if profile not in ("distribution", "shortlist") or type(k) is not int or k < 1:
        raise ValueError("invalid objective profile or shortlist k")
    if averaging not in ("decision", "group"):
        raise ValueError("invalid averaging rule")
    schema = read_json(questions) if questions else None
    cases = normalize(
        read_data(data, schema, text_column, label_column), schema, fractional=fractional
    )
    test_cases = (
        normalize(read_data(test, schema, text_column, label_column), schema, fractional=fractional)
        if test
        else None
    )
    ratios = ratios or ((0.7, 0.1, 0.2, 0) if test else (0.65, 0.1, 0.15, 0.1))
    splits = split_cases(cases, ratios, seed, test_cases)
    expanded, report = {}, {}
    for split in SPLITS:
        expanded[split] = expand(splits[split], train_source if split == "train" else eval_source)
        if not expanded[split]:
            raise ValueError(f"{split} has no labels from its declared target source")
        report[split] = describe(splits[split], expanded[split])
    tasks = sorted({r["task"] for r in expanded["train"]})
    for split in SPLITS:
        if {r["task"] for r in expanded[split]} != set(tasks):
            raise ValueError(
                f"{split} task coverage differs from train; supply more data or a suitable split seed"
            )
    for items in expanded.values():
        if profile == "shortlist" and any(
            r["type"] != "choice" or r["hard"] is None or len(r["labels"]) < k for r in items
        ):
            raise ValueError(
                "shortlist requires hard-labeled choice decisions with at least k options"
            )
    # A question's semantics may not drift across cases under the same task ID.
    # Dynamic choice catalogs are allowed, but type and instruction remain fixed.
    semantics = {}
    schemas = {}
    task_schemas = {}
    for split in SPLITS:
        for r in expanded[split]:
            meaning = (r["type"], r["question"]["instructions"])
            if r["type"] != "choice":
                meaning += (digest(r["question"]),)
            if r["task"] in semantics and semantics[r["task"]] != meaning:
                raise ValueError(f"schema semantics change within task {r['task']}")
            semantics[r["task"]] = meaning
            schema_id = digest([r["question"], r["labels"]])
            schemas[schema_id] = r["question"]
            entry = task_schemas.setdefault(
                r["task"], {"workflow": r["workflow"], "qid": r["qid"], "schemas": {}}
            )
            entry["schemas"][schema_id] = r["labels"]
    path.mkdir(parents=True)
    memberships = {
        name: [
            {"id": c["id"], "source_group": c["group_id"], "split_group": c["split_group"]}
            for c in rows
        ]
        for name, rows in splits.items()
    }
    for split in SPLITS:
        write_jsonl(path / "data" / f"{split}.jsonl", expanded[split])
    write_json(path / "data-report.json", report)
    write_json(path / "splits.json", memberships)
    write_json(path / "schemas.json", schemas)
    write_json(
        path / "input-contract.json",
        {
            "tasks": task_schemas,
            "state": "Only information available before the decision; no labels/provenance in model input",
            "option_order": "Preserve the chosen schema's criteria insertion order",
            "dynamic_candidates": "Choice catalogs may vary; keep task meaning/instructions fixed",
        },
    )
    manifest = {
        "version": 1,
        "status": "prepared",
        "sources": {
            "data": file_hash(data),
            "test": file_hash(test) if test else None,
            "questions": file_hash(questions) if questions else None,
        },
        "split_seed": seed,
        "ratios": list(ratios),
        "train_source": train_source,
        "eval_source": eval_source,
        "fractional_scores": fractional,
        "profile": profile,
        "k": k,
        "averaging": averaging,
        "task_weights": {t: 1 / len(tasks) for t in tasks},
        "objective": "macro_ce" if profile == "distribution" else "macro_hit_at_k",
        "direction": "min" if profile == "distribution" else "max",
        "files": tree_hashes(path),
        "warnings": [
            "No count threshold guarantees calibration or generalization.",
            "Group random split; temporal splitting/near-duplicate audit remain user responsibilities.",
        ],
    }
    seal(path, manifest)
    return manifest


def freeze(path: Path, base: Path, **settings):
    from .campaign import campaign_lock

    path = path.resolve()
    with campaign_lock(path):
        return _freeze(path, base, **settings)


def _freeze(
    path,
    base,
    *,
    budget=300.0,
    overshoot=30.0,
    timeout=1800.0,
    max_len=1024,
    head_max_len=256,
    seeds=(0, 1),
    max_trials=20,
    min_delta=0.0,
    task_regression=0.1,
    max_vram_mb=None,
    compute_limit=None,
):
    manifest = load_contract(path)
    if manifest["status"] != "prepared":
        raise ValueError("campaign is already frozen")
    import math

    if (
        any(not math.isfinite(v) for v in (budget, overshoot, timeout, min_delta, task_regression))
        or budget <= 0
        or overshoot < 0
        or timeout <= budget + overshoot
        or min_delta < 0
        or task_regression < 0
        or len(seeds) != 2
        or seeds[0] == seeds[1]
        or max_trials < 4
        or max_len < head_max_len
        or head_max_len < 8
    ):
        raise ValueError("invalid budget, token limits, seeds or promotion gates")
    if max_vram_mb is not None and (not math.isfinite(max_vram_mb) or max_vram_mb <= 0):
        raise ValueError("VRAM cap must be positive")
    compute_limit = compute_limit if compute_limit is not None else max_trials * timeout
    if not math.isfinite(compute_limit) or compute_limit <= 0:
        raise ValueError("compute limit must be positive")
    from .backend import LayaBackend, hardware

    hw = hardware()
    if not hw["cuda"]:
        raise ValueError(
            "freeze requires the intended CUDA GPU; data preparation and unit tests work on CPU"
        )
    base = Path(base).resolve()
    backend = LayaBackend(base, max_len, head_max_len, device="cpu")
    token_report = {}
    for split in SPLITS:
        encoded, stats = backend.encode(records(path, split))
        write_jsonl(path / "cache" / f"{split}.jsonl", encoded)
        token_report[split] = stats
    write_json(path / "token-report.json", token_report)
    write_json(path / "environment.lock.json", environment())
    manifest.update(
        status="frozen",
        base_checkpoint=str(base),
        base_files=tree_hashes(base),
        hardware=hw,
        max_len=max_len,
        head_max_len=head_max_len,
        budget_seconds=budget,
        overshoot_seconds=overshoot,
        timeout_seconds=timeout,
        seeds=list(seeds),
        max_trials=max_trials,
        min_delta=min_delta,
        task_regression=task_regression,
        max_vram_mb=max_vram_mb,
        compute_limit_seconds=compute_limit,
        calibration={"min_items": 10, "min_groups": 10, "bounds": [0.5, 5]},
        infrastructure=infrastructure(),
    )
    # Only immutable inputs belong in the contract, not locks or mutable run summaries.
    for split in SPLITS:
        name = f"cache/{split}.jsonl"
        manifest["files"][name] = file_hash(path / name)
    for name in ("token-report.json", "environment.lock.json"):
        manifest["files"][name] = file_hash(path / name)
    seal(path, manifest)
    return manifest

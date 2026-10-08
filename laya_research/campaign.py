"""Single-writer run supervision, append-only events and explicit final-test access."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

from .contract import ROOT, load_contract
from .io import file_hash, read_json, read_jsonl, tree_hashes, write_json
from .metrics import bootstrap_delta
from .selection import compare, paired_metrics


def timestamp():
    return datetime.now(UTC).isoformat()


@contextmanager
def campaign_lock(path):
    lock = path / ".writer.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ValueError(
            "campaign has a writer lock; inspect status, then use recover after its process stops"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def event(path, payload):
    import json

    with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": timestamp(), **payload}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_result(path, report):
    columns = [
        "run_id",
        "seed",
        "recipe_hash",
        "objective_name",
        "objective_value",
        "train_seconds",
        "total_seconds",
        "status",
        "hypothesis",
    ]
    result = {k: report.get(k) for k in columns}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    if not (path / "results.tsv").exists():
        writer.writeheader()
    writer.writerow(result)
    with (path / "results.tsv").open("a", encoding="utf-8", newline="") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())


def resolve_run(path, identity):
    if (
        not isinstance(identity, str)
        or not identity
        or any(c not in "0123456789abcdef-" for c in identity)
    ):
        raise ValueError("invalid run ID")
    run = path / "runs" / identity
    if not run.is_dir():
        raise ValueError("unknown run ID")
    return run


def child(command, log, timeout, *, training_dir=None, budget=None, overshoot=None):
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        start, train_start = time.monotonic(), None
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if (
                    training_dir
                    and (training_dir / "training-start.json").exists()
                    and train_start is None
                ):
                    train_start = time.monotonic()
                # Allow separately bounded checkpoint serialization after the update budget.
                if elapsed > timeout or (
                    train_start is not None
                    and time.monotonic() - train_start > budget + overshoot + 120
                ):
                    raise TimeoutError("child exceeded supervisor deadline")
                time.sleep(0.1)
            if process.returncode:
                tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
                if "out of memory" in tail.lower():
                    raise MemoryError("CUDA out of memory; see log")
                if "TimeoutError" in tail:
                    raise TimeoutError("training budget overrun; see log")
                if "ValueError" in tail:
                    raise ValueError("child rejected invalid data/weights/numerics; see log")
                raise RuntimeError(f"child exited {process.returncode}; see {log}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        return time.monotonic() - start


def run(path, recipe, seed, hypothesis, untuned=False):
    path, recipe = path.resolve(), recipe.resolve()
    with campaign_lock(path):
        manifest = load_contract(path, frozen=True, runtime=True)
        if (path / "selection-frozen.json").exists():
            raise ValueError("selection is finalized; create a new campaign")
        if seed not in manifest["seeds"]:
            raise ValueError("seed must belong to the frozen confirmation schedule")
        reports = [read_json(p) for p in (path / "runs").glob("*/run.json")]
        spent = sum(r.get("total_seconds", 0) for r in reports)
        if len(reports) >= manifest["max_trials"] or spent >= manifest["compute_limit_seconds"]:
            raise ValueError("campaign trial/compute limit reached")
        if not hypothesis.strip():
            raise ValueError("record a hypothesis before running")
        identity = uuid.uuid4().hex
        directory = path / "runs" / identity
        (directory / "candidate").mkdir(parents=True)
        shutil.copy2(recipe, directory / "recipe.py")
        recipe_hash = file_hash(directory / "recipe.py") if not untuned else "untuned-base"
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
            ).stdout.strip()
            diff = subprocess.run(
                ["git", "diff", "--", str(recipe)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        except FileNotFoundError:
            commit, diff = None, "Git unavailable; exact recipe snapshot is authoritative.\n"
        from .io import atomic_text

        atomic_text(directory / "recipe.diff", diff)
        report = {
            "run_id": identity,
            "status": "running",
            "started": timestamp(),
            "contract_hash": manifest["contract_hash"],
            "recipe_hash": recipe_hash,
            "commit": commit,
            "seed": seed,
            "hypothesis": hypothesis,
            "objective_name": manifest["objective"],
            "objective_value": None,
            "untuned": untuned,
        }
        write_json(directory / "run.json", report)
        context = {
            key: manifest[key]
            for key in (
                "base_checkpoint",
                "max_len",
                "head_max_len",
                "budget_seconds",
                "overshoot_seconds",
            )
        }
        context.update(
            seed=seed,
            train_cache=str(path / "cache" / "train.jsonl"),
            output=str(directory / "candidate"),
            untuned=untuned,
        )
        write_json(directory / "context.json", context)
        event(path, {"event": "run_started", "run_id": identity, "recipe_hash": recipe_hash})
        started = time.monotonic()
        timeout = min(manifest["timeout_seconds"], manifest["compute_limit_seconds"] - spent)
        try:
            if not untuned:
                child(
                    [
                        sys.executable,
                        str(directory / "recipe.py"),
                        "--context",
                        str(directory / "context.json"),
                    ],
                    directory / "train.log",
                    timeout,
                    training_dir=directory / "candidate",
                    budget=manifest["budget_seconds"],
                    overshoot=manifest["overshoot_seconds"],
                )
                training = read_json(directory / "candidate" / "training.json")
                import math

                if (
                    not math.isfinite(training["train_seconds"])
                    or training["updates"] < 1
                    or training["seed"] != seed
                    or training["train_seconds"] < manifest["budget_seconds"]
                    or training["train_seconds"]
                    > manifest["budget_seconds"] + manifest["overshoot_seconds"]
                ):
                    raise ValueError("training did not honor budget/seed/update contract")
                report.update(train_seconds=training["train_seconds"], training=training)
            load_contract(path, frozen=True, runtime=True)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("trial total timeout exhausted before evaluation")
            child(
                [
                    sys.executable,
                    "-m",
                    "laya_research.worker",
                    "--campaign",
                    str(path),
                    "--run",
                    str(directory),
                ],
                directory / "eval.log",
                remaining,
            )
            evaluation = read_json(directory / "evaluation.json")
            if evaluation["contract_hash"] != manifest["contract_hash"]:
                raise ValueError("evaluation contract mismatch")
            cap = manifest["max_vram_mb"]
            if (
                cap is not None
                and max(
                    evaluation["timing"]["peak_vram_mb"],
                    report.get("training", {}).get("peak_vram_mb", 0),
                )
                > cap
            ):
                raise ValueError("VRAM gate failed")
            if not untuned and file_hash(directory / "recipe.py") != recipe_hash:
                raise ValueError("recipe snapshot changed during trial")
            load_contract(path, frozen=True, runtime=True)
            report.update(
                status="evaluated",
                metrics=evaluation["metrics"],
                objective_value=evaluation["metrics"][manifest["objective"]],
                evaluation_hash=file_hash(directory / "evaluation.json"),
            )
        except (Exception, KeyboardInterrupt) as exc:
            status = (
                "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "timeout"
                if isinstance(exc, TimeoutError)
                else "oom"
                if isinstance(exc, MemoryError)
                else "invalid"
                if isinstance(exc, (ValueError, FileNotFoundError))
                else "crash"
            )
            report.update(status=status, error=f"{type(exc).__name__}: {exc}")
        finally:
            report["total_seconds"] = time.monotonic() - started
            write_json(directory / "run.json", report)
            append_result(path, report)
            event(path, {"event": "run_finished", "run_id": identity, "status": report["status"]})
        return report


def checked_report(path, identity):
    directory = resolve_run(path, identity)
    report = read_json(directory / "run.json")
    if report["status"] != "evaluated":
        raise ValueError("only independently evaluated runs can be selected")
    if file_hash(directory / "evaluation.json") != report["evaluation_hash"]:
        raise ValueError("evaluation report changed")
    if not report["untuned"] and file_hash(directory / "recipe.py") != report["recipe_hash"]:
        raise ValueError("recipe snapshot changed")
    evaluation = read_json(directory / "evaluation.json")
    if evaluation["metrics"] != report["metrics"]:
        raise ValueError("run summary metrics differ from evaluator output")
    if tree_hashes(directory / "artifact") != evaluation["artifact_hashes"]:
        raise ValueError("model artifact changed")
    return report


def select(path, identities, baseline=False):
    with campaign_lock(path):
        manifest = load_contract(path, frozen=True, runtime=True)
        if (path / "selection-frozen.json").exists():
            raise ValueError("final-test selection is already frozen")
        candidate = [checked_report(path, x) for x in identities]
        paired_metrics(candidate, manifest)
        state_path = path / "state.json"
        if baseline:
            if state_path.exists():
                raise ValueError("baseline already established")
            state = {"baseline": identities, "incumbent": identities}
            verdict = {"status": "baseline", "runs": identities}
        else:
            state = read_json(state_path)
            incumbent = [checked_report(path, x) for x in state["incumbent"]]
            original = [checked_report(path, x) for x in state["baseline"]]
            verdict = compare(candidate, incumbent, original, manifest)
            verdict["runs"] = identities
            if manifest["averaging"] == "decision":
                verdict["group_bootstrap_seed0"] = bootstrap_delta(
                    read_jsonl(resolve_run(path, identities[0]) / "dev-predictions.jsonl"),
                    read_jsonl(resolve_run(path, state["incumbent"][0]) / "dev-predictions.jsonl"),
                    manifest["task_weights"],
                    "ce" if manifest["profile"] == "distribution" else "hit_at_k",
                )
            if verdict["status"] == "keep":
                state["incumbent"] = identities
        event(path, {"event": "selection", **verdict, "state": state})
        write_json(state_path, state)
        return verdict


def finalize(path):
    with campaign_lock(path):
        manifest = load_contract(path, frozen=True, runtime=True)
        state = read_json(path / "state.json")
        selection = {
            "incumbent": state["incumbent"],
            "baseline": state["baseline"],
            "contract_hash": manifest["contract_hash"],
        }
        frozen_path = path / "selection-frozen.json"
        if frozen_path.exists() and read_json(frozen_path) != selection:
            raise ValueError("final-test selection cannot change")
        write_json(frozen_path, selection)  # fail closed even if evaluation is interrupted
        if (path / "final-report.json").exists():
            return read_json(path / "final-report.json")
        results = {}
        # Never choose a training seed on test: first frozen seed is the export seed.
        for role in ("baseline", "incumbent"):
            identity = state[role][0]
            checked_report(path, identity)
            directory = resolve_run(path, identity)
            if not (directory / "test-evaluation.json").exists():
                child(
                    [
                        sys.executable,
                        "-m",
                        "laya_research.worker",
                        "--campaign",
                        str(path),
                        "--run",
                        str(directory),
                        "--final",
                    ],
                    directory / "test.log",
                    manifest["timeout_seconds"],
                )
            results[role] = read_json(directory / "test-evaluation.json")
        result = {
            "selection": selection,
            "test": results,
            "artifact": str(resolve_run(path, state["incumbent"][0]) / "artifact"),
            "warning": "Test is an audit, not a new selection criterion. Further selection needs a fresh holdout.",
        }
        write_json(path / "final-report.json", result)
        event(path, {"event": "finalized", "runs": state["incumbent"]})
        return result


def recover(path, confirm_stopped=False):
    if not confirm_stopped:
        raise ValueError("confirm all campaign processes have stopped with --confirm-stopped")
    (path / ".writer.lock").unlink(missing_ok=True)
    recovered = []
    with campaign_lock(path):
        # Write-ahead selection events let recovery repair a crash between event and pointer writes.
        if (path / "events.jsonl").exists():
            selections = [
                e
                for e in read_jsonl(path / "events.jsonl")
                if e.get("event") == "selection" and "state" in e
            ]
            if selections:
                write_json(path / "state.json", selections[-1]["state"])
        for file in (path / "runs").glob("*/run.json"):
            report = read_json(file)
            if report["status"] == "running":
                report.update(
                    status="interrupted",
                    error="Recovered unfinished trial; weights are not resumed",
                    total_seconds=load_contract(path)["timeout_seconds"],
                )
                write_json(file, report)
                append_result(path, report)
                recovered.append(report["run_id"])
        event(path, {"event": "recovery", "runs": recovered})
    return {"recovered": recovered}

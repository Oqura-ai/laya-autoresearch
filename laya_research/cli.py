"""Small command-line interface; orchestration remains an external coding agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import contract
from .io import read_json, write_json


def parser():
    p = argparse.ArgumentParser(
        description="Laya dataset preparation and reproducible research campaigns"
    )
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="validate, group-split and freeze a data/objective draft")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--questions", type=Path)
    prep.add_argument("--test", type=Path)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--train-source", choices=["expected", "gold"], default="expected")
    prep.add_argument("--eval-source", choices=["expected", "gold"], default="expected")
    prep.add_argument("--ratios", type=float, nargs=4)
    prep.add_argument("--fractional-scores", action="store_true")
    prep.add_argument("--profile", choices=["distribution", "shortlist"], default="distribution")
    prep.add_argument("--k", type=int, default=1)
    prep.add_argument("--averaging", choices=["decision", "group"], default="decision")
    prep.add_argument("--text-column", default="text")
    prep.add_argument("--label-column", default="label")
    freeze = sub.add_parser(
        "freeze", help="tokenize and bind campaign to local model, environment and CUDA GPU"
    )
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument("--base-checkpoint", type=Path, required=True)
    freeze.add_argument("--budget", type=float, default=300)
    freeze.add_argument("--overshoot", type=float, default=30)
    freeze.add_argument("--timeout", type=float, default=1800)
    freeze.add_argument("--max-len", type=int, default=1024)
    freeze.add_argument("--head-max-len", type=int, default=256)
    freeze.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    freeze.add_argument("--max-trials", type=int, default=20)
    freeze.add_argument(
        "--min-delta", type=float, required=True, help="pilot-derived minimum practical improvement"
    )
    freeze.add_argument(
        "--task-regression", type=float, required=True, help="maximum per-task CE regression"
    )
    freeze.add_argument("--max-vram-mb", type=float)
    freeze.add_argument("--compute-limit", type=float)
    for name, help_text in (
        ("references", "score uniform and train-only prior on dev"),
        ("status", "inspect progress without accessing test metrics"),
        ("finalize", "freeze selection and audit on final test"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--campaign", type=Path, required=True)
    run = sub.add_parser(
        "run", help="snapshot recipe, train, independently evaluate and record one trial"
    )
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--recipe", type=Path, default=contract.ROOT / "train.py")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--hypothesis", required=True)
    run.add_argument(
        "--untuned",
        action="store_true",
        help="recalibrated base-model reference, no gradient updates",
    )
    select = sub.add_parser(
        "select", help="compare matched-seed pair and retain/discard; no Git resets"
    )
    select.add_argument("--campaign", type=Path, required=True)
    select.add_argument("--runs", nargs=2, required=True, metavar=("SEED0_RUN", "SEED1_RUN"))
    select.add_argument("--baseline", action="store_true")
    recover = sub.add_parser(
        "recover", help="recover interrupted trials only AFTER all writer/child processes stop"
    )
    recover.add_argument("--campaign", type=Path, required=True)
    recover.add_argument("--confirm-stopped", action="store_true")
    demo = sub.add_parser("import-demo", help="download a pinned typed-decisions revision")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--revision", required=True)
    sub.add_parser("doctor", help="inspect dependencies and GPU without downloads or training")
    return p


def dispatch(args):
    cmd = args.command
    if cmd == "prepare":
        return contract.prepare(
            args.campaign,
            args.data,
            questions=args.questions,
            test=args.test,
            seed=args.seed,
            train_source=args.train_source,
            eval_source=args.eval_source,
            ratios=args.ratios,
            fractional=args.fractional_scores,
            profile=args.profile,
            k=args.k,
            averaging=args.averaging,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    if cmd == "freeze":
        return contract.freeze(
            args.campaign,
            args.base_checkpoint,
            budget=args.budget,
            overshoot=args.overshoot,
            timeout=args.timeout,
            max_len=args.max_len,
            head_max_len=args.head_max_len,
            seeds=args.seeds,
            max_trials=args.max_trials,
            min_delta=args.min_delta,
            task_regression=args.task_regression,
            max_vram_mb=args.max_vram_mb,
            compute_limit=args.compute_limit,
        )
    if cmd == "references":
        from .baselines import references
        from .campaign import campaign_lock

        with campaign_lock(args.campaign):
            manifest = contract.load_contract(args.campaign)
            if (args.campaign / "selection-frozen.json").exists():
                raise ValueError("campaign is finalized")
            result = references(
                contract.records(args.campaign, "train"),
                contract.records(args.campaign, "dev"),
                manifest,
            )
            write_json(args.campaign / "references.json", result)
        return result
    if cmd == "status":
        manifest = contract.load_contract(args.campaign)
        summaries = [read_json(p) for p in sorted((args.campaign / "runs").glob("*/run.json"))]
        state = args.campaign / "state.json"
        return {
            "status": manifest["status"],
            "contract_hash": manifest["contract_hash"],
            "state": read_json(state) if state.exists() else None,
            "finalized": (args.campaign / "selection-frozen.json").exists(),
            "runs": [
                {k: r.get(k) for k in ("run_id", "status", "seed", "objective_value", "hypothesis")}
                for r in summaries
            ],
        }
    if cmd in ("run", "select", "finalize", "recover"):
        from . import campaign

        if cmd == "run":
            return campaign.run(
                args.campaign, args.recipe, args.seed, args.hypothesis, args.untuned
            )
        if cmd == "select":
            return campaign.select(args.campaign, args.runs, args.baseline)
        if cmd == "finalize":
            return campaign.finalize(args.campaign)
        return campaign.recover(args.campaign, args.confirm_stopped)
    if cmd == "import-demo":
        from .adapters import typed_decisions

        return typed_decisions(args.output, args.revision)
    if cmd == "doctor":
        import importlib.util

        found = {
            x: importlib.util.find_spec(x) is not None
            for x in ("torch", "laya", "transformers", "datasets")
        }
        hw = None
        if found["torch"]:
            from .backend import hardware

            hw = hardware()
        return {
            "dependencies": found,
            "hardware": hw,
            "ready_for_training": bool(found["laya"] and hw and hw["cuda"]),
            "note": "doctor detects availability, not validated GPU integration",
        }
    raise ValueError("unknown command")


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "run" and result["status"] != "evaluated":
            raise SystemExit(1)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        p.exit(2, f"error: {exc}\n")

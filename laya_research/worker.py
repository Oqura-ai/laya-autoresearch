"""Fixed evaluator subprocess; never imports the candidate recipe."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .backend import LayaBackend, hardware
from .calibration import fit_temperatures
from .contract import load_contract, records
from .io import read_json, read_jsonl, tree_hashes, write_json, write_jsonl
from .metrics import evaluate


def evaluate_run(campaign: Path, run: Path, final=False):
    manifest = load_contract(campaign, frozen=True, runtime=True)
    if hardware() != manifest["hardware"]:
        raise ValueError("GPU/runtime differs from the frozen campaign")
    backend = LayaBackend(
        Path(manifest["base_checkpoint"]), manifest["max_len"], manifest["head_max_len"]
    )
    if final:
        report = read_json(run / "evaluation.json")
        if tree_hashes(run / "artifact") != report["artifact_hashes"]:
            raise ValueError("selected artifact changed")
        backend.load_weights(run / "artifact" / "model.safetensors")
        temperatures = report["temperatures"]
        split = "test"
    else:
        context = read_json(run / "context.json")
        backend.load_weights(
            None if context.get("untuned") else run / "candidate" / "model.safetensors"
        )
        calibration = records(campaign, "calibration")
        split = "dev"
        # save_checkpoint quantizes to fp16. Reload BEFORE selection scoring so metrics
        # always describe the exact exported weights, including an untuned fp32 base.
        backend.save(run / "artifact", {kind: 1.0 for kind in ("choice", "score", "noul")})
        backend.model = None
        import gc

        gc.collect()
        backend.torch.cuda.empty_cache()
        backend.load_weights(run / "artifact" / "model.safetensors")
        # Quantization can change logits; refit calibration on the exported weights.
        logits, calibration_timing = backend.predict(
            read_jsonl(campaign / "cache" / "calibration.jsonl")
        )
        temperatures, fit = fit_temperatures(
            calibration,
            logits,
            min_items=manifest["calibration"]["min_items"],
            min_groups=manifest["calibration"]["min_groups"],
        )
        backend.save(run / "artifact", temperatures)
    evaluation = records(campaign, split)
    predictions, timing = backend.predict(read_jsonl(campaign / "cache" / f"{split}.jsonl"))
    metrics, rows = evaluate(
        evaluation,
        predictions,
        manifest["task_weights"],
        temperatures,
        manifest["k"],
        manifest["averaging"],
    )
    report = {
        "contract_hash": manifest["contract_hash"],
        "split": split,
        "metrics": metrics,
        "temperatures": temperatures,
        "timing": timing,
    }
    if final:
        write_json(run / "test-evaluation.json", report)
        write_jsonl(run / "test-predictions.jsonl", rows)
    else:
        report.update(
            calibration=fit,
            calibration_timing=calibration_timing,
            sdk_parity=backend.sdk_parity(run / "artifact", evaluation, predictions, temperatures),
        )
        for name in (
            "manifest.json",
            "schemas.json",
            "input-contract.json",
            "environment.lock.json",
        ):
            shutil.copy2(campaign / name, run / "artifact" / name)
        report["artifact_hashes"] = tree_hashes(run / "artifact")
        write_json(run / "evaluation.json", report)
        write_jsonl(run / "dev-predictions.jsonl", rows)
    load_contract(campaign, frozen=True, runtime=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.campaign, args.run, args.final)

"""Explicit download adapters. No network access in training/evaluation workers."""

from pathlib import Path

from .io import loads, write_jsonl


def typed_decisions(output: Path, revision: str):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("supply the dataset's immutable 40-character Hub commit, not main")
    if output.exists():
        raise ValueError("refusing to overwrite dataset directory")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the demo extra to download typed-decisions") from exc
    collected = {}
    for split in ("train", "test"):
        result = []
        for workflow in (
            "agent_trace_observability",
            "customer_service",
            "invoice_processing",
            "security_incidents",
        ):
            dataset = load_dataset(
                "LocalLLaMA/typed-decisions", workflow, split=split, revision=revision
            )
            for index, raw in enumerate(dataset):
                parsed = {
                    k: loads(raw[k]) if isinstance(raw[k], str) else raw[k]
                    for k in ("state", "questions", "gold")
                }
                original_id = str(raw.get("id", index))
                identity = f"{workflow}/{split}/{original_id}"
                # Group IDs intentionally exclude split so supplied cross-split IDs are detected.
                group = f"{workflow}/{raw.get('group_id', original_id)}"
                if "id" not in raw and "group_id" not in raw:
                    group = identity  # no claim that coincident row numbers are related cases
                result.append(
                    {
                        **parsed,
                        "id": identity,
                        "group_id": group,
                        "workflow": workflow,
                        "metadata": {
                            "label_source": "typed-decisions teacher distribution",
                            "dataset_revision": revision,
                            "language": "en",
                        },
                    }
                )
        collected[split] = result
    output.mkdir(parents=True)
    for split, result in collected.items():
        write_jsonl(output / f"{split}.jsonl", result)
    return {name: len(rows) for name, rows in collected.items()}

"""The only module that knows Laya's low-level model/encoding APIs.

Optional imports stay here so dataset tools and metrics work without a GPU stack.
The evaluator builds architecture/tokenizer from the frozen BASE, not candidate code/config.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

from .io import read_json


def hardware():
    import torch

    cuda = torch.cuda.is_available()
    return {
        "cuda": cuda,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    }


class LayaBackend:
    def __init__(self, base: Path, max_len: int, head_max_len: int, device="cuda"):
        try:
            import torch
            from laya import common, train
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the training extra and a compatible CUDA PyTorch build first"
            ) from exc
        for name in ("to_internal", "make_item", "encode_item", "save_checkpoint", "rlcd_loss"):
            if not hasattr(train, name):
                raise RuntimeError(f"Installed Laya lacks required training API: {name}")
        self.torch, self.common, self.train = torch, common, train
        self.base = Path(base)
        for name in ("rl_agent_config.json", "model.safetensors", "encoder/config.json"):
            if not (self.base / name).is_file():
                raise ValueError(f"incomplete local checkpoint: missing {name}")
        self.cfg = read_json(self.base / "rl_agent_config.json")
        if self.cfg.get("option_layout", "sequential") != "sequential":
            raise ValueError("v1 supports sequential-layout checkpoints only")
        self.cfg.update(max_len=max_len, head_max_len=head_max_len, amp_dtype="fp16")
        self.max_len, self.head_max_len = max_len, head_max_len
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(
            str(self.base / "tokenizer"), local_files_only=True
        )
        self.model = None

    def load_weights(self, weights: Path | None = None):
        from safetensors.torch import load_file

        self.model = self.common.build_model(self.cfg, encoder_dir=str(self.base / "encoder"))
        state = load_file(str(weights or self.base / "model.safetensors"))
        if any(
            not self.torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point()
        ):
            raise ValueError("checkpoint contains nonfinite weights")
        self.model.load_state_dict(state, strict=True)
        self.model.float().to(self.device)
        return self.model

    def encode(self, records):
        encoded, state_cache, stats = (
            [],
            {},
            {"decisions": 0, "truncated": 0, "state_tokens_dropped": 0},
        )
        for r in records:
            if r["case_id"] not in state_cache:
                text = self.common.serialize_state(r["state"]).replace(self.tok.mask_token, " ")
                state_cache[r["case_id"]] = self.common.encode_text(
                    self.tok, text, add_special_tokens=False
                )["input_ids"]
            state_ids = state_cache[r["case_id"]]
            q = self.train.to_internal(r["qid"], r["question"])
            item, reason = self.train.make_item(
                self.tok, q, r["target"], state_ids[: self.max_len], self.head_max_len, self.max_len
            )
            if reason:
                raise ValueError(
                    f"{r['id']}: encoding refused ({reason}); change limits in a new campaign"
                )
            ids, markers, option_stats, truncation = self.common.build_sequence(
                self.tok,
                None,
                q,
                self.max_len,
                self.head_max_len,
                state_ids=state_ids,
                return_stats=True,
                return_truncation_stats=True,
            )
            if len(markers) != len(r["labels"]) or option_stats["options_distinct"] != len(markers):
                raise ValueError(f"{r['id']}: missing or collapsed option markers")
            encoded.append(
                {
                    "id": r["id"],
                    "ids": ids,
                    "markers": markers,
                    "qtype": self.common.QTYPES[r["type"]],
                    "target": r["target"],
                    "item": item,
                    "group_id": r["group_id"],
                }
            )
            stats["decisions"] += 1
            stats["truncated"] += int(truncation["truncated"])
            stats["state_tokens_dropped"] += truncation["state_tokens_dropped"]
        return encoded, stats

    def batch(self, items):
        return self.common.collate_items([items], self.tok.pad_token_id)

    def forward(self, batch, *, amp=False, frozen_encoder=False):
        args = [
            batch[k].to(self.device)
            for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        ]
        with self.torch.autocast(self.device.type, dtype=self.torch.float16, enabled=amp):
            logits, _ = self.model(*args, detach_encoder=frozen_encoder)
        return logits.float()

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, encoded, batch_size=8):
        if self.model is None:
            self.load_weights()
        self.model.eval()
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        # Length sorting reduces padding; predictions remain keyed to immutable IDs.
        ordered = sorted(encoded, key=lambda r: (len(r["ids"]), r["id"]))
        predictions, times = {}, []
        with torch.inference_mode():
            self.forward(self.batch(ordered[: min(batch_size, len(ordered))]))
            self.synchronize()
            start = time.monotonic()
            for offset in range(0, len(ordered), batch_size):
                chunk = ordered[offset : offset + batch_size]
                before = time.monotonic()
                logits = self.forward(self.batch(chunk))
                self.synchronize()
                times.append(time.monotonic() - before)
                for r, z in zip(chunk, logits.cpu().tolist(), strict=True):
                    predictions[r["id"]] = z[: len(r["target"])]
            seconds = time.monotonic() - start
        times.sort()
        return predictions, {
            "seconds": seconds,
            "decisions_per_second": len(encoded) / seconds,
            "batch_latency_p50_ms": 1000 * times[len(times) // 2],
            "batch_latency_p95_ms": 1000 * times[int(0.95 * (len(times) - 1))],
            "batch_size": batch_size,
            "precision": "fp32",
            "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2**20
            if self.device.type == "cuda"
            else 0,
        }

    def save(self, directory, temperatures):
        cfg = dict(self.cfg)
        for key in (
            "temperature_by_options",
            "lang_temperatures",
            "binning_map",
            "abstention_thresholds",
        ):
            cfg.pop(key, None)
        cfg["temperature"] = [temperatures[name] for name in ("choice", "score", "noul")]
        self.train.save_checkpoint(self.model, self.tok, cfg, str(directory))

    def sdk_parity(self, artifact, records, predictions, temperatures):
        """Check saved artifact with the SDK's fp32 path; GPU AMP parity is a separate concern."""
        import math

        import laya

        from .metrics import log_softmax

        # Release evaluator model before loading SDK; never hold two large GPU models.
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        chosen, seen = [], set()
        for r in records:
            key = (r["type"], len(r["labels"]))
            if key not in seen and len(chosen) < 12:
                seen.add(key)
                chosen.append(r)
        errors = []
        with laya.load(str(artifact), device=str(self.device), backend="eager") as agent:
            agent.amp_enabled = False
            for r in chosen:
                answer = agent.system_one(r["state"], {r["qid"]: r["question"]})["answers"][
                    r["qid"]
                ]
                served = (
                    [1 - answer["noul"], answer["noul"]]
                    if r["type"] == "noul"
                    else [answer["probabilities"][key] for key in r["labels"]]
                )
                p = [
                    math.exp(x) for x in log_softmax(predictions[r["id"]], temperatures[r["type"]])
                ]
                errors.append(max(abs(a - b) for a, b in zip(served, p, strict=True)))
        maximum = max(errors)
        if maximum > 1e-3:
            raise ValueError(f"SDK fp32 parity failed: max probability error {maximum}")
        return {
            "decisions": len(errors),
            "max_probability_error": maximum,
            "tolerance": 1e-3,
            "precision": "SDK AMP disabled; serialization rounds to 4 decimals",
        }

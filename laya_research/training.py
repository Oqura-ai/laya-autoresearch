"""Time-budgeted single-GPU reference loop. Editable recipe lives in ../train.py."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import LayaBackend
from .io import read_json, read_jsonl, write_json


@dataclass(frozen=True)
class Recipe:
    loss: str = "rlcd"
    micro_batch: int = 8
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    label_smoothing: float = 0.0
    freeze_encoder: bool = False
    shuffle_choice_options: bool = False
    gradient_checkpointing: bool = True
    amp: bool = True

    def validate(self):
        if self.loss not in ("rlcd", "soft-ce"):
            raise ValueError("loss must be rlcd or soft-ce")
        for name in ("micro_batch", "grad_accum", "rl_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.loss == "rlcd" and self.rl_samples < 2:
            raise ValueError("RLCD requires at least two noise samples for an advantage")
        for name in ("encoder_lr", "head_lr", "min_lr", "grad_clip", "sigma_start", "sigma_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.label_smoothing < 1 or self.min_lr > min(self.encoder_lr, self.head_lr):
            raise ValueError("invalid smoothing or learning-rate floor")
        for name in ("weight_decay", "w_sph", "w_rps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def accumulation_windows(items, micro_batch, grad_accum):
    """Yield actual windows. Partial final windows divide by THEIR example count."""
    size = micro_batch * grad_accum
    for start in range(0, len(items), size):
        window = items[start : start + size]
        yield [window[i : i + micro_batch] for i in range(0, len(window), micro_batch)]


def train(context_path: str, recipe: Recipe):
    recipe.validate()
    context = read_json(Path(context_path))
    output = Path(context["output"])
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("reference training requires one CUDA GPU")
    seed = context["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = random.Random(seed)
    backend = LayaBackend(
        Path(context["base_checkpoint"]), context["max_len"], context["head_max_len"]
    )
    model = backend.load_weights()
    if recipe.freeze_encoder:
        model.encoder.requires_grad_(False)
    elif recipe.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.head_checkpointing = recipe.gradient_checkpointing
    model.train()
    if recipe.freeze_encoder:
        model.encoder.eval()
    groups = []
    for encoder, lr in ((True, recipe.encoder_lr), (False, recipe.head_lr)):
        params = [
            p
            for n, p in model.named_parameters()
            if n.startswith("encoder.") == encoder and p.requires_grad
        ]
        if params:
            groups.append({"params": params, "lr": lr, "initial_lr": lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=recipe.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=recipe.amp)
    items = read_jsonl(Path(context["train_cache"]))
    updates, examples, total_loss, seen_groups = 0, 0, 0.0, set()
    torch.cuda.reset_peak_memory_stats()
    backend.synchronize()
    write_json(output / "training-start.json", {"budget_seconds": context["budget_seconds"]})
    start = time.monotonic()
    while time.monotonic() - start < context["budget_seconds"]:
        order = list(items)
        rng.shuffle(order)
        for chunks in accumulation_windows(order, recipe.micro_batch, recipe.grad_accum):
            elapsed = time.monotonic() - start
            if elapsed >= context["budget_seconds"]:
                break
            progress = min(1.0, elapsed / context["budget_seconds"])
            for group in optimizer.param_groups:
                group["lr"] = (
                    recipe.min_lr
                    + (group["initial_lr"] - recipe.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                )
            sigma = recipe.sigma_start + (recipe.sigma_end - recipe.sigma_start) * progress
            count = sum(len(chunk) for chunk in chunks)
            optimizer.zero_grad(set_to_none=True)
            for chunk in chunks:
                encoded = chunk
                if recipe.shuffle_choice_options:
                    encoded = []
                    for r in chunk:
                        order_options = list(range(len(r["target"])))
                        if r["item"]["q"]["t"] == "choice":
                            rng.shuffle(order_options)
                        encoded.append(
                            backend.train.encode_item(
                                backend.tok,
                                r["item"],
                                context["max_len"],
                                context["head_max_len"],
                                order_options,
                            )
                        )
                batch = backend.batch(encoded)
                logits = backend.forward(
                    batch, amp=recipe.amp, frozen_encoder=recipe.freeze_encoder
                )
                mask, target = (
                    batch["marker_mask"].to(backend.device),
                    batch["target"].to(backend.device),
                )
                target = (
                    1 - recipe.label_smoothing
                ) * target + recipe.label_smoothing * mask / mask.sum(-1, keepdim=True)
                if recipe.loss == "rlcd":
                    loss = backend.train.rlcd_loss(
                        logits,
                        target,
                        mask,
                        batch["qtype"].to(backend.device),
                        sigma,
                        recipe.rl_samples,
                        recipe.w_sph,
                        recipe.w_rps,
                    )
                else:
                    loss = backend.train.soft_ce_loss(logits, target, mask)
                if not torch.isfinite(loss).item():
                    raise ValueError("nonfinite training loss")
                scaler.scale(loss * len(chunk) / count).backward()
                total_loss += loss.item() * len(chunk)
                examples += len(chunk)
                seen_groups.update(r["group_id"] for r in chunk)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            if not torch.isfinite(norm).item():
                raise ValueError("nonfinite training gradients")
            scaler.step(optimizer)
            scaler.update()
            backend.synchronize()
            updates += 1
    seconds = time.monotonic() - start
    if not updates or seconds > context["budget_seconds"] + context["overshoot_seconds"]:
        raise TimeoutError("training exceeded safe update-boundary overshoot or made no updates")
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
        str(output / "model.safetensors"),
    )
    write_json(
        output / "training.json",
        {
            "recipe": asdict(recipe),
            "updates": updates,
            "examples": examples,
            "unique_groups": len(seen_groups),
            "loss": total_loss / examples,
            "train_seconds": seconds,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "seed": seed,
        },
    )

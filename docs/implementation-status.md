# Implementation status

This records what exists, not an assertion that all GPU acceptance checks passed.

Local verification on 7 October 2026: **41 tests passed, 1 real-GPU test skipped**
(`python -m pytest -q`); unittest discovery also passed. Ruff static checks and
format checks passed. The example-data CLI workflow and simulated complete campaign
lifecycle were exercised. No real Laya weights were downloaded or trained.

## Stage 1 — implemented and locally tested

- Strict hard/soft/missing-label handling; explicit train/evaluation source choice.
- JSONL, declared-schema CSV, pinned-revision typed-decisions import adapter.
- Whole-case splitting, transitive source-group/exact-state grouping, official-test
  overlap checks, deterministic assignments and coverage reports.
- Frozen task objectives, input schemas, file fingerprints and data inspection.
- Full-distribution CE/L2, ordinal metrics, hard-label classification, ECE, shortlist
  ranking, task/group aggregation and deterministic calibration tests.

Still empirical: live public-dataset import/schema inspection, label quality,
near-duplicate audit and representative split support. Splitting is group-aware
greedy allocation, not automatic workflow stratification or temporal validation.

## Stage 2 — implementation present; real GPU acceptance pending

- Thin editable recipe plus a modular single-GPU RLCD/soft-CE reference loop.
- Token caches, fixed base initialization, actual-count gradient accumulation,
  elapsed-time schedule, update-boundary budget and subprocess deadline.
- Independent evaluator loads architecture/tokenizer from the fixed base and only
  candidate weights; no candidate evaluation-code imports.
- Export/reload before calibration/scoring, per-type temperature fit, SDK fp32 parity
  gate, artifact checksums, timing and memory tracking.
- Uniform/prior references and recalibrated untuned model path.

Not yet verified here: Laya package installation/API behavior on real tensors,
CUDA training, checkpoint reload, served parity, measured throughput or quality.
This workspace has PyTorch 2.0.1+cpu and no installed Laya. Training requires a
separate compatible CUDA environment with PyTorch >=2.4. PyPI Laya 0.4.0 was
verified available; the package itself was not downloaded or executed locally.

Differences from the original proposal:

- Added an explicit `freeze` after CPU-only preparation. It binds tokenization,
  model files, installed dependency/source fingerprint and actual hardware.
- A platform-specific resolver lock is not fabricated. `environment.lock.json`
  captures the installed environment; generate/test a resolver lock on the pilot GPU.
- Task weights are equal in v1 (with decision or group averaging). Source choice is
  per partition, not per task. More elaborate policies need a new explicit contract.
- Shipped bucket/language calibration is not used as a baseline result; raw and
  our recalibrated base are measured. No claim of reproducing a shipped benchmark.
- Operational diagnostics report warmed model **batch** latency and throughput.
  End-to-end application latency and business-specific gates are future extensions.
- SDK parity samples distinct type/option-count combinations, up to 12 decisions,
  in fp32. It is not exhaustive proof or a claim about default CUDA AMP parity.

## Stage 3 — orchestration tested with deterministic test doubles

- Frozen recipe snapshots, per-run logs/artifacts, authoritative JSON, append-only
  TSV and events, separate baseline/incumbent pointers.
- Matched two-seed selection, original-baseline and incumbent task-regression gates,
  descriptive paired group bootstrap, finite trial/compute caps.
- Failure/timeout/invalid/OOM/interruption statuses with no fake metric values.
- Single-writer lock, explicit interruption recovery, no automatic Git rollback,
  no upload, and no model deletion.
- Final test is a separate selection-freezing action; exports the first frozen
  seed's incumbent without using test for another round of selection.

The first implementation evaluates both seeds for every recipe, rather than adding
a provisional promotion queue. This costs more compute but keeps confirmation
explicit. The measured incumbent seed pair is reused in comparisons; it is not
retrained for every challenger. Bootstrap intervals are currently emitted only for
decision-weighted campaigns and do not remove adaptive model-selection bias.

Full model-driven campaigns remain pending the GPU smoke test and a measured pilot.
GPU location, real task/dataset, acceptable error costs, shortlist k (if applicable)
and authorized compute budget still need the human's choices before real research.

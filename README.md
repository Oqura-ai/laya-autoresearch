# Laya Research

AutoResearch-style workspace for fine-tuning **typed decisions**, not text
generation. Supply labeled cases, freeze an evaluation contract, and let a coding
agent change one training recipe at a time. Every candidate starts from the same
base checkpoint and is evaluated by a separate, fixed process.

**Implementation status:** data preparation, metrics, calibration, supervision,
selection and tracking have local automated tests. Real Laya training/SDK parity
has an opt-in CUDA integration test and still needs to pass on the intended GPU.
There are no measured training-quality or speed claims yet.

This project is self-contained in `laya-autoresearch/`. The original
`../autoresearch/` and fine-tuning notebook in the parent directory are unchanged references.
[build-plan.md](build-plan.md) explains the research design;
[docs/implementation-status.md](docs/implementation-status.md) tracks deviations and remaining checks.

## Start here: no GPU or package installation needed

Python 3.11+ is enough for dataset preparation and the core test suite:

```console
cd laya-autoresearch
python -m unittest discover -s tests -v
python prepare.py doctor
python prepare.py prepare --data examples/support.jsonl --questions examples/questions.json --campaign campaigns/support
python prepare.py references --campaign campaigns/support
python prepare.py status --campaign campaigns/support
```

All subsequent commands run from this directory, not the parent workspace.

The 20 support tickets are an original **plumbing fixture**, not a meaningful
training benchmark. Small calibration partitions explicitly fall back to T=1.
Use representative, independently grouped real examples for research.

Preparation produces `manifest.json`, `data-report.json`, `splits.json`,
`schemas.json`, `input-contract.json` and four normalized JSONL partitions. Inspect
them before starting a GPU campaign. Existing campaigns are never overwritten.

## Dataset contract

One JSON object per line. Either embed `questions` in each case or supply a shared
`--questions questions.json`. See [examples](examples/questions.json).

```json
{"id":"ticket-42","group_id":"thread-17","state":"The invoice is duplicated.","questions":{"team":{"type":"choice","instructions":"Who handles this request?","criteria":{"support":"Product issues","accounts":"Billing"}}},"expected":{"team":"accounts"},"metadata":{"label_source":"human_review"}}
```

- `choice`: string label ID matching declared criteria; `noul`: JSON `true`/`false`;
  `score`: integer index into the ordered rubric. Fractional scores need the explicit
  `--fractional-scores` interpolation policy.
- Soft supervision uses `gold.<question>.probabilities`, with every option key,
  finite nonnegative probabilities, and sum 1 within `1e-4`. For noul use
  `false/true`; for score use `0/1/2/...` strings.
- Select sources explicitly: `--train-source gold --eval-source expected`, for
  example. There is no silent fallback between hard and teacher labels. Missing
  labels are counted and masked, never treated as negatives.
- Use `group_id` for the strongest relevant boundary: conversation, customer,
  template, trajectory, etc. Without it each case ID is its own source group.
  All questions stay with their case. Exact duplicate states union their groups.
  Near-duplicate and temporal leakage still require a dataset audit.
- `workflow` defaults to `default`; a task is `(workflow, question_id)`. All four
  partitions must cover the same tasks. Preparation fails if support is missing.
- Metadata and labels never enter the encoder input. Use only information available
  **before** the decision in `state`. A historical action is not automatically a
  verified correct action.
- CSV supports `id,text,label` plus optional `group_id`, with an explicitly supplied
  **single-choice** schema. Use `--text-column`/`--label-column` for other names.

Custom default split is 65/10/15/10 by independent groups, allocated approximately
by case counts. `--test heldout.jsonl` instead splits the training input 70/10/20,
preserving the external test and rejecting cross-file group/state overlap. Larger
groups may prevent exact ratios. This is deterministic group splitting, not
guaranteed stratification. Do not search split seeds for better model scores.

### Typed-decisions demonstration

On a network-enabled machine, install the `demo` extra and supply an immutable
dataset commit from its Hub history:

```console
python prepare.py import-demo --output data/typed --revision DATASET_COMMIT_SHA
python prepare.py prepare --data data/typed/train.jsonl --test data/typed/test.jsonl --train-source gold --eval-source gold --campaign campaigns/typed
```

The adapter preserves workflow IDs and source provenance. These are teacher
distributions, not independently verified real-world outcomes. The live dataset
download has not been exercised in this workspace.

## GPU setup and campaign freeze

Use a dedicated virtual environment with a compatible **CUDA** PyTorch build.
Then install the project extras:

```console
python -m pip install -e ".[train,dev,demo]"
python prepare.py doctor
```

The optional dependency pins Laya 0.4.0, the
[published version inspected during implementation](https://pypi.org/project/laya/0.4.0/).
The rest of the CUDA stack needs validation on your machine. Do not interpret this
as a tested universal dependency lock. Freeze records exact installed versions,
direct-install provenance, Laya source hashes, model/tokenizer hashes and hardware
in `environment.lock.json` and the manifest; later drift invalidates runs. Generate
and retain a resolver lock on the actual GPU environment before a real campaign.

Download the **base** Laya checkpoint beforehand at an immutable Hub revision,
using `huggingface_hub.snapshot_download(..., revision="COMMIT_SHA")`. Point freeze
at the local directory containing `rl_agent_config.json`, `model.safetensors`,
`encoder/` and `tokenizer/`. No downloads occur inside training/evaluation workers.
Do not use the already typed-decisions-fine-tuned model as a clean benchmark base.

Pilot on a disposable campaign first. Determine a useful time budget, two training
seeds, practical promotion margin and acceptable per-task CE regression. Then
create/freeze the research campaign; the numeric margins below are **examples to
replace with pilot-derived values**, not statistical guarantees:

```console
python prepare.py freeze --campaign campaigns/support --base-checkpoint checkpoints/laya --budget 300 --overshoot 30 --timeout 1800 --max-len 1024 --head-max-len 256 --seeds 0 1 --max-trials 20 --min-delta 0.005 --task-regression 0.05
```

Freeze requires the intended CUDA device, validates required Laya APIs, tokenizes
each partition, refuses collapsed/missing option markers, and reports truncation.
Token limits are fixed **before** encoding. It also locks code and dependencies.
Change infrastructure, schema, model, environment or hardware? Start a new campaign.
Optional `--max-vram-mb` and `--compute-limit` add resource gates. The compute limit
counts recorded trial elapsed time, including failed/confirmation/reference runs;
final auditing is separate.

## Run, compare, finalize

```console
python prepare.py run --campaign campaigns/support --seed 0 --untuned --hypothesis "Recalibrated base reference"
python prepare.py run --campaign campaigns/support --seed 0 --hypothesis "Notebook-style RLCD baseline"
python prepare.py run --campaign campaigns/support --seed 1 --hypothesis "Matched-seed RLCD baseline"
python prepare.py select --campaign campaigns/support --baseline --runs BASE_SEED0_ID BASE_SEED1_ID
```

Edit only [train.py](train.py). For the first ablation set `loss="soft-ce"`, run
both seeds, then compare:

```console
python prepare.py run --campaign campaigns/support --seed 0 --hypothesis "Soft CE only"
python prepare.py run --campaign campaigns/support --seed 1 --hypothesis "Soft CE confirmation"
python prepare.py select --campaign campaigns/support --runs CANDIDATE_SEED0_ID CANDIDATE_SEED1_ID
python prepare.py status --campaign campaigns/support
```

The IDs come from run output. Both runs must have the same recipe hash. Comparison
uses paired means and checks per-task CE against both the incumbent and original
baseline. Keep/discard changes only a checkpoint pointer, never your Git working
tree. Each trial reloads the base; it does not continue the incumbent's training.

`program.md` tells an external coding agent how to repeat this finite loop. There
is no hidden autonomous agent service. Use an isolated Git worktree, preserve user
changes, and do not give test feedback to the research agent.

When research is finished and the human approves the audit:

```console
python prepare.py finalize --campaign campaigns/support
```

Finalization locks selection **before** accessing test, evaluates the original
baseline and selected candidate at the first frozen seed, and writes
`final-report.json`. It does not choose another seed/model based on test or retrain
on dev. The selected `runs/<id>/artifact/` is directly loadable:

```python
import json
import laya

questions = json.load(open("examples/questions.json", encoding="utf-8"))
with laya.load("campaigns/support/runs/SELECTED_ID/artifact", device="cuda") as model:
    model.amp_enabled = False  # reproduce this harness's fixed fp32 evaluator
    result = model.system_one("Our invoice has two identical charges.", questions)
    print(result["answers"])
```

Use your saved task schemas and preprocessing contract, not the support example,
for a real model. Stock SDK CUDA autocast may differ numerically; the included
parity gate checks the explicit fp32 path plus four-decimal probability rounding.

## Modules and invariants

| Module | Responsibility |
|---|---|
| `data.py`, `adapters.py` | Input validation, explicit target sources, group splits, CSV/Hub adapters |
| `contract.py`, `io.py` | Fingerprints, immutable campaign settings, atomic JSON files |
| `metrics.py`, `calibration.py` | Stable CE, diagnostics, group weighting, bounded temperature fitting |
| `backend.py` | Laya model, tokenization, collation, checkpoint and SDK boundary |
| `training.py` | Fixed-budget reference loop and actual-count gradient accumulation |
| `worker.py` | Independent calibration/evaluation of exported weights |
| `campaign.py`, `selection.py` | Child supervision, tracking, paired promotion, recovery/finalization |
| `cli.py` | Thin developer-facing commands |

The default objective is task-balanced **calibrated CE**, in nats. `--averaging
group` first averages repeated decisions within independent groups. Raw CE,
entropy/KL, full-distribution squared error (all types), ordinal RPS, scalar MAE,
hard-label confusion/F1, ECE bin support, teacher argmax agreement, and slices are
reported separately. Unavailable metrics are null with support counts. Soft labels
are never silently relabeled as verified outcomes.

For single-label shortlisting use `--profile shortlist --k 3` at preparation. Every
decision must be a hard-labeled choice with at least k candidates. Optimize
task-balanced hit@k; equal hit@k can use the fixed CE tie-break. Multi-label relevance,
retrieval misses, business costs and threshold fitting need a separately designed
contract; they are not silently approximated by this profile.

Training uses eager single-GPU FP16 (configurable in the recipe), a monotonic budget
including the first update, safe update-boundary overshoot, and actual-example
weighting for partial accumulation windows. Fixed evaluation is fp32, length-sorted
and batched; reported latency is **batch** latency, not full application latency.
Candidate weights are quantized/exported/reloaded before calibration and scoring.

## Inspect and recover

Each run contains the exact recipe, Git provenance where available, context,
training/evaluation logs, metrics, calibration, per-decision dev predictions, timing,
artifact hashes and the model artifact. `results.tsv` is an append-only trial
summary; `events.jsonl` records selection events; `state.json` identifies the
baseline/incumbent. JSON reports are authoritative. Failed objectives are blank/null,
not zero. All checkpoints are retained; nothing is silently pruned or uploaded.

After an interruption, stop any surviving trial processes first:

```console
python prepare.py recover --campaign campaigns/support --confirm-stopped
```

Recovery marks unfinished runs interrupted and conservatively charges their timeout
against compute usage. It does not resume partial weights. A shared-filesystem
coding agent is trusted: hashes and separate evaluation protect against accidental
drift, not a malicious recipe or agent. Keep secrets out of campaign artifacts.

## Developer verification

```console
python -m unittest discover -s tests -v
python -m ruff check laya_research tests prepare.py train.py
python -m ruff format --check laya_research tests prepare.py train.py
```

Set `LAYA_TEST_CHECKPOINT` to a local compatible checkpoint on CUDA, then run
`python -m unittest tests.test_gpu_smoke -v` for real training, calibration, export
and SDK-reload validation. That test is skipped without the explicit checkpoint.
The fake-child lifecycle test validates orchestration only, not model behavior.

The included `.github/workflows/tests.yml` runs when this directory is the root
of its own GitHub repository. GitHub does not discover nested workflows from the
parent workspace repository.

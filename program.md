# Laya research procedure

You are the external research agent. This file is the operating procedure, not an
agent service. Read README.md, the campaign manifest, and data/token reports first.
Run commands from `laya-autoresearch/`, the directory containing this file.
Do not run a campaign without the user's declared GPU/compute budget and task objective.

## Fixed versus editable

- Edit only `train.py` during a frozen campaign. Change one well-described hypothesis
  at a time. The helpers in `laya_research/`, `prepare.py`, this file, dependencies,
  datasets, schemas, calibration, metrics, seeds and resource limits are frozen.
- Use a separate Git worktree for research; preserve unrelated user changes.
  Never run destructive Git resets or delete model artifacts to discard a candidate.
- Never inspect `data/test.jsonl`, `cache/test.jsonl`, or test reports for experiments.
  This is a workflow boundary, not an adversarial filesystem sandbox.
- Do not publish checkpoints, call paid labeling APIs, or change task semantics.

## Establish the campaign

1. The human prepares data and reviews target sources, group separation, label/task
   coverage and truncation. They pilot on the intended GPU, then freeze the budget,
   two seeds, promotion margin, task CE regression tolerance and resource limits.
   Pilot margins must not be inferred from final test.
2. Run `references` for uniform and train-label prior. Run `run --untuned` for a
   recalibrated base-model reference. The reference's `macro_ce_raw` is uncalibrated,
   not the checkpoint's shipped per-bucket calibration.
3. Run the notebook-style RLCD recipe at both manifest seeds. Record the two run IDs,
   in seed order. Establish the initial baseline with `select --baseline --runs A B`.
4. Run the soft-CE-only ablation under the SAME budget, dataset and seeds.

## Each experiment

1. Write a hypothesis and make the intended `train.py` change. Commit only that
   file in the isolated research worktree if Git is being used. Recipe snapshots
   and hashes are authoritative even if there is no commit.
2. Use `python prepare.py run --campaign PATH --seed SEED --hypothesis TEXT`.
   Run BOTH declared seeds with the identical recipe. Trials, including failures,
   untuned references and confirmation seeds, count toward the campaign cap.
3. Inspect `run.json`, `evaluation.json`, `dev-predictions.jsonl` and logs. Failures
   have no objective value. Do not use printed training reward for selection.
4. Call `select --campaign PATH --runs SEED0_RUN SEED1_RUN`. The fixed comparator
   checks paired objectives and per-task CE regressions against BOTH the original
   baseline and incumbent. Exact ties are discarded (shortlist has a fixed CE tie-break).
5. A keep changes the incumbent pointer, not the source file. For a discard, restore
   the desired recipe manually from its recorded snapshot without touching other
   files. Start every trial from the same base model, never from the incumbent weights.

## Failures, stopping and handoff

- A running trial owns `.writer.lock`. Do not start another writer. After a machine
  interruption, stop any remaining parent/child process, inspect logs, and only then
  use `recover --confirm-stopped`. Incomplete weights are not resumed.
- Stay within both trial and total elapsed compute caps. CUDA numerical variation
  and repeated dev selection remain real; bootstrap intervals are descriptive.
- Stop at the agreed limit. Ask the human to approve finalization; then `finalize`
  freezes selection and audits the original baseline and incumbent's FIRST declared
  seed on test. Do not pick a new recipe or seed based on that result.
- Deliver the final report, selected recipe, artifact directory, model/input contract,
  and caveats. No automatic retraining on dev, uploads, or checkpoint pruning.

"""The ONLY editable file during a campaign. Run through prepare.py, not directly.

Hypothesis examples: loss='soft-ce'; lower encoder_lr; freeze_encoder=True;
shuffle_choice_options=True. Keep architecture, data, seeds and budget unchanged.
"""

import argparse

from laya_research.training import Recipe, train

RECIPE = Recipe(
    loss="rlcd",
    micro_batch=8,
    grad_accum=8,
    encoder_lr=2.5e-5,
    head_lr=1e-4,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    train(args.context, RECIPE)

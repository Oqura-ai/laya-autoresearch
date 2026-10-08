import unittest

from laya_research.training import Recipe, accumulation_windows


class TrainingTests(unittest.TestCase):
    def test_actual_partial_windows(self):
        windows = list(accumulation_windows(list(range(13)), 3, 2))
        self.assertEqual([sum(map(len, w)) for w in windows], [6, 6, 1])
        self.assertEqual([len(chunk) for w in windows for chunk in w], [3, 3, 3, 3, 1])

    def test_bad_recipe(self):
        for kwargs in (
            {"micro_batch": 0},
            {"grad_accum": True},
            {"loss": "accuracy"},
            {"encoder_lr": float("nan")},
            {"label_smoothing": 1},
        ):
            with self.assertRaises(ValueError):
                Recipe(**kwargs).validate()

    def test_partial_accumulation_matches_full_batch_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch not installed")
        x, y = torch.arange(5, dtype=torch.float32), torch.arange(5, dtype=torch.float32) * 2
        full, accumulated = (
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(1.0, requires_grad=True),
        )
        ((full * x - y) ** 2).mean().backward()
        chunks = next(accumulation_windows(list(range(5)), 3, 8))
        for indices in chunks:
            loss = ((accumulated * x[indices] - y[indices]) ** 2).mean()
            (loss * len(indices) / 5).backward()
        self.assertAlmostEqual(full.grad.item(), accumulated.grad.item(), places=5)

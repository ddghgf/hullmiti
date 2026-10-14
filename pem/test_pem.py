from __future__ import annotations

import unittest

import torch

from pem.core import modulate_embeddings, neighbor_similarity, patch_credibility
from pem.controller import PEMConfig


class PEMTests(unittest.TestCase):
    def test_default_config_is_disabled(self):
        self.assertFalse(PEMConfig().enabled)

    def test_true_grid_neighbors(self):
        features = torch.zeros(1, 4, 2)
        features[0, :, 0] = torch.tensor([1.0, 1.0, -1.0, -1.0])
        similarity = neighbor_similarity(features, 2, 2)
        self.assertEqual(similarity.shape, (1, 4))
        self.assertTrue(torch.isfinite(similarity).all())

    def test_grid_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            neighbor_similarity(torch.randn(1, 5, 3), 2, 3)

    def test_credibility_range_and_no_mutation(self):
        features = torch.randn(2, 16, 8)
        saved = features.clone()
        credibility, stats = patch_credibility(features, 4, 4)
        self.assertTrue(torch.equal(features, saved))
        self.assertTrue(((credibility >= 0) & (credibility <= 1)).all())
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))

    def test_strength_zero_exact_identity(self):
        projected = torch.randn(1, 9, 12, dtype=torch.float16)
        output, _ = modulate_embeddings(projected, torch.rand(1, 9), strength=0, risk_bias=0.25)
        self.assertTrue(torch.equal(output, projected))

    def test_full_credibility_identity(self):
        projected = torch.randn(1, 9, 12)
        output, _ = modulate_embeddings(projected, torch.ones(1, 9), strength=1, risk_bias=1)
        self.assertTrue(torch.equal(output, projected))

    def test_low_credibility_moves_to_global(self):
        projected = torch.randn(1, 9, 12)
        global_embedding = projected.mean(dim=1, keepdim=True).expand_as(projected)
        output, _ = modulate_embeddings(projected, torch.zeros(1, 9), strength=1, risk_bias=0)
        self.assertTrue(torch.allclose(output, global_embedding, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

"""Tests for turboquant_codebook: 4-bit codebook generation (Lloyd-Max and K-means)."""

import unittest
import numpy as np
from turboquant_codebook import generate_turboquant_lloyd_max, generate_turboquant_kmeans


class _TestTurboQuant4bitBase:
    """Shared tests for 4-bit TurboQuant codebook generation.

    Subclasses must set cls.config in setUpClass and define expected values.
    """

    # ── Structure tests ──────────────────────────────────────────────

    def test_codebook_length(self):
        """4-bit => 16 levels in the codebook."""
        self.assertEqual(len(self.config["int8_codebook"]), 16)

    def test_codebook_symmetry(self):
        """First 8 entries are positive; last 8 are their negations."""
        cb = self.config["int8_codebook"]
        for i in range(8):
            self.assertEqual(cb[i], -cb[i + 8],
                             f"codebook[{i}] and codebook[{i+8}] should be negations")

    def test_codebook_positive_ascending(self):
        """Positive half of codebook should be strictly ascending."""
        pos = self.config["int8_codebook"][:8]
        for i in range(len(pos) - 1):
            self.assertLess(pos[i], pos[i + 1])

    def test_codebook_max_is_127(self):
        """Largest positive codebook entry must be 127 (int8 max)."""
        self.assertEqual(self.config["int8_codebook"][7], 127)

    def test_pos_boundaries_length(self):
        """Should have half_levels - 1 = 7 positive boundaries."""
        self.assertEqual(len(self.config["pos_boundaries"]), 7)

    def test_pos_boundaries_ascending(self):
        """Positive boundaries should be strictly ascending."""
        bounds = self.config["pos_boundaries"]
        for i in range(len(bounds) - 1):
            self.assertLess(bounds[i], bounds[i + 1])

    # ── Value tests ──────────────────────────────────────────────────

    def test_int8_codebook_values(self):
        """Match the expected codebook."""
        expected = [6, 18, 31, 44, 58, 75, 96, 127,
                    -6, -18, -31, -44, -58, -75, -96, -127]
        self.assertEqual(self.config["int8_codebook"], expected)

    def test_pos_boundaries_values(self):
        """Match the expected positive boundaries."""
        expected = [0.2583, 0.5225, 0.7997, 1.0995, 1.4374, 1.8438, 2.4011]
        for got, want in zip(self.config["pos_boundaries"], expected):
            self.assertAlmostEqual(got, want, places=3)

    def test_int8_scale(self):
        """int8_scale should be close to documented 46.472x."""
        self.assertAlmostEqual(self.config["int8_scale"], 46.472, delta=0.01)

    # ── Sanity / invariant tests ─────────────────────────────────────

    def test_debias_factor_greater_than_one(self):
        """Debias factor should be >= 1 (quantization always adds noise)."""
        self.assertGreaterEqual(self.config["debias_factor"], 1.0)

    def test_squared_centroids_close_to_raw(self):
        """squared_centroids_with_debias[i] should be close to squared_centroids[i] (within 5%)."""
        for p, r in zip(self.config["comparison"]["squared_centroids_with_debias"],
                        self.config["comparison"]["squared_centroids"]):
            self.assertAlmostEqual(p, r, delta=r * 0.05)

    def test_all_codebook_entries_fit_int8(self):
        """Every codebook entry must fit in signed int8 range [-128, 127]."""
        for val in self.config["int8_codebook"]:
            self.assertGreaterEqual(val, -128)
            self.assertLessEqual(val, 127)

    def test_squared_centroids_values(self):
        """squared_centroids = q_i^2 for the positive codebook entries."""
        expected = [36, 324, 961, 1936, 3364, 5625, 9216, 16129]
        for got, want in zip(self.config["comparison"]["squared_centroids"], expected):
            self.assertAlmostEqual(got, want, delta=0.01)


class TestLloydMax4bit(_TestTurboQuant4bitBase, unittest.TestCase):
    """Test 4-bit codebook via Lloyd-Max (numerical integration)."""

    @classmethod
    def setUpClass(cls):
        cls.config = generate_turboquant_lloyd_max(bits=4)

    def test_debias_factor(self):
        """Debias factor for Lloyd-Max."""
        self.assertAlmostEqual(self.config["debias_factor"], 1.00478, places=3)

    def test_squared_centroids_with_debias_values(self):
        """squared_centroids_with_debias for Lloyd-Max."""
        expected = [35.64, 323.15, 941.89, 1918.15, 3370.59, 5613.64, 9187.77, 16052.2]
        for got, want in zip(self.config["comparison"]["squared_centroids_with_debias"], expected):
            self.assertAlmostEqual(got, want, delta=2.0)

    def test_zz_print_summary(self):
        """Print Lloyd-Max codebook."""
        c = self.config
        print("\n" + "=" * 60)
        print("TurboQuant 4-bit Codebook (Lloyd-Max)")
        print("=" * 60)
        print(f"  int8_codebook:   {c['int8_codebook']}")
        print(f"  pos_boundaries:  {[round(b, 7) for b in c['pos_boundaries']]}")
        print(f"  debias_factor:   {c['debias_factor']:.8f}")
        print(f"  int8_scale:      {c['int8_scale']:.4f}")
        print(f"  squared_centroids:             {[int(r) for r in c['comparison']['squared_centroids']]}")
        print(f"  squared_centroids_with_debias: {[round(p, 2) for p in c['comparison']['squared_centroids_with_debias']]}")
        print("=" * 60)


class TestKmeans4bit(_TestTurboQuant4bitBase, unittest.TestCase):
    """Test 4-bit codebook via K-means (analytical PDF/CDF)."""

    @classmethod
    def setUpClass(cls):
        cls.config = generate_turboquant_kmeans(bits=4)

    def test_debias_factor(self):
        """Debias factor for K-means."""
        self.assertAlmostEqual(self.config["debias_factor"], 1.00478463, places=5)

    def test_squared_centroids_with_debias_values(self):
        """squared_centroids_with_debias for K-means."""
        expected = [35.64, 323.15, 941.89, 1918.15, 3370.59, 5613.64, 9187.77, 16052.2]
        for got, want in zip(self.config["comparison"]["squared_centroids_with_debias"], expected):
            self.assertAlmostEqual(got, want, delta=1.0)

    def test_zz_print_summary(self):
        """Print K-means codebook."""
        c = self.config
        print("\n" + "=" * 60)
        print("TurboQuant 4-bit Codebook (K-means)")
        print("=" * 60)
        print(f"  int8_codebook:   {c['int8_codebook']}")
        print(f"  pos_boundaries:  {[round(b, 7) for b in c['pos_boundaries']]}")
        print(f"  debias_factor:   {c['debias_factor']:.8f}")
        print(f"  int8_scale:      {c['int8_scale']:.4f}")
        print(f"  squared_centroids:             {[int(r) for r in c['comparison']['squared_centroids']]}")
        print(f"  squared_centroids_with_debias: {[round(p, 2) for p in c['comparison']['squared_centroids_with_debias']]}")
        print("=" * 60)


class TestBothAlgorithmsAgree(unittest.TestCase):
    """Verify Lloyd-Max and K-means produce the same int8 codebook."""

    @classmethod
    def setUpClass(cls):
        cls.lloyd_max = generate_turboquant_lloyd_max(bits=4)
        cls.kmeans = generate_turboquant_kmeans(bits=4)

    def test_same_codebook(self):
        """Both algorithms should produce identical int8 codebooks."""
        self.assertEqual(self.lloyd_max["int8_codebook"],
                         self.kmeans["int8_codebook"])

    def test_same_boundaries(self):
        """Boundaries should agree to 3 decimal places."""
        for lm, km in zip(self.lloyd_max["pos_boundaries"],
                          self.kmeans["pos_boundaries"]):
            self.assertAlmostEqual(lm, km, places=3)

    def test_debias_close(self):
        """Debias factors should be close."""
        self.assertAlmostEqual(self.lloyd_max["debias_factor"],
                               self.kmeans["debias_factor"], places=3)

    def test_zz_print_diff(self):
        """Print both codebooks to visually compare differences, if any."""
        lm = self.lloyd_max
        km = self.kmeans

        print("\n" + "=" * 60)
        print("TurboQuant 4-bit Codebook Comparison")
        print("=" * 60)
        
        # 1. Codebooks
        print(f"[int8_codebook]")
        print(f"  LM: {lm['int8_codebook']}")
        print(f"  KM: {km['int8_codebook']}")
        diffs = [f"Idx {i}: LM={l}, KM={k}" for i, (l, k) in enumerate(zip(lm['int8_codebook'], km['int8_codebook'])) if l != k]
        print(f"  Codebook Diff: {len(diffs)} items" + ("" if not diffs else f" -> {diffs}"))
        print()

        # 2. Boundaries
        print(f"[pos_boundaries]")
        print(f"  LM: {[round(b, 7) for b in lm['pos_boundaries']]}")
        print(f"  KM: {[round(b, 7) for b in km['pos_boundaries']]}")
        max_b_diff = max(abs(l - k) for l, k in zip(lm['pos_boundaries'], km['pos_boundaries']))
        print(f"  Max Diff: {max_b_diff:.6f}")
        print()
        
        # 3. Debias Factor
        print(f"[debias_factor]")
        print(f"  LM: {lm['debias_factor']:.8f}")
        print(f"  KM: {km['debias_factor']:.8f}")
        print(f"  Diff: {abs(lm['debias_factor'] - km['debias_factor']):.8f}")
        print()

        # 4. Int8 Scale
        print(f"[int8_scale]")
        print(f"  LM: {lm['int8_scale']:.8f}")
        print(f"  KM: {km['int8_scale']:.8f}")
        print(f"  Diff: {abs(lm['int8_scale'] - km['int8_scale']):.8f}")
        print()
        
        # 5. Pseudo Sqr Cen
        print(f"[squared_centroids_with_debias]")
        print(f"  LM: {[round(p, 2) for p in lm['comparison']['squared_centroids_with_debias']]}")
        print(f"  KM: {[round(p, 2) for p in km['comparison']['squared_centroids_with_debias']]}")
        max_p_diff = max(abs(l - k) for l, k in zip(lm['comparison']['squared_centroids_with_debias'], km['comparison']['squared_centroids_with_debias']))
        print(f"  Max Diff: {max_p_diff:.6f}")
        
        print("=" * 60)

if __name__ == "__main__":
    unittest.main()

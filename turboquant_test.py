import unittest
import numpy as np

from turboquant import TurboQuant, HadamardRotation

class TestHadamardRotation(unittest.TestCase):
    def test_power_of_2_no_mixing(self):
        """Test standard FWHT on a power of 2 d=8."""
        d = 8
        rot = HadamardRotation(d, use_signs=False)
        self.assertEqual(rot.r, 1)
        self.assertEqual(rot.two_k, 8)
        self.assertIsNone(rot.M)
        
        # Test 1D array
        x = np.array([1.0] * 8)
        y = rot(x)
        self.assertEqual(y.shape, (8,))
        # The sum of all ones should be 8 on the first component and 0 elsewhere
        self.assertAlmostEqual(y[0], 8.0)
        for i in range(1, 8):
            self.assertAlmostEqual(y[i], 0.0)

    def test_odd_dimension(self):
        """Test on an odd dimension where FWHT is identity and mixing matrix does everything."""
        d = 5
        rot = HadamardRotation(d, seed=123)
        self.assertEqual(rot.r, 5)
        self.assertEqual(rot.two_k, 1)
        self.assertIsNotNone(rot.M)
        self.assertEqual(rot.M.shape, (5, 5))
        
        # Test norm preservation
        x = np.random.randn(5)
        norm_x = np.linalg.norm(x)
        y = rot(x)
        norm_y = np.linalg.norm(y)
        # Because we used a pure orthogonal matrix, the norm should be preserved exactly!
        # (Unlike FWHT which scales the norm by sqrt(2^k))
        self.assertAlmostEqual(norm_x, norm_y)

    def test_composite_dimension(self):
        """Test on dimension d = r * 2^k, e.g., 12 = 3 * 4."""
        d = 12
        rot = HadamardRotation(d, seed=42)
        self.assertEqual(rot.r, 3)
        self.assertEqual(rot.two_k, 4)
        
        x = np.random.randn(12)
        norm_x = np.linalg.norm(x)
        y = rot(x)
        norm_y = np.linalg.norm(y)
        
        # Norm is scaled by sqrt(2^k) = sqrt(4) = 2.0
        self.assertAlmostEqual(norm_y, norm_x * 2.0)

    def test_batching(self):
        """Test that rotation supports multidimensional batch arrays."""
        d = 12
        rot = HadamardRotation(d, seed=42)
        X = np.random.randn(5, 12)
        Y = rot(X)
        self.assertEqual(Y.shape, (5, 12))
        
        # Compare with single element mapping
        for i in range(5):
            np.testing.assert_allclose(Y[i], rot(X[i]))

class TestTurboQuant(unittest.TestCase):
    def test_docstring_example(self):
        """
        Mock the exact mathematical example from the docstring:
        U = [2.0, 4.0, -2.0, 0.0]
        V = [7.0, -1.0, 0.0, 1.0] 
        FWHT should yield:
        U_rot = [4.0, -4.0, 8.0, 0.0]  (or a permutation with the same norms)
        V_rot = [7.0, 7.0, 5.0, 9.0]
        """
        tq = TurboQuant(d=4, use_signs=False)
        
        U = np.array([2.0, 4.0, -2.0, 0.0])
        V = np.array([7.0, -1.0, 0.0, 1.0])
        
        # Test rotation values
        U_rot = tq.rotation(U)
        V_rot = tq.rotation(V)
        
        np.testing.assert_allclose(np.abs(U_rot), np.abs([4.0, -4.0, 8.0, 0.0]), rtol=1e-5)
        np.testing.assert_allclose(np.abs(V_rot), np.abs([7.0, 7.0, 5.0, 9.0]), rtol=1e-5)
        
        tq.add_base_embeddings(U)
        # Check integer quantization is identical to docstring 
        # (It maps onto indices which correspond to [44, -44, 75, 6] up to permutation/signs)
        np.testing.assert_allclose(np.sort(np.abs(tq.U_int8[0])), [6, 44, 44, 75])
        
        # Check Scale_U is around 0.05030 (based on 4.8990 / sqrt(9485.58))
        np.testing.assert_allclose(tq.Scale_U[0], 0.05030, rtol=1e-3)
        
        score = tq.score(V)
        # Expected Final Score
        np.testing.assert_allclose(score, 10.825, rtol=0.01)

    def test_scoring_accuracy_batch(self):
        """Test accuracy using a batch of real embeddings."""
        d = 128
        N = 1000
        Q = 10
        # Create normal distributions
        np.random.seed(42)
        U = np.random.randn(N, d)
        V = np.random.randn(Q, d)
        
        # Exact dot products
        exact_scores = np.dot(V, U.T)
        
        # TurboQuant estimation
        tq = TurboQuant(d=d, seed=123)
        tq.add_base_embeddings(U)
        tq_scores = tq.score(V)
        
        self.assertEqual(tq_scores.shape, (Q, N))
        
        # Ensure Mean Absolute Error is small
        mae = np.mean(np.abs(exact_scores - tq_scores))
        
        # Typical exact scores scale as variance ~ d. (For normal(0,1), mean=0, std=sqrt(d)=11.3)
        # So we expect roughly less than 1.0 error for 4-bit config on length 128.
        self.assertLess(mae, 1.0)
        
        # Rank correlation sanity check (top 5 matched for the first query)
        exact_top5 = np.argsort(-exact_scores[0])[:5]
        tq_top5 = np.argsort(-tq_scores[0])[:5]
        
        # Should have good overlap in the top results
        overlap = len(set(exact_top5).intersection(set(tq_top5)))
        self.assertGreaterEqual(overlap, 1)

    def test_compare_all_distributions(self):
        """Compare relative error of 1 query vs 1000 128-d embeddings across distributions."""
        np.random.seed(42)
        d = 128
        N = 1000
        Q = 1
        
        distributions = {
            "Normal N(0,1)": (
                np.random.randn(N, d), 
                np.random.randn(Q, d)
            ),
            "Normal N(5,2)": (
                np.random.randn(N, d)*2.0 + 5.0, 
                np.random.randn(Q, d)*2.0 + 5.0
            ),
            "Uniform U(1,5)": (
                np.random.uniform(1.0, 5.0, size=(N, d)), 
                np.random.uniform(1.0, 5.0, size=(Q, d))
            ),
            "Exponential Exp(1)": (
                np.random.exponential(1.0, size=(N, d)), 
                np.random.exponential(1.0, size=(Q, d))
            )
        }
        
        print("\n" + "="*65)
        print(f"{'Distribution':<20} | {'MAE':<10} | {'Cosine Sim':<12} | {'Mean Rel Err':<12}")
        print("-" * 65)
        
        for name, (U, V) in distributions.items():
            U = U.astype(np.float32)
            # Take the exact 1D query array to explicitly pass a single vector natively
            V_1d = V[0].astype(np.float32)
            
            # The resulting dot product is exactly shape (1000,) representing all 1000 base embedding scores
            exact = np.dot(V_1d, U.T)
            
            tq = TurboQuant(d=d, seed=123)
            tq.add_base_embeddings(U)
            tq_scores = tq.score(V_1d)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(exact - tq_scores))
            
            # Cosine similarity (Ranking Correlation)
            cos_sim = np.dot(exact, tq_scores) / (np.linalg.norm(exact) * np.linalg.norm(tq_scores) + 1e-12)
            
            # Relative error = |exact - tq| / |exact|
            safe_exact = np.where(np.abs(exact) < 1e-6, 1e-6, exact)
            rel_error = np.abs(exact - tq_scores) / np.abs(safe_exact)
            mre = np.mean(rel_error)
            
            print(f"{name:<20} | {mae:<10.4f} | {cos_sim:<12.4f} | {mre:>12.2%}")
            
        print("="*65)
        print("Note on Metrics:")
        print(" - MAE (Mean Absolute Error): The raw mathematical difference between exact and quantized dot products.")
        print(" - Cosine Sim: Measures how perfectly the relative Top-K sorting order is preserved (1.0 = flawless).")
        print(" - Mean Rel Err: Percentage delta. Unreliable / artificially massive for N(0,1) due to dividing near zero.")
        print("="*65 + "\n")

if __name__ == '__main__':
    unittest.main()

import time
import argparse
import numpy as np
import turboquant
from turboquant import TurboQuant

def run_bench(d, N, Q, label):
    print(f"\n--- Benchmarking {label} (d={d}, N={N}, Q={Q}) ---")
    U = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(Q, d).astype(np.float32)
    
    tq = TurboQuant(d=d, seed=42)
    
    # 1. Profile add_base_embeddings
    start = time.perf_counter()
    tq.add_base_embeddings(U)
    t_add = time.perf_counter() - start
    print(f"add_base_embeddings: {t_add:.4f}s")
    
    # 2. Profile score (batched)
    start = time.perf_counter()
    _ = tq.score(V)
    t_score_batch = time.perf_counter() - start
    print(f"score (batched Q={Q}): {t_score_batch:.4f}s")
    
    # 3. Profile score (single query)
    start = time.perf_counter()
    for i in range(Q):
        _ = tq.score(V[i])
    t_score_single = time.perf_counter() - start
    print(f"score (single query x {Q}): {t_score_single:.4f}s")
    
    return t_add, t_score_batch, t_score_single

def profile_turboquant(d=1024, N=10000, Q=100):
    # 1. Run with C++ if available
    has_cpp = turboquant.cpp_fwht_inplace is not None
    if has_cpp:
        res_cpp = run_bench(d, N, Q, "C++ Core")
        
        # 2. Force NumPy fallback
        orig_cpp = turboquant.cpp_fwht_inplace
        turboquant.cpp_fwht_inplace = None
        res_numpy = run_bench(d, N, Q, "NumPy Fallback")
        turboquant.cpp_fwht_inplace = orig_cpp
        
        # 3. Comparison
        print("\n" + "="*40)
        print(f"{'Operation':<20} | {'NumPy':<10} | {'C++':<10} | {'Speedup':<8}")
        print("-" * 55)
        ops = ["Add Base", "Score Batch", "Score Single"]
        for i, op in enumerate(ops):
            speedup = res_numpy[i] / res_cpp[i]
            print(f"{op:<20} | {res_numpy[i]:.4f}s   | {res_cpp[i]:.4f}s   | {speedup:.2f}x")
        print("="*40)
    else:
        print("C++ extension not found. Running NumPy only.")
        run_bench(d, N, Q, "NumPy Only")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile TurboQuant performance")
    parser.add_argument("-d", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("-N", type=int, default=10000, help="Number of base embeddings")
    parser.add_argument("-Q", type=int, default=100, help="Number of queries")
    args = parser.parse_args()

    profile_turboquant(d=args.d, N=args.N, Q=args.Q)

import time
import numpy as np
from turboquant import TurboQuant

def profile_turboquant(d=1024, N=10000, Q=100):
    print(f"Profiling TurboQuant with d={d}, N={N}, Q={Q}")
    
    U = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(Q, d).astype(np.float32)
    
    tq = TurboQuant(d=d, seed=42)
    
    # 1. Profile add_base_embeddings
    start_time = time.perf_counter()
    tq.add_base_embeddings(U)
    end_time = time.perf_counter()
    print(f"add_base_embeddings took: {end_time - start_time:.4f} seconds")
    
    # 2. Profile score (batched)
    start_time = time.perf_counter()
    tq_scores = tq.score(V)
    end_time = time.perf_counter()
    print(f"score (batched Q={Q}) took: {end_time - start_time:.4f} seconds")
    
    # 3. Profile score (single query)
    start_time = time.perf_counter()
    for i in range(Q):
        _ = tq.score(V[i])
    end_time = time.perf_counter()
    print(f"score (single query x {Q}) took: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    profile_turboquant()

import numpy as np

try:
    from fast_hadamard import (
        fwht_inplace as cpp_fwht_inplace,
        compute_dot_products as cpp_compute_dot_products
    )
except ImportError:
    cpp_fwht_inplace = None
    cpp_compute_dot_products = None

# -----------------------------------------------------------------------------
# 4-bit TurboQuant Codebook parameters 
# Pre-computed analytically using K-means on the standard normal distribution
# -----------------------------------------------------------------------------
INT8_CODEBOOK = np.array([6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127], dtype=np.int8)
POS_BOUNDARIES = np.array([0.2582925, 0.5225402, 0.799742, 1.0995197, 1.4373973, 1.8437953, 2.4010493], dtype=np.float64)

# Mathematical centroid energies (q^2)
SQUARED_CENTROIDS = np.array([36, 324, 961, 1936, 3364, 5625, 9216, 16129], dtype=np.float64)

# Pre-baked energy terms using analytical formulas (q * E[z])
SQUARED_CENTROIDS_WITH_DEBIAS = np.array([35.64, 323.14, 941.89, 1918.14, 3370.57, 5613.62, 9187.76, 16052.2], dtype=np.float64)


class HadamardRotation:
    """
    Applies an O(d log d) unnormalized random rotation using the Fast 
    Walsh-Hadamard Transform (FWHT), avoiding an explicit d x d matrix.
    
    The rotation is a key component of TurboQuant. It spreads the information 
    across all dimensions, making the data more Gaussian and reducing the 
    clipping error during quantization.
    
    If the dimension `d` is not a power of 2, it decomposes `d = r * 2^k` where
    `r` is an odd integer. It applies the FWHT to the size 2^k subspaces and 
    then mixes these subspaces with a randomly generated `r x r` orthogonal matrix.
    """
    def __init__(self, d: int, seed: int = 42, use_signs: bool = True):
        self.d = d
        # Decompose d = r * 2^k for structured rotation
        r = d
        while r % 2 == 0 and r > 1:
            r //= 2
        self.r = r
        self.two_k = d // r
        
        # Prepare the r x r mixing matrix M for arbitrary dimensions
        rng = np.random.RandomState(seed)
        if self.r > 1:
            # Generate a random orthogonal matrix M via QR decomposition
            A = rng.randn(self.r, self.r)
            q, _ = np.linalg.qr(A)
            self.M = q.astype(np.float64)
        else:
            self.M = None
            
        # Optional random diagonal sign flipping (D) to ensure the distribution 
        # is zero-centered and symmetric, satisfying the Central Limit Theorem.
        if use_signs:
            self.signs = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
        else:
            self.signs = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the structured rotation y = M * H * D * x.
        H is the Hadamard matrix block, D is the diagonal sign matrix.
        """
        if self.d == 1:
            return x.copy()
            
        original_shape = x.shape
        if original_shape[-1] != self.d:
            raise ValueError(f"Expected last dimension {self.d}, got {original_shape[-1]}")
        
        # 1. Apply random signs (D) to decorrelate input dimensions
        if self.signs is not None:
            out = x * self.signs 
        else:
            out = x.astype(np.float64, copy=True)
        
        # 2. Fast Walsh-Hadamard Transform (H) on the 2^k dimension
        # Reshape to (..., r, 2^k) to process Hadamard blocks
        out_view = out.reshape(original_shape[:-1] + (self.r, self.two_k))
        
        n = self.two_k
        k = n.bit_length() - 1
        if k > 0:
            if cpp_fwht_inplace is not None:
                # Use high-performance C++ implementation to transform the 2^k subspace in-place
                cpp_fwht_inplace(out_view)
            else:
                # Iterative FWHT implementation with O(d log d) complexity (NumPy fallback)
                for i in range(k):
                    s = 1 << i
                    # View as pairs of blocks to perform butterfly operations
                    out_split = out_view.reshape(out_view.shape[:-1] + (n // (s << 1), 2, s))
                    
                    # Use a temp for half the data to avoid overwriting during in-place calculation
                    tmp = out_split[..., 0, :].copy()
                    out_split[..., 0, :] += out_split[..., 1, :]
                    out_split[..., 1, :] *= -1.0
                    out_split[..., 1, :] += tmp
                
        # 3. Mix across the r-subspaces using orthogonal matrix M
        if self.r > 1:
            # Contract M over the r-dimension: out = einsum(M_ij, out_jk)
            out = np.einsum('ij, ...jk -> ...ik', self.M, out_view).reshape(original_shape)
            
        return out


class TurboQuant:
    """
    TurboQuant Engine: High-performance 4-bit Asymmetric Quantization.
    
    It quantizes base embeddings (U) into 4-bit integers and stores them.
    Scoring involves quantizing the query (V) to 8-bit and performing 
    accelerated integer dot products with analytical debiasing.
    """
    def __init__(self, d: int, seed: int = 42, use_signs: bool = True):
        self.d = d
        self.sqrt_d = np.sqrt(d)
        self.rotation = HadamardRotation(d, seed, use_signs=use_signs)
        self.U_int8 = None
        self.U_int8_f64_T = None # Cached for high-speed scoring
        self.Scale_U = None
        
    def add_base_embeddings(self, U: np.ndarray):
        """
        Pre-process and quantize base embeddings U (N, d).
        Stored as int8 for memory efficiency and fast scoring.
        """
        U = np.asarray(U, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(1, -1)
            
        # 1. Capture original L2 norms for final score reconstruction
        Norm_U = np.linalg.norm(U, axis=-1)
        Norm_U[Norm_U < 1e-12] = 1e-12
        
        # 2. Rotate to Gaussian space (O(d log d))
        U_rot = self.rotation(U)
        
        # 3. Project onto unit sphere (L2 normalization) and scale by sqrt(d)
        # This standardizes the distribution to match our pre-computed codebook.
        Norm_U_rot = np.linalg.norm(U_rot, axis=-1)
        Norm_U_rot[Norm_U_rot < 1e-12] = 1e-12
        U_rot /= Norm_U_rot[..., None]
        U_rot *= self.sqrt_d
        
        # 4. Asymmetric Quantization onto 4-bit codebook
        # We digitize the absolute values and keep the signs separately.
        U_abs = np.abs(U_rot)
        buckets = np.digitize(U_abs, POS_BOUNDARIES)
        
        # Map to codebook indices (0-7 for positive, 8-15 for negative)
        cb_indices = buckets + (U_rot < 0).astype(np.int32) * 8
        self.U_int8 = INT8_CODEBOOK[cb_indices]
        
        # Pre-cache float64 version to avoid repeated casting during score()
        self.U_int8_f64_T = self.U_int8.astype(np.float64).T
        
        # 5. Compute the base scaling factor self.Scale_U
        # Uses analytical squared centroids with debiasing for energy preservation.
        Pseudo_Sqr_Sum = np.sum(SQUARED_CENTROIDS_WITH_DEBIAS[buckets], axis=-1)
        self.Scale_U = Norm_U / np.sqrt(Pseudo_Sqr_Sum)
        
    def score(self, V: np.ndarray) -> np.ndarray:
        """
        Compute accelerated dot products between query V and stored U.
        Score = (V_int8 @ U_int8^T) * Scale_U * Scale_V.
        """
        if self.U_int8 is None:
            raise ValueError("No base embeddings added yet.")
            
        V = np.asarray(V, dtype=np.float64)
        is_1d = V.ndim == 1
        if is_1d:
            V = V.reshape(1, -1)
            
        # 1. Capture query norm
        Norm_V = np.linalg.norm(V, axis=-1)
        Norm_V[Norm_V < 1e-12] = 1e-12
        
        # 2. Rotate query to the same Gaussian space as U
        V_rot = self.rotation(V)
        
        # 3. Dynamic Linear Scaling for the query (V)
        # Maps the query max absolute value to 127 to maximize int8 resolution.
        V_max = np.max(np.abs(V_rot), axis=-1, keepdims=True)
        V_max[V_max < 1e-12] = 1e-12
        V_scale = 127.0 / V_max
        
        # 4. Quantize V to 8-bit signed integers
        V_int8 = np.round(V_rot * V_scale).astype(np.int8)
        
        # 5. Accelerated Integer Dot Product
        if cpp_compute_dot_products is not None:
            # Use high-performance SIMD implementation (Result: int32)
            # Cast to float64 for subsequent analytical scaling
            P = cpp_compute_dot_products(V_int8, self.U_int8).astype(np.float64)
        else:
            # Fallback to NumPy dot product
            # We use the cached float64 version of U for compatibility with np.dot
            P = np.dot(V_int8.astype(np.float64), self.U_int8_f64_T)
        
        # 6. Final reconstruction scaling
        # Scale_V = Norm_V / sqrt(sum(V_int8^2))
        V_int8_sqr_sum = np.sum(V_int8.astype(np.float64)**2, axis=-1)
        V_int8_sqr_sum[V_int8_sqr_sum < 1e-12] = 1e-12
        Scale_V = Norm_V / np.sqrt(V_int8_sqr_sum)
        
        # Final Score: P(Q, N) * Scale_U(N) * Scale_V(Q, 1)
        # Reconstructs the floating point dot product from integer results.
        scores = P * (self.Scale_U * Scale_V[:, None])
        
        return scores[0] if is_1d else scores

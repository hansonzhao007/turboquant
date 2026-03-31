import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

"""
TurboQuant 4-bit Parameters (Standardized N(0, 1)):
--------------------------------------------------
int8_codebook:   [6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127]
pos_boundaries:  [0.2583, 0.5225, 0.7997, 1.0995, 1.4374, 1.8438, 2.4011]
debias_factor:   1.00478463
int8_scale:      46.4720

COMPARISON (Centroid Energy):
squared_centroids (q^2):          [36, 324, 961, 1936, 3364, 5625, 9216, 16129]
squared_centroids_with_debias:    [35.64, 323.15, 941.89, 1918.15, 3370.59, 5613.64, 9187.77, 16052.2]
(Note: squared_centroids_with_debias[i] = q_i * E[z_scaled_i]. These are baked into the norm calculation.)

Full Flow Example (4D Vector, d=4) - ASYMMETRIC SCORING:
1. Input Vectors (Original Space):
   U (Base)  = [2.0, 4.0, -2.0, 0.0]  (Normally we quantize many base embeddings)
   V (Query) = [7.0, -1.0, 0.0, 1.0]  (Used to calculate the score to all base embeddings)
   => Original Exact Dot Product (U·V) = 10.0

   Original L2 Norms (Crucial for final scaling):
   Norm_U = sqrt(24) = 4.8990
   Norm_V = sqrt(51) = 7.1414

2. Rotation (Integer Hadamard Matrix H4):
   U_rot = H4 * U = [4.0, -4.0, 8.0, 0.0]  (Norm_U_rot = 9.7980)
   V_rot = H4 * V = [7.0, 7.0, 5.0, 9.0]   (Norm_V_rot = 14.2828)

3. Asymmetric Quantization (Base vs Query):
   -- A. Base (U) uses 4-bit TurboQuant Codebook (Abs Value + Sign Mapping):
      U_std = (U_rot / Norm_U_rot) * sqrt(d) = [0.8165, -0.8165, 1.6330, 0.0000]

      Mapping mechanism for U_std -> U_int8:
      * val =  0.8165: abs(val) in [0.800, 1.100) -> bucket 3 -> codebook[3] = 44. Sign (+) -> 44
      * val = -0.8165: abs(val) in [0.800, 1.100) -> bucket 3 -> codebook[3+8] = -44. Sign (-) -> -44
      * val =  1.6330: abs(val) in [1.437, 1.844) -> bucket 5 -> codebook[5] = 75. Sign (+) -> 75
      * val =  0.0000: abs(val) in [0.000, 0.258) -> bucket 0 -> codebook[0] = 6. Sign (+) -> 6

      U_int8 = [44, -44, 75, 6]

   -- B. Query (V) uses 8-bit Linear Scaling (Max absolute value = 9.0):
      V_scale = 127.0 / 9.0 = 14.111
      V_int8 = round(V_rot * V_scale) = [99, 99, 71, 127]

4. Raw Int8 SIMD Dot Product (P):
   P = sum(U_int8 * V_int8) = 6087

5. Final Score Calculation (The C++ Trick):
   -- Scale_U (Bakes debias_factor in via squared_centroids_with_debias):
      Pseudo_Sqr_Sum = sum(squared_centroids_with_debias[buckets]) = 9442.44
      Scale_U = Norm_U / sqrt(Pseudo_Sqr_Sum) = 0.05041

   -- Scale_V (Standard linear scale):
      Scale_V = Norm_V / sqrt(sum(V_int8^2)) = 0.03536

   Final Score = P * Scale_U * Scale_V = 10.85
"""


def _build_codebook(right_half: np.ndarray, bits: int) -> dict:
    """Shared post-processing: given positive-half centroids, build the full codebook."""
    n_levels = 2**bits
    half_levels = n_levels // 2

    # Scale to int8
    int8_scale = 127.0 / np.max(right_half)
    pos_int8 = np.round(right_half * int8_scale).astype(np.int8)

    # CODEBOOK ORDER: [6...127, -6...-127]
    int8_codebook = np.concatenate([pos_int8, -pos_int8])

    # Positive boundaries
    pos_boundaries = (right_half[:-1] + right_half[1:]) / 2.0

    # Debias factor using analytical PDF/CDF
    full_centroids = np.concatenate([-right_half[::-1], right_half])
    all_boundaries = np.zeros(n_levels + 1)
    all_boundaries[0] = -np.inf
    all_boundaries[-1] = np.inf
    for i in range(1, n_levels):
        all_boundaries[i] = (full_centroids[i-1] + full_centroids[i]) / 2.0

    e_q2 = 0.0
    e_zq = 0.0
    for i in range(n_levels):
        a, b = all_boundaries[i], all_boundaries[i+1]
        q = full_centroids[i]
        prob = norm.cdf(b) - norm.cdf(a)          # P(a <= Z <= b)
        ez_contrib = norm.pdf(a) - norm.pdf(b)    # integral of z*pdf(z) from a to b
        e_q2 += q**2 * prob
        e_zq += q * ez_contrib

    debias_factor = np.sqrt(e_q2) / e_zq

    # Energy comparison
    z_scaled_expected = right_half * int8_scale
    squared_centroids = pos_int8.astype(np.float64) ** 2
    squared_centroids_with_debias = pos_int8.astype(np.float64) * z_scaled_expected / debias_factor

    return {
        "int8_codebook": int8_codebook.tolist(),
        "pos_boundaries": pos_boundaries.tolist(),
        "debias_factor": float(debias_factor),
        "int8_scale": float(int8_scale),
        "comparison": {
            "squared_centroids": squared_centroids.tolist(),
            "squared_centroids_with_debias": squared_centroids_with_debias.tolist()
        }
    }


def generate_turboquant_lloyd_max(bits: int, iterations: int = 2000) -> dict:
    """Generate codebook using Lloyd-Max algorithm with numerical integration (quad)."""
    n_levels = 2**bits
    half_levels = n_levels // 2

    # Lloyd-Max iterations on the full distribution
    centroids = np.linspace(-3, 3, n_levels)
    boundaries = np.zeros(n_levels + 1)
    boundaries[0], boundaries[-1] = -np.inf, np.inf

    for _ in range(iterations):
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0

        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            num, _ = quad(lambda x: x * norm.pdf(x), boundaries[i], boundaries[i+1])
            den, _ = quad(norm.pdf, boundaries[i], boundaries[i+1])
            new_centroids[i] = num / den if den > 0 else centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    right_half = centroids[half_levels:]
    return _build_codebook(right_half, bits)


def generate_turboquant_kmeans(bits: int, iterations: int = 2000,
                               max_sample_value: float = 10) -> dict:
    """Generate codebook using k-means with analytical PDF/CDF centroid computation."""
    n_levels = 2**bits
    half_levels = n_levels // 2

    # Initialize centroids evenly across the sample range (z-space: [1e-10, max_sample_value])
    centroids = np.linspace(1e-10, max_sample_value, half_levels)

    # K-means iterations with analytical centroid computation (PDF/CDF)
    for _ in range(iterations):
        # Boundaries: midpoints between adjacent centroids
        boundaries = np.zeros(half_levels + 1)
        boundaries[0] = 0.0
        boundaries[-1] = np.inf
        for i in range(1, half_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0

        # Centroid update using conditional expectation:
        # E[Z | a <= Z <= b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        new_centroids = np.zeros(half_levels)
        for i in range(half_levels):
            a, b = boundaries[i], boundaries[i+1]
            pdf_diff = norm.pdf(a) - norm.pdf(b)
            cdf_diff = norm.cdf(b) - norm.cdf(a)
            new_centroids[i] = pdf_diff / cdf_diff if cdf_diff > 0 else centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    return _build_codebook(centroids, bits)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <arm_neon.h>

namespace py = pybind11;

/**
 * HW-Accelerated In-place Fast Walsh-Hadamard Transform using ARM NEON.
 * Complexity: O(d log d)
 */
void fwht_internal(double* data, size_t d) {
    for (size_t step = 1; step < d; step <<= 1) {
        size_t group_size = step << 1;
        
        // When step >= 2, we can perform vectorized additions/subtractions
        // using 128-bit NEON registers (holding 2 doubles).
        if (step >= 2) {
            for (size_t i = 0; i < d; i += group_size) {
                for (size_t j = 0; j < step; j += 2) {
                    // Load two elements from the 'a' block and two from the 'b' block
                    float64x2_t v_a = vld1q_f64(&data[i + j]);
                    float64x2_t v_b = vld1q_f64(&data[i + j + step]);
                    
                    // a' = a + b
                    // b' = a - b
                    float64x2_t sum = vaddq_f64(v_a, v_b);
                    float64x2_t diff = vsubq_f64(v_a, v_b);
                    
                    // Store the results back
                    vst1q_f64(&data[i + j], sum);
                    vst1q_f64(&data[i + j + step], diff);
                }
            }
        } else {
            // Scalar path for the very first step (step=1) where pairwise distance is 1
            for (size_t i = 0; i < d; i += group_size) {
                double a = data[i];
                double b = data[i + 1];
                data[i] = a + b;
                data[i + 1] = a - b;
            }
        }
    }
}

/**
 * In-place FWHT for Python.
 * No data copy occurs; the input NumPy array buffer is directly modified.
 */
void fwht_inplace(py::array_t<double, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim < 1) {
        throw std::runtime_error("Input must have at least 1 dimension.");
    }
    
    size_t d = buf.shape[buf.ndim - 1];
    
    // Check if d is a power of 2
    if (d == 0 || (d & (d - 1)) != 0) {
        throw std::runtime_error("Fast Hadamard Transform requires dimension to be a power of 2.");
    }
    
    double* data = static_cast<double*>(buf.ptr);
    
    // Calculate number of vectors in the batch
    size_t num_vectors = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) {
        num_vectors *= buf.shape[i];
    }
    
    // Apply FWHT to each vector in-place
    for (size_t i = 0; i < num_vectors; ++i) {
        fwht_internal(data + i * d, d);
    }
}

/**
 * SIMD-accelerated signed 8-bit integer matrix multiplication using ARM NEON.
 * Computes Result(Q, N) = Query(Q, d) @ Database(N, d)^T
 */
py::array_t<int32_t> compute_dot_products(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> queries,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> database) {
    
    py::buffer_info q_buf = queries.request();
    py::buffer_info d_buf = database.request();
    
    if (q_buf.ndim != 2 || d_buf.ndim != 2) {
        throw std::runtime_error("Inputs must be 2D matrices (Q, d) and (N, d).");
    }
    
    size_t Q = q_buf.shape[0];
    size_t d = q_buf.shape[1];
    size_t N = d_buf.shape[0];
    
    if (d != d_buf.shape[1]) {
        throw std::runtime_error("Dimension mismatch between query and database.");
    }
    
    auto result = py::array_t<int32_t>({Q, N});
    py::buffer_info r_buf = result.request();
    int32_t* r_ptr = static_cast<int32_t*>(r_buf.ptr);
    
    const int8_t* q_ptr = static_cast<const int8_t*>(q_buf.ptr);
    const int8_t* d_ptr = static_cast<const int8_t*>(d_buf.ptr);
    
    // Performance: Interleave across N (database rows) to improve instruction-level parallelism.
    // Each query vector is reused across multiple database vectors.
    for (size_t q = 0; q < Q; ++q) {
        const int8_t* q_row = q_ptr + q * d;
        
        size_t n = 0;
        for (; n + 3 < N; n += 4) {
            const int8_t* b0 = d_ptr + (n + 0) * d;
            const int8_t* b1 = d_ptr + (n + 1) * d;
            const int8_t* b2 = d_ptr + (n + 2) * d;
            const int8_t* b3 = d_ptr + (n + 3) * d;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            size_t i = 0;
            // Hot loop uses vdotq_s32 (Apple M1/M2/M3/M4 dot product extension)
            for (; i + 15 < d; i += 16) {
                int8x16_t va = vld1q_s8(q_row + i);
                acc0 = vdotq_s32(acc0, va, vld1q_s8(b0 + i));
                acc1 = vdotq_s32(acc1, va, vld1q_s8(b1 + i));
                acc2 = vdotq_s32(acc2, va, vld1q_s8(b2 + i));
                acc3 = vdotq_s32(acc3, va, vld1q_s8(b3 + i));
            }
            
            r_ptr[q * N + n + 0] = vgetq_lane_s32(acc0, 0) + vgetq_lane_s32(acc0, 1) + vgetq_lane_s32(acc0, 2) + vgetq_lane_s32(acc0, 3);
            r_ptr[q * N + n + 1] = vgetq_lane_s32(acc1, 0) + vgetq_lane_s32(acc1, 1) + vgetq_lane_s32(acc1, 2) + vgetq_lane_s32(acc1, 3);
            r_ptr[q * N + n + 2] = vgetq_lane_s32(acc2, 0) + vgetq_lane_s32(acc2, 1) + vgetq_lane_s32(acc2, 2) + vgetq_lane_s32(acc2, 3);
            r_ptr[q * N + n + 3] = vgetq_lane_s32(acc3, 0) + vgetq_lane_s32(acc3, 1) + vgetq_lane_s32(acc3, 2) + vgetq_lane_s32(acc3, 3);
            
            // Handle tail of d if not multiple of 16
            for (; i < d; ++i) {
                int32_t val_q = q_row[i];
                r_ptr[q * N + n + 0] += val_q * b0[i];
                r_ptr[q * N + n + 1] += val_q * b1[i];
                r_ptr[q * N + n + 2] += val_q * b2[i];
                r_ptr[q * N + n + 3] += val_q * b3[i];
            }
        }
        
        // Handle tail of N if not multiple of 4
        for (; n < N; ++n) {
            const int8_t* b = d_ptr + n * d;
            int32x4_t acc = vdupq_n_s32(0);
            size_t i = 0;
            for (; i + 15 < d; i += 16) {
                acc = vdotq_s32(acc, vld1q_s8(q_row + i), vld1q_s8(b + i));
            }
            int32_t dot = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
            for (; i < d; ++i) dot += (int32_t)q_row[i] * (int32_t)b[i];
            r_ptr[q * N + n] = dot;
        }
    }
    
    return result;
}

PYBIND11_MODULE(fast_hadamard, m) {
    m.doc() = "High-performance C++ implementation of the Fast Walsh-Hadamard Transform and SIMD Dot Products.";
    m.def("fwht_inplace", &fwht_inplace, "Computes the Fast Walsh-Hadamard Transform in-place on the last dimension of the input array.",
          py::arg("input"));
    m.def("compute_dot_products", &compute_dot_products, "Accelerated signed 8-bit integer dot product (Q, d) @ (N, d)^T.",
          py::arg("queries"), py::arg("database"));
}

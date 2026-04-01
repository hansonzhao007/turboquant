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

PYBIND11_MODULE(fast_hadamard, m) {
    m.doc() = "High-performance C++ implementation of the Fast Walsh-Hadamard Transform (SIMD Optimized).";
    m.def("fwht_inplace", &fwht_inplace, "Computes the Fast Walsh-Hadamard Transform in-place on the last dimension of the input array.",
          py::arg("input"));
}

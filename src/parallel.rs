//! Parallel matrix multiplication using Rayon.
//!
//! This module provides multi-threaded matrix multiplication implementations
//! using Rayon's work-stealing thread pool. It includes:
//!
//! - `multiply_parallel`: Generic parallel multiplication for any numeric type
//! - `multiply_parallel_simd`: Combined parallel + SIMD for f32/f64 (fastest)
//!
//! The parallelization strategy uses row-level parallelism for the outer loop,
//! combined with cache-efficient tiling for the inner computation.

use crate::Matrix;
use rayon::prelude::*;

/// Tile size for cache-efficient blocked multiplication
const TILE_SIZE: usize = 64;

/// Minimum matrix dimension to use parallelization
/// Below this threshold, sequential is often faster due to thread overhead
const PARALLEL_THRESHOLD: usize = 64;

impl<T> Matrix<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + Send + Sync,
{
    /// Parallel matrix multiplication using Rayon.
    ///
    /// Uses work-stealing parallelism across rows with cache-efficient tiling.
    /// Falls back to sequential multiplication for small matrices where
    /// thread overhead would outweigh parallelism benefits.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(100, 100, vec![1; 10000]).unwrap();
    /// let b = Matrix::from_vec(100, 100, vec![2; 10000]).unwrap();
    /// let c = a.multiply_parallel(&b).unwrap();
    /// ```
    pub fn multiply_parallel(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // Use sequential for small matrices
        if self.rows < PARALLEL_THRESHOLD || self.cols < PARALLEL_THRESHOLD || other.cols < PARALLEL_THRESHOLD {
            return self.multiply(other);
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // Pre-allocate result with default values
        let mut result_data = vec![T::default(); m * n];

        // Parallel iteration over row tiles
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, result_row)| {
                // Each thread computes one row of the result
                for jj in (0..n).step_by(TILE_SIZE) {
                    let j_end = (jj + TILE_SIZE).min(n);

                    for kk in (0..k).step_by(TILE_SIZE) {
                        let k_end = (kk + TILE_SIZE).min(k);

                        for j in jj..j_end {
                            let mut sum = result_row[j].clone();

                            for ki in kk..k_end {
                                let a_val = self.data[i * k + ki].clone();
                                let b_val = other.data[ki * n + j].clone();
                                sum = sum + (a_val * b_val);
                            }

                            result_row[j] = sum;
                        }
                    }
                }
            });

        Ok(Matrix {
            rows: m,
            cols: n,
            data: result_data,
        })
    }
}

// =============================================================================
// Parallel + SIMD Implementation for f32
// =============================================================================

impl Matrix<f32> {
    /// Combined parallel and SIMD matrix multiplication for f32.
    ///
    /// This is the fastest multiplication method, combining:
    /// - Multi-threaded parallelism via Rayon (row-level)
    /// - SIMD vectorization (AVX2/NEON for inner dot products)
    /// - Cache-efficient tiling (64x64 blocks)
    /// - Transposed B matrix for sequential memory access
    ///
    /// # Performance
    /// On a typical modern CPU, this can achieve 3-5x speedup over
    /// single-threaded SIMD, and 10-15x over naive sequential.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a: Matrix<f32> = Matrix::from_vec(256, 256,
    ///     (0..65536).map(|i| i as f32).collect()).unwrap();
    /// let b: Matrix<f32> = Matrix::from_vec(256, 256,
    ///     (0..65536).map(|i| (i * 2) as f32).collect()).unwrap();
    /// let c = a.multiply_parallel_simd(&b).unwrap();
    /// ```
    pub fn multiply_parallel_simd(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // Use sequential SIMD for small matrices
        if self.rows < PARALLEL_THRESHOLD || self.cols < PARALLEL_THRESHOLD || other.cols < PARALLEL_THRESHOLD {
            return self.multiply_simd(other);
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // Transpose B for better cache access pattern
        let b_transposed = other.transpose();

        // Pre-allocate result
        let mut result_data = vec![0.0_f32; m * n];

        // Parallel iteration over rows
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, result_row)| {
                let a_row = &self.data[i * k..(i + 1) * k];

                for j in 0..n {
                    let b_row = &b_transposed.data[j * k..(j + 1) * k];
                    result_row[j] = Self::dot_product_simd_f32(a_row, b_row);
                }
            });

        Ok(Matrix {
            rows: m,
            cols: n,
            data: result_data,
        })
    }

    /// SIMD-accelerated dot product for f32 vectors
    #[inline]
    fn dot_product_simd_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { Self::dot_product_avx2_f32(a, b) };
            } else if is_x86_feature_detected!("avx") {
                return unsafe { Self::dot_product_avx_f32(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::dot_product_neon_f32(a, b) };
        }

        // Scalar fallback
        #[allow(unreachable_code)]
        {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn dot_product_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let n = a.len();
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            i += 8;
        }

        // Horizontal sum
        let sum128_lo = _mm256_castps256_ps128(sum);
        let sum128_hi = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn dot_product_avx_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let n = a.len();
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
            i += 8;
        }

        let sum128_lo = _mm256_castps256_ps128(sum);
        let sum128_hi = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_product_neon_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let n = a.len();
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= n {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            sum = vfmaq_f32(sum, a_vec, b_vec);
            i += 4;
        }

        let mut result = vaddvq_f32(sum);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }
}

// =============================================================================
// Parallel + SIMD Implementation for f64
// =============================================================================

impl Matrix<f64> {
    /// Combined parallel and SIMD matrix multiplication for f64.
    ///
    /// This is the fastest multiplication method for double precision,
    /// combining multi-threaded parallelism with SIMD vectorization.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a: Matrix<f64> = Matrix::from_vec(256, 256,
    ///     (0..65536).map(|i| i as f64).collect()).unwrap();
    /// let b: Matrix<f64> = Matrix::from_vec(256, 256,
    ///     (0..65536).map(|i| (i * 2) as f64).collect()).unwrap();
    /// let c = a.multiply_parallel_simd(&b).unwrap();
    /// ```
    pub fn multiply_parallel_simd(&self, other: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // Use sequential SIMD for small matrices
        if self.rows < PARALLEL_THRESHOLD || self.cols < PARALLEL_THRESHOLD || other.cols < PARALLEL_THRESHOLD {
            return self.multiply_simd(other);
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // Transpose B for better cache access pattern
        let b_transposed = other.transpose();

        // Pre-allocate result
        let mut result_data = vec![0.0_f64; m * n];

        // Parallel iteration over rows
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, result_row)| {
                let a_row = &self.data[i * k..(i + 1) * k];

                for j in 0..n {
                    let b_row = &b_transposed.data[j * k..(j + 1) * k];
                    result_row[j] = Self::dot_product_simd_f64(a_row, b_row);
                }
            });

        Ok(Matrix {
            rows: m,
            cols: n,
            data: result_data,
        })
    }

    /// SIMD-accelerated dot product for f64 vectors
    #[inline]
    fn dot_product_simd_f64(a: &[f64], b: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { Self::dot_product_avx2_f64(a, b) };
            } else if is_x86_feature_detected!("avx") {
                return unsafe { Self::dot_product_avx_f64(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { Self::dot_product_neon_f64(a, b) };
        }

        // Scalar fallback
        #[allow(unreachable_code)]
        {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn dot_product_avx2_f64(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::x86_64::*;

        let n = a.len();
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        while i + 4 <= n {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
            sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
            i += 4;
        }

        // Horizontal sum
        let sum128_lo = _mm256_castpd256_pd128(sum);
        let sum128_hi = _mm256_extractf128_pd(sum, 1);
        let sum128 = _mm_add_pd(sum128_lo, sum128_hi);
        let sum_high = _mm_unpackhi_pd(sum128, sum128);
        let sum_scalar = _mm_add_sd(sum128, sum_high);
        let mut result = _mm_cvtsd_f64(sum_scalar);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn dot_product_avx_f64(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::x86_64::*;

        let n = a.len();
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;

        while i + 4 <= n {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
            let prod = _mm256_mul_pd(a_vec, b_vec);
            sum = _mm256_add_pd(sum, prod);
            i += 4;
        }

        let sum128_lo = _mm256_castpd256_pd128(sum);
        let sum128_hi = _mm256_extractf128_pd(sum, 1);
        let sum128 = _mm_add_pd(sum128_lo, sum128_hi);
        let sum_high = _mm_unpackhi_pd(sum128, sum128);
        let sum_scalar = _mm_add_sd(sum128, sum_high);
        let mut result = _mm_cvtsd_f64(sum_scalar);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_product_neon_f64(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::aarch64::*;

        let n = a.len();
        let mut sum = vdupq_n_f64(0.0);
        let mut i = 0;

        while i + 2 <= n {
            let a_vec = vld1q_f64(a.as_ptr().add(i));
            let b_vec = vld1q_f64(b.as_ptr().add(i));
            sum = vfmaq_f64(sum, a_vec, b_vec);
            i += 2;
        }

        let mut result = vaddvq_f64(sum);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }
}

/// Returns the number of threads in the Rayon thread pool.
///
/// This is useful for understanding the level of parallelism available.
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_count() {
        let count = thread_count();
        assert!(count >= 1);
        println!("Rayon thread pool size: {}", count);
    }

    #[test]
    fn test_multiply_parallel_basic() {
        let a = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let b = Matrix::from_vec(3, 3, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();

        let result_parallel = a.multiply_parallel(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        assert_eq!(result_parallel, result_naive);
    }

    #[test]
    fn test_multiply_parallel_large() {
        let size = 128;
        let a_data: Vec<i32> = (0..size * size).map(|i| (i % 100) as i32).collect();
        let b_data: Vec<i32> = (0..size * size).map(|i| ((i * 2) % 100) as i32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_parallel = a.multiply_parallel(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        assert_eq!(result_parallel, result_naive);
    }

    #[test]
    fn test_multiply_parallel_rectangular() {
        let a = Matrix::from_vec(100, 150, (0..15000).map(|i| (i % 100) as i32).collect()).unwrap();
        let b = Matrix::from_vec(150, 80, (0..12000).map(|i| ((i * 2) % 100) as i32).collect()).unwrap();

        let result_parallel = a.multiply_parallel(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        assert_eq!(result_parallel, result_naive);
    }

    #[test]
    fn test_multiply_parallel_dimension_mismatch() {
        let a: Matrix<i32> = Matrix::new(100, 100);
        let b: Matrix<i32> = Matrix::new(50, 100);

        assert!(a.multiply_parallel(&b).is_err());
    }

    #[test]
    fn test_multiply_parallel_simd_f32_basic() {
        let a = Matrix::from_vec(3, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let b = Matrix::from_vec(3, 3, vec![9.0_f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();

        let result_parallel = a.multiply_parallel_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_parallel.rows {
            for j in 0..result_parallel.cols {
                let p_val = *result_parallel.get(i, j).unwrap();
                let n_val = *result_naive.get(i, j).unwrap();
                assert!(
                    (p_val - n_val).abs() < 1e-5,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, p_val, n_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_parallel_simd_f32_large() {
        let size = 256;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_parallel = a.multiply_parallel_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_parallel.rows {
            for j in 0..result_parallel.cols {
                let p_val = *result_parallel.get(i, j).unwrap();
                let n_val = *result_naive.get(i, j).unwrap();
                let rel_diff = if n_val.abs() > 1e-10 {
                    (p_val - n_val).abs() / n_val.abs()
                } else {
                    (p_val - n_val).abs()
                };
                assert!(
                    rel_diff < 1e-4,
                    "Mismatch at ({}, {}): {} vs {} (rel diff: {})",
                    i, j, p_val, n_val, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_multiply_parallel_simd_f64_basic() {
        let a = Matrix::from_vec(3, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let b = Matrix::from_vec(3, 3, vec![9.0_f64, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();

        let result_parallel = a.multiply_parallel_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_parallel.rows {
            for j in 0..result_parallel.cols {
                let p_val = *result_parallel.get(i, j).unwrap();
                let n_val = *result_naive.get(i, j).unwrap();
                assert!(
                    (p_val - n_val).abs() < 1e-10,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, p_val, n_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_parallel_simd_f64_large() {
        let size = 256;
        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 100) as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| ((i * 2) % 100) as f64).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_parallel = a.multiply_parallel_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_parallel.rows {
            for j in 0..result_parallel.cols {
                let p_val = *result_parallel.get(i, j).unwrap();
                let n_val = *result_naive.get(i, j).unwrap();
                let rel_diff = if n_val.abs() > 1e-10 {
                    (p_val - n_val).abs() / n_val.abs()
                } else {
                    (p_val - n_val).abs()
                };
                assert!(
                    rel_diff < 1e-10,
                    "Mismatch at ({}, {}): {} vs {} (rel diff: {})",
                    i, j, p_val, n_val, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_multiply_parallel_simd_rectangular() {
        let a = Matrix::from_vec(100, 150, (0..15000).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(150, 80, (0..12000).map(|i| (i * 2) as f32).collect()).unwrap();

        let result_parallel = a.multiply_parallel_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_parallel.rows {
            for j in 0..result_parallel.cols {
                let p_val = *result_parallel.get(i, j).unwrap();
                let n_val = *result_naive.get(i, j).unwrap();
                let rel_diff = if n_val.abs() > 1e-10 {
                    (p_val - n_val).abs() / n_val.abs()
                } else {
                    (p_val - n_val).abs()
                };
                assert!(
                    rel_diff < 1e-4,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, p_val, n_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_parallel_simd_dimension_mismatch() {
        let a: Matrix<f32> = Matrix::new(100, 100);
        let b: Matrix<f32> = Matrix::new(50, 100);

        assert!(a.multiply_parallel_simd(&b).is_err());
    }

    #[test]
    fn test_multiply_parallel_simd_identity_f32() {
        let size = 128;
        let mut identity_data = vec![0.0_f32; size * size];
        for i in 0..size {
            identity_data[i * size + i] = 1.0;
        }

        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 50) as f32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let identity = Matrix::from_vec(size, size, identity_data).unwrap();

        let result = a.multiply_parallel_simd(&identity).unwrap();

        for i in 0..size {
            for j in 0..size {
                let result_val = *result.get(i, j).unwrap();
                let a_val = *a.get(i, j).unwrap();
                assert!(
                    (result_val - a_val).abs() < 1e-5,
                    "Identity multiplication failed at ({}, {})",
                    i, j
                );
            }
        }
    }
}

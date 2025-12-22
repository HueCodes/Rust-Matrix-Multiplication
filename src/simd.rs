//! SIMD-accelerated matrix multiplication operations.
//!
//! This module provides hardware-accelerated matrix multiplication using:
//! - AVX2/AVX on x86_64 (256-bit vectors: 8 x f32 or 4 x f64)
//! - SSE on x86_64 fallback (128-bit vectors: 4 x f32 or 2 x f64)
//! - NEON on aarch64 (128-bit vectors: 4 x f32 or 2 x f64)
//!
//! The implementation automatically detects CPU features at runtime and
//! selects the best available instruction set.

use crate::Matrix;

/// Tile size for cache-efficient blocked multiplication
const TILE_SIZE: usize = 64;

/// Check if AVX2 is available at runtime (x86_64 only)
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if AVX is available at runtime (x86_64 only)
#[cfg(target_arch = "x86_64")]
fn has_avx() -> bool {
    is_x86_feature_detected!("avx")
}

// =============================================================================
// f32 SIMD Implementation
// =============================================================================

impl Matrix<f32> {
    /// SIMD-accelerated matrix multiplication for f32 matrices.
    ///
    /// Automatically selects the best available SIMD instruction set:
    /// - AVX2 on modern x86_64 processors (8-wide f32)
    /// - AVX on older x86_64 processors (8-wide f32)
    /// - SSE on legacy x86_64 (4-wide f32)
    /// - NEON on ARM64 processors (4-wide f32)
    /// - Scalar fallback on other architectures
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let b = Matrix::from_vec(3, 2, vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    /// let c = a.multiply_simd(&b).unwrap();
    /// ```
    pub fn multiply_simd(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                // SAFETY: We've verified AVX2 is available
                return Ok(unsafe { self.multiply_avx2(other) });
            } else if has_avx() {
                // SAFETY: We've verified AVX is available
                return Ok(unsafe { self.multiply_avx_f32(other) });
            } else {
                return Ok(unsafe { self.multiply_sse_f32(other) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is always available on aarch64
            return Ok(unsafe { self.multiply_neon_f32(other) });
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar implementation
            self.multiply(other)
        }
    }

    /// AVX2 implementation for f32 (8-wide SIMD)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn multiply_avx2(&self, other: &Matrix<f32>) -> Matrix<f32> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // Transpose B for better cache access pattern (row-major access on both)
        let b_transposed = other.transpose();

        let mut result = Matrix::new(m, n);

        // Process in tiles for better cache utilization
        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = _mm256_setzero_ps();
                        let mut kk = 0;

                        // Process 8 elements at a time with AVX2
                        while kk + 8 <= k {
                            let a_vec = _mm256_loadu_ps(a_row.as_ptr().add(kk));
                            let b_vec = _mm256_loadu_ps(b_row.as_ptr().add(kk));
                            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            kk += 8;
                        }

                        // Horizontal sum of the 256-bit vector
                        let sum128_lo = _mm256_castps256_ps128(sum);
                        let sum128_hi = _mm256_extractf128_ps(sum, 1);
                        let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
                        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
                        let mut scalar_sum = _mm_cvtss_f32(sum32);

                        // Handle remaining elements
                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }

    /// AVX implementation for f32 (8-wide SIMD, without FMA)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn multiply_avx_f32(&self, other: &Matrix<f32>) -> Matrix<f32> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = _mm256_setzero_ps();
                        let mut kk = 0;

                        while kk + 8 <= k {
                            let a_vec = _mm256_loadu_ps(a_row.as_ptr().add(kk));
                            let b_vec = _mm256_loadu_ps(b_row.as_ptr().add(kk));
                            let prod = _mm256_mul_ps(a_vec, b_vec);
                            sum = _mm256_add_ps(sum, prod);
                            kk += 8;
                        }

                        // Horizontal sum
                        let sum128_lo = _mm256_castps256_ps128(sum);
                        let sum128_hi = _mm256_extractf128_ps(sum, 1);
                        let sum128 = _mm_add_ps(sum128_lo, sum128_hi);
                        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
                        let mut scalar_sum = _mm_cvtss_f32(sum32);

                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }

    /// SSE implementation for f32 (4-wide SIMD)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe fn multiply_sse_f32(&self, other: &Matrix<f32>) -> Matrix<f32> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for i in 0..m {
            let a_row = &self.data[i * k..(i + 1) * k];

            for j in 0..n {
                let b_row = &b_transposed.data[j * k..(j + 1) * k];

                let mut sum = _mm_setzero_ps();
                let mut kk = 0;

                while kk + 4 <= k {
                    let a_vec = _mm_loadu_ps(a_row.as_ptr().add(kk));
                    let b_vec = _mm_loadu_ps(b_row.as_ptr().add(kk));
                    let prod = _mm_mul_ps(a_vec, b_vec);
                    sum = _mm_add_ps(sum, prod);
                    kk += 4;
                }

                // Horizontal sum for SSE
                let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
                let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
                let mut scalar_sum = _mm_cvtss_f32(sum32);

                while kk < k {
                    scalar_sum += a_row[kk] * b_row[kk];
                    kk += 1;
                }

                result.data[i * n + j] = scalar_sum;
            }
        }

        result
    }

    /// NEON implementation for f32 (4-wide SIMD)
    #[cfg(target_arch = "aarch64")]
    unsafe fn multiply_neon_f32(&self, other: &Matrix<f32>) -> Matrix<f32> {
        use std::arch::aarch64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = vdupq_n_f32(0.0);
                        let mut kk = 0;

                        // Process 4 elements at a time with NEON
                        while kk + 4 <= k {
                            let a_vec = vld1q_f32(a_row.as_ptr().add(kk));
                            let b_vec = vld1q_f32(b_row.as_ptr().add(kk));
                            sum = vfmaq_f32(sum, a_vec, b_vec);
                            kk += 4;
                        }

                        // Horizontal sum
                        let mut scalar_sum = vaddvq_f32(sum);

                        // Handle remaining elements
                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }
}

// =============================================================================
// f64 SIMD Implementation
// =============================================================================

impl Matrix<f64> {
    /// SIMD-accelerated matrix multiplication for f64 matrices.
    ///
    /// Automatically selects the best available SIMD instruction set:
    /// - AVX2 on modern x86_64 processors (4-wide f64)
    /// - AVX on older x86_64 processors (4-wide f64)
    /// - SSE2 on legacy x86_64 (2-wide f64)
    /// - NEON on ARM64 processors (2-wide f64)
    /// - Scalar fallback on other architectures
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let b = Matrix::from_vec(3, 2, vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    /// let c = a.multiply_simd(&b).unwrap();
    /// ```
    pub fn multiply_simd(&self, other: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                // SAFETY: We've verified AVX2 is available
                return Ok(unsafe { self.multiply_avx2_f64(other) });
            } else if has_avx() {
                // SAFETY: We've verified AVX is available
                return Ok(unsafe { self.multiply_avx_f64(other) });
            } else {
                return Ok(unsafe { self.multiply_sse2_f64(other) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is always available on aarch64
            return Ok(unsafe { self.multiply_neon_f64(other) });
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar implementation
            self.multiply(other)
        }
    }

    /// AVX2 implementation for f64 (4-wide SIMD with FMA)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn multiply_avx2_f64(&self, other: &Matrix<f64>) -> Matrix<f64> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = _mm256_setzero_pd();
                        let mut kk = 0;

                        // Process 4 f64 elements at a time
                        while kk + 4 <= k {
                            let a_vec = _mm256_loadu_pd(a_row.as_ptr().add(kk));
                            let b_vec = _mm256_loadu_pd(b_row.as_ptr().add(kk));
                            sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
                            kk += 4;
                        }

                        // Horizontal sum for f64
                        let sum128_lo = _mm256_castpd256_pd128(sum);
                        let sum128_hi = _mm256_extractf128_pd(sum, 1);
                        let sum128 = _mm_add_pd(sum128_lo, sum128_hi);
                        let sum_high = _mm_unpackhi_pd(sum128, sum128);
                        let sum_scalar = _mm_add_sd(sum128, sum_high);
                        let mut scalar_sum = _mm_cvtsd_f64(sum_scalar);

                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }

    /// AVX implementation for f64 (4-wide SIMD, without FMA)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn multiply_avx_f64(&self, other: &Matrix<f64>) -> Matrix<f64> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = _mm256_setzero_pd();
                        let mut kk = 0;

                        while kk + 4 <= k {
                            let a_vec = _mm256_loadu_pd(a_row.as_ptr().add(kk));
                            let b_vec = _mm256_loadu_pd(b_row.as_ptr().add(kk));
                            let prod = _mm256_mul_pd(a_vec, b_vec);
                            sum = _mm256_add_pd(sum, prod);
                            kk += 4;
                        }

                        // Horizontal sum
                        let sum128_lo = _mm256_castpd256_pd128(sum);
                        let sum128_hi = _mm256_extractf128_pd(sum, 1);
                        let sum128 = _mm_add_pd(sum128_lo, sum128_hi);
                        let sum_high = _mm_unpackhi_pd(sum128, sum128);
                        let sum_scalar = _mm_add_sd(sum128, sum_high);
                        let mut scalar_sum = _mm_cvtsd_f64(sum_scalar);

                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }

    /// SSE2 implementation for f64 (2-wide SIMD)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn multiply_sse2_f64(&self, other: &Matrix<f64>) -> Matrix<f64> {
        use std::arch::x86_64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for i in 0..m {
            let a_row = &self.data[i * k..(i + 1) * k];

            for j in 0..n {
                let b_row = &b_transposed.data[j * k..(j + 1) * k];

                let mut sum = _mm_setzero_pd();
                let mut kk = 0;

                while kk + 2 <= k {
                    let a_vec = _mm_loadu_pd(a_row.as_ptr().add(kk));
                    let b_vec = _mm_loadu_pd(b_row.as_ptr().add(kk));
                    let prod = _mm_mul_pd(a_vec, b_vec);
                    sum = _mm_add_pd(sum, prod);
                    kk += 2;
                }

                // Horizontal sum
                let sum_high = _mm_unpackhi_pd(sum, sum);
                let sum_scalar = _mm_add_sd(sum, sum_high);
                let mut scalar_sum = _mm_cvtsd_f64(sum_scalar);

                while kk < k {
                    scalar_sum += a_row[kk] * b_row[kk];
                    kk += 1;
                }

                result.data[i * n + j] = scalar_sum;
            }
        }

        result
    }

    /// NEON implementation for f64 (2-wide SIMD)
    #[cfg(target_arch = "aarch64")]
    unsafe fn multiply_neon_f64(&self, other: &Matrix<f64>) -> Matrix<f64> {
        use std::arch::aarch64::*;

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let b_transposed = other.transpose();
        let mut result = Matrix::new(m, n);

        for ii in (0..m).step_by(TILE_SIZE) {
            let i_end = (ii + TILE_SIZE).min(m);
            for jj in (0..n).step_by(TILE_SIZE) {
                let j_end = (jj + TILE_SIZE).min(n);

                for i in ii..i_end {
                    let a_row = &self.data[i * k..(i + 1) * k];

                    for j in jj..j_end {
                        let b_row = &b_transposed.data[j * k..(j + 1) * k];

                        let mut sum = vdupq_n_f64(0.0);
                        let mut kk = 0;

                        // Process 2 f64 elements at a time with NEON
                        while kk + 2 <= k {
                            let a_vec = vld1q_f64(a_row.as_ptr().add(kk));
                            let b_vec = vld1q_f64(b_row.as_ptr().add(kk));
                            sum = vfmaq_f64(sum, a_vec, b_vec);
                            kk += 2;
                        }

                        // Horizontal sum
                        let mut scalar_sum = vaddvq_f64(sum);

                        // Handle remaining elements
                        while kk < k {
                            scalar_sum += a_row[kk] * b_row[kk];
                            kk += 1;
                        }

                        result.data[i * n + j] = scalar_sum;
                    }
                }
            }
        }

        result
    }
}

// =============================================================================
// CPU Feature Detection Helpers
// =============================================================================

/// Returns a string describing the SIMD instruction set that will be used.
///
/// This is useful for debugging and performance analysis.
///
/// # Example
/// ```
/// use matrix_multiply::simd_instruction_set;
///
/// println!("Using SIMD: {}", simd_instruction_set());
/// ```
pub fn simd_instruction_set() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2 + FMA";
        } else if is_x86_feature_detected!("avx2") {
            return "AVX2";
        } else if is_x86_feature_detected!("avx") {
            return "AVX";
        } else {
            return "SSE2";
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return "NEON";
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return "Scalar (no SIMD)";
    }
}

/// Returns true if SIMD acceleration is available on this platform.
pub fn simd_available() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        true
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_instruction_set() {
        let isa = simd_instruction_set();
        assert!(!isa.is_empty());
        println!("Detected SIMD instruction set: {}", isa);
    }

    #[test]
    fn test_simd_available() {
        let available = simd_available();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(available);
        println!("SIMD available: {}", available);
    }

    #[test]
    fn test_multiply_simd_f32_basic() {
        let a = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result_simd = a.multiply_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_simd.rows {
            for j in 0..result_simd.cols {
                let simd_val = *result_simd.get(i, j).unwrap();
                let naive_val = *result_naive.get(i, j).unwrap();
                assert!(
                    (simd_val - naive_val).abs() < 1e-5,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, simd_val, naive_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_simd_f64_basic() {
        let a = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result_simd = a.multiply_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_simd.rows {
            for j in 0..result_simd.cols {
                let simd_val = *result_simd.get(i, j).unwrap();
                let naive_val = *result_naive.get(i, j).unwrap();
                assert!(
                    (simd_val - naive_val).abs() < 1e-10,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, simd_val, naive_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_simd_f32_large() {
        let size = 128;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_simd = a.multiply_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_simd.rows {
            for j in 0..result_simd.cols {
                let simd_val = *result_simd.get(i, j).unwrap();
                let naive_val = *result_naive.get(i, j).unwrap();
                let rel_diff = if naive_val.abs() > 1e-10 {
                    (simd_val - naive_val).abs() / naive_val.abs()
                } else {
                    (simd_val - naive_val).abs()
                };
                assert!(
                    rel_diff < 1e-4,
                    "Mismatch at ({}, {}): {} vs {} (rel diff: {})",
                    i, j, simd_val, naive_val, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_multiply_simd_f64_large() {
        let size = 128;
        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 100) as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| ((i * 2) % 100) as f64).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_simd = a.multiply_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_simd.rows {
            for j in 0..result_simd.cols {
                let simd_val = *result_simd.get(i, j).unwrap();
                let naive_val = *result_naive.get(i, j).unwrap();
                let rel_diff = if naive_val.abs() > 1e-10 {
                    (simd_val - naive_val).abs() / naive_val.abs()
                } else {
                    (simd_val - naive_val).abs()
                };
                assert!(
                    rel_diff < 1e-10,
                    "Mismatch at ({}, {}): {} vs {} (rel diff: {})",
                    i, j, simd_val, naive_val, rel_diff
                );
            }
        }
    }

    #[test]
    fn test_multiply_simd_non_aligned() {
        // Test with sizes that don't align to SIMD width
        let a = Matrix::from_vec(5, 7, (0..35).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(7, 3, (0..21).map(|i| i as f32).collect()).unwrap();

        let result_simd = a.multiply_simd(&b).unwrap();
        let result_naive = a.multiply(&b).unwrap();

        for i in 0..result_simd.rows {
            for j in 0..result_simd.cols {
                let simd_val = *result_simd.get(i, j).unwrap();
                let naive_val = *result_naive.get(i, j).unwrap();
                assert!(
                    (simd_val - naive_val).abs() < 1e-4,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, simd_val, naive_val
                );
            }
        }
    }

    #[test]
    fn test_multiply_simd_dimension_mismatch() {
        let a: Matrix<f32> = Matrix::new(2, 3);
        let b: Matrix<f32> = Matrix::new(2, 2);
        assert!(a.multiply_simd(&b).is_err());

        let a: Matrix<f64> = Matrix::new(2, 3);
        let b: Matrix<f64> = Matrix::new(2, 2);
        assert!(a.multiply_simd(&b).is_err());
    }

    #[test]
    fn test_multiply_simd_identity_f32() {
        let size = 64;
        let mut identity_data = vec![0.0_f32; size * size];
        for i in 0..size {
            identity_data[i * size + i] = 1.0;
        }

        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 50) as f32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let identity = Matrix::from_vec(size, size, identity_data).unwrap();

        let result = a.multiply_simd(&identity).unwrap();

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

    #[test]
    fn test_multiply_simd_identity_f64() {
        let size = 64;
        let mut identity_data = vec![0.0_f64; size * size];
        for i in 0..size {
            identity_data[i * size + i] = 1.0;
        }

        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 50) as f64).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let identity = Matrix::from_vec(size, size, identity_data).unwrap();

        let result = a.multiply_simd(&identity).unwrap();

        for i in 0..size {
            for j in 0..size {
                let result_val = *result.get(i, j).unwrap();
                let a_val = *a.get(i, j).unwrap();
                assert!(
                    (result_val - a_val).abs() < 1e-10,
                    "Identity multiplication failed at ({}, {})",
                    i, j
                );
            }
        }
    }
}

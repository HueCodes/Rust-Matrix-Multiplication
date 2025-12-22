mod parallel;
mod simd;

pub use parallel::thread_count;
pub use simd::{simd_available, simd_instruction_set};

/// A generic matrix structure for numerical computations
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Clone + Default> Matrix<T> {
    /// Create a new matrix with default values
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    /// Create a matrix from a vector of data
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} does not match matrix dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Get an element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(row * self.cols + col)
    }

    /// Set an element at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!("Index out of bounds: ({}, {})", row, col));
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }
}

impl<T> Matrix<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    /// Naive matrix multiplication (O(n³))
    pub fn multiply(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        const TILE_THRESHOLD: usize = 128;
        const TILE_SIZE: usize = 32;

        if self.rows >= TILE_THRESHOLD
            && self.cols >= TILE_THRESHOLD
            && other.cols >= TILE_THRESHOLD
        {
            return Ok(self.multiply_tiled(other, TILE_SIZE));
        }

        let mut result: Matrix<T> = Matrix::new(self.rows, other.cols);
        let a_cols = self.cols;
        let b_cols = other.cols;

        for i in 0..self.rows {
            for j in 0..b_cols {
                let mut sum = T::default();
                let mut a_idx = i * a_cols;
                let mut b_idx = j;

                for _ in 0..a_cols {
                    // Direct index math avoids repeated bounds checks from `get`.
                    let a_val = self.data[a_idx].clone();
                    let b_val = other.data[b_idx].clone();
                    sum = sum + (a_val * b_val);
                    a_idx += 1;
                    b_idx += b_cols;
                }

                result.data[i * b_cols + j] = sum;
            }
        }

        Ok(result)
    }

    fn multiply_tiled(&self, other: &Matrix<T>, tile: usize) -> Matrix<T> {
        debug_assert!(tile > 0);

        let mut result: Matrix<T> = Matrix::new(self.rows, other.cols);
        let rows = self.rows;
        let shared = self.cols;
        let other_cols = other.cols;

        for ii in (0..rows).step_by(tile) {
            let i_max = (ii + tile).min(rows);
            for kk in (0..shared).step_by(tile) {
                let k_max = (kk + tile).min(shared);
                for jj in (0..other_cols).step_by(tile) {
                    let j_max = (jj + tile).min(other_cols);

                    for i in ii..i_max {
                        let row_offset_a = i * shared;
                        let row_offset_res = i * other_cols;

                        for j in jj..j_max {
                            let mut sum = result.data[row_offset_res + j].clone();

                            for k in kk..k_max {
                                let a_val = self.data[row_offset_a + k].clone();
                                let b_val = other.data[k * other_cols + j].clone();
                                sum = sum + (a_val * b_val);
                            }

                            result.data[row_offset_res + j] = sum;
                        }
                    }
                }
            }
        }

        result
    }

    /// Strassen's algorithm for matrix multiplication (faster for large matrices)
    ///
    /// Strassen's algorithm reduces the complexity from O(n³) to O(n^2.807) by using
    /// 7 recursive multiplications instead of 8. This is beneficial for large matrices
    /// (typically n > 512) but has overhead for smaller matrices.
    ///
    /// The algorithm:
    /// 1. Pads matrices to the nearest power of 2 if needed
    /// 2. Divides matrices into 4 quadrants recursively
    /// 3. Computes 7 products (M1-M7) using specific combinations
    /// 4. Combines results to form the output matrix
    /// 5. Uses standard multiplication below a threshold to avoid excessive recursion
    pub fn multiply_strassen(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // Base case threshold - use standard multiplication for small matrices
        // Testing shows Strassen becomes beneficial around 512x512
        const STRASSEN_THRESHOLD: usize = 128;

        if self.rows <= STRASSEN_THRESHOLD
            || self.cols <= STRASSEN_THRESHOLD
            || other.cols <= STRASSEN_THRESHOLD {
            return self.multiply(other);
        }

        // For non-square or non-power-of-2 matrices, pad to next power of 2
        let max_dim = self.rows.max(self.cols).max(other.cols);
        let padded_size = max_dim.next_power_of_two();

        // Only pad if necessary
        if self.rows == padded_size && self.cols == padded_size
            && other.rows == padded_size && other.cols == padded_size
            && padded_size.is_power_of_two() {
            // Already suitable dimensions, use direct Strassen
            return Ok(self.strassen_recursive(other, STRASSEN_THRESHOLD));
        }

        // Pad matrices to power of 2
        let a_padded = self.pad_to_size(padded_size, padded_size);
        let b_padded = other.pad_to_size(padded_size, padded_size);

        // Perform Strassen on padded matrices
        let result_padded = a_padded.strassen_recursive(&b_padded, STRASSEN_THRESHOLD);

        // Extract the actual result (remove padding)
        Ok(result_padded.submatrix(0, self.rows, 0, other.cols))
    }

    /// Internal recursive Strassen implementation
    ///
    /// Assumes square matrices with power-of-2 dimensions
    fn strassen_recursive(&self, other: &Matrix<T>, threshold: usize) -> Matrix<T> {
        let n = self.rows;

        // Base case: use standard multiplication
        if n <= threshold {
            return self.multiply(other).unwrap();
        }

        // Divide matrices into quadrants
        let mid = n / 2;

        // Extract quadrants of A
        let a11 = self.submatrix(0, mid, 0, mid);
        let a12 = self.submatrix(0, mid, mid, n);
        let a21 = self.submatrix(mid, n, 0, mid);
        let a22 = self.submatrix(mid, n, mid, n);

        // Extract quadrants of B
        let b11 = other.submatrix(0, mid, 0, mid);
        let b12 = other.submatrix(0, mid, mid, n);
        let b21 = other.submatrix(mid, n, 0, mid);
        let b22 = other.submatrix(mid, n, mid, n);

        // Compute the 7 Strassen products
        // M1 = (A11 + A22) * (B11 + B22)
        let m1 = a11.add(&a22).unwrap()
            .strassen_recursive(&b11.add(&b22).unwrap(), threshold);

        // M2 = (A21 + A22) * B11
        let m2 = a21.add(&a22).unwrap()
            .strassen_recursive(&b11, threshold);

        // M3 = A11 * (B12 - B22)
        let m3 = a11.strassen_recursive(
            &b12.subtract(&b22).unwrap(), threshold);

        // M4 = A22 * (B21 - B11)
        let m4 = a22.strassen_recursive(
            &b21.subtract(&b11).unwrap(), threshold);

        // M5 = (A11 + A12) * B22
        let m5 = a11.add(&a12).unwrap()
            .strassen_recursive(&b22, threshold);

        // M6 = (A21 - A11) * (B11 + B12)
        let m6 = a21.subtract(&a11).unwrap()
            .strassen_recursive(&b11.add(&b12).unwrap(), threshold);

        // M7 = (A12 - A22) * (B21 + B22)
        let m7 = a12.subtract(&a22).unwrap()
            .strassen_recursive(&b21.add(&b22).unwrap(), threshold);

        // Compute result quadrants
        // C11 = M1 + M4 - M5 + M7
        let c11 = m1.add(&m4).unwrap()
            .subtract(&m5).unwrap()
            .add(&m7).unwrap();

        // C12 = M3 + M5
        let c12 = m3.add(&m5).unwrap();

        // C21 = M2 + M4
        let c21 = m2.add(&m4).unwrap();

        // C22 = M1 - M2 + M3 + M6
        let c22 = m1.subtract(&m2).unwrap()
            .add(&m3).unwrap()
            .add(&m6).unwrap();

        // Combine quadrants into result matrix
        let mut result = Matrix::new(n, n);
        result.copy_submatrix_into(&c11, 0, 0);
        result.copy_submatrix_into(&c12, 0, mid);
        result.copy_submatrix_into(&c21, mid, 0);
        result.copy_submatrix_into(&c22, mid, mid);

        result
    }

    /// Pad matrix to specified dimensions with default values
    fn pad_to_size(&self, new_rows: usize, new_cols: usize) -> Matrix<T> {
        let mut result = Matrix::new(new_rows, new_cols);

        // Copy existing data
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i * new_cols + j] = self.data[i * self.cols + j].clone();
            }
        }

        result
    }
}

impl<T> Matrix<T>
where
    T: Clone + Default + std::ops::Add<Output = T>,
{
    /// Add two matrices element-wise
    pub fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot add with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i].clone() + other.data[i].clone();
        }

        Ok(result)
    }
}

impl<T> Matrix<T>
where
    T: Clone + Default + std::ops::Sub<Output = T>,
{
    /// Subtract two matrices element-wise
    pub fn subtract(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot subtract {}x{} from {}x{}",
                self.rows, self.cols, other.rows, other.cols, self.rows, self.cols
            ));
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i].clone() - other.data[i].clone();
        }

        Ok(result)
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    /// Transpose the matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix {
            rows: self.cols,
            cols: self.rows,
            data: Vec::with_capacity(self.data.len()),
        };

        for j in 0..self.cols {
            for i in 0..self.rows {
                result.data.push(self.data[i * self.cols + j].clone());
            }
        }

        result
    }

    /// Extract a submatrix from this matrix
    ///
    /// Returns a new matrix containing elements from [row_start..row_end, col_start..col_end)
    pub fn submatrix(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Matrix<T> {
        let sub_rows = row_end - row_start;
        let sub_cols = col_end - col_start;
        let mut data = Vec::with_capacity(sub_rows * sub_cols);

        for i in row_start..row_end {
            for j in col_start..col_end {
                data.push(self.data[i * self.cols + j].clone());
            }
        }

        Matrix {
            rows: sub_rows,
            cols: sub_cols,
            data,
        }
    }

    /// Copy a submatrix into this matrix at the specified position
    ///
    /// Copies all elements from `source` into this matrix starting at (row_offset, col_offset)
    pub fn copy_submatrix_into(&mut self, source: &Matrix<T>, row_offset: usize, col_offset: usize) {
        for i in 0..source.rows {
            for j in 0..source.cols {
                let target_idx = (row_offset + i) * self.cols + (col_offset + j);
                self.data[target_idx] = source.data[i * source.cols + j].clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<i32> = Matrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 9);
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let m = Matrix::from_vec(2, 3, data).unwrap();
        assert_eq!(m.get(0, 0), Some(&1));
        assert_eq!(m.get(1, 2), Some(&6));
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m: Matrix<i32> = Matrix::new(2, 2);
        m.set(0, 0, 10).unwrap();
        m.set(1, 1, 20).unwrap();
        assert_eq!(m.get(0, 0), Some(&10));
        assert_eq!(m.get(1, 1), Some(&20));
    }

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7, 8, 9, 10, 11, 12]).unwrap();
        let c = a.multiply(&b).unwrap();

        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.get(0, 0), Some(&58));
        assert_eq!(c.get(0, 1), Some(&64));
        assert_eq!(c.get(1, 0), Some(&139));
        assert_eq!(c.get(1, 1), Some(&154));
    }

    #[test]
    fn test_matrix_multiply_dimension_mismatch() {
        let a: Matrix<i32> = Matrix::new(2, 3);
        let b: Matrix<i32> = Matrix::new(2, 2);
        assert!(a.multiply(&b).is_err());
    }

    #[test]
    fn test_matrix_add() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
        let c = a.add(&b).unwrap();

        assert_eq!(c.get(0, 0), Some(&6));
        assert_eq!(c.get(0, 1), Some(&8));
        assert_eq!(c.get(1, 0), Some(&10));
        assert_eq!(c.get(1, 1), Some(&12));
    }

    #[test]
    fn test_matrix_transpose() {
        let a = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = a.transpose();

        assert_eq!(b.rows, 3);
        assert_eq!(b.cols, 2);
        assert_eq!(b.get(0, 0), Some(&1));
        assert_eq!(b.get(0, 1), Some(&4));
        assert_eq!(b.get(1, 0), Some(&2));
        assert_eq!(b.get(1, 1), Some(&5));
        assert_eq!(b.get(2, 0), Some(&3));
        assert_eq!(b.get(2, 1), Some(&6));
    }

    #[test]
    fn test_identity_multiplication() {
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let identity = Matrix::from_vec(2, 2, vec![1, 0, 0, 1]).unwrap();
        let result = a.multiply(&identity).unwrap();

        assert_eq!(result, a);
    }

    #[test]
    fn test_matrix_subtract() {
        let a = Matrix::from_vec(2, 2, vec![10, 8, 6, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 3, 2, 1]).unwrap();
        let c = a.subtract(&b).unwrap();

        assert_eq!(c.get(0, 0), Some(&5));
        assert_eq!(c.get(0, 1), Some(&5));
        assert_eq!(c.get(1, 0), Some(&4));
        assert_eq!(c.get(1, 1), Some(&3));
    }

    #[test]
    fn test_submatrix() {
        let m = Matrix::from_vec(4, 4, vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ]).unwrap();

        let sub = m.submatrix(1, 3, 1, 3);
        assert_eq!(sub.rows, 2);
        assert_eq!(sub.cols, 2);
        assert_eq!(sub.get(0, 0), Some(&6));
        assert_eq!(sub.get(0, 1), Some(&7));
        assert_eq!(sub.get(1, 0), Some(&10));
        assert_eq!(sub.get(1, 1), Some(&11));
    }

    #[test]
    fn test_copy_submatrix_into() {
        let mut dest = Matrix::from_vec(4, 4, vec![0; 16]).unwrap();
        let src = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();

        dest.copy_submatrix_into(&src, 1, 1);

        assert_eq!(dest.get(1, 1), Some(&1));
        assert_eq!(dest.get(1, 2), Some(&2));
        assert_eq!(dest.get(2, 1), Some(&3));
        assert_eq!(dest.get(2, 2), Some(&4));
        assert_eq!(dest.get(0, 0), Some(&0));
    }

    #[test]
    fn test_pad_to_size() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let padded = m.pad_to_size(4, 4);

        assert_eq!(padded.rows, 4);
        assert_eq!(padded.cols, 4);
        assert_eq!(padded.get(0, 0), Some(&1));
        assert_eq!(padded.get(1, 2), Some(&6));
        assert_eq!(padded.get(2, 0), Some(&0));
        assert_eq!(padded.get(3, 3), Some(&0));
    }

    #[test]
    fn test_strassen_small_matrix() {
        // Test with a small matrix (below threshold)
        let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        assert_eq!(result_naive, result_strassen);
    }

    #[test]
    fn test_strassen_power_of_2() {
        // Test with power-of-2 dimensions
        let a = Matrix::from_vec(4, 4, vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ]).unwrap();
        let b = Matrix::from_vec(4, 4, vec![
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1,
        ]).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        assert_eq!(result_naive, result_strassen);
    }

    #[test]
    fn test_strassen_non_power_of_2() {
        // Test with non-power-of-2 dimensions (requires padding)
        let a = Matrix::from_vec(3, 3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]).unwrap();
        let b = Matrix::from_vec(3, 3, vec![
            9, 8, 7,
            6, 5, 4,
            3, 2, 1,
        ]).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        assert_eq!(result_naive, result_strassen);
    }

    #[test]
    fn test_strassen_rectangular() {
        // Test with rectangular matrices
        let a = Matrix::from_vec(3, 4, vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
        ]).unwrap();
        let b = Matrix::from_vec(4, 2, vec![
            2, 1,
            4, 3,
            6, 5,
            8, 7,
        ]).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        assert_eq!(result_naive, result_strassen);
    }

    #[test]
    fn test_strassen_large_matrix() {
        // Test with larger matrix to actually use Strassen algorithm
        let size = 256;
        let a_data: Vec<i32> = (0..size * size).map(|i| (i % 100) as i32).collect();
        let b_data: Vec<i32> = (0..size * size).map(|i| ((i * 2) % 100) as i32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let b = Matrix::from_vec(size, size, b_data).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        assert_eq!(result_naive, result_strassen);
    }

    #[test]
    fn test_strassen_with_floats() {
        // Test with floating point numbers
        let a: Matrix<f64> = Matrix::from_vec(4, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]).unwrap();
        let b: Matrix<f64> = Matrix::from_vec(4, 4, vec![
            0.5, 0.25, 0.125, 0.0625,
            1.0, 0.5, 0.25, 0.125,
            1.5, 0.75, 0.375, 0.1875,
            2.0, 1.0, 0.5, 0.25,
        ]).unwrap();

        let result_naive = a.multiply(&b).unwrap();
        let result_strassen = a.multiply_strassen(&b).unwrap();

        // For floats, check approximate equality
        for i in 0..result_naive.rows {
            for j in 0..result_naive.cols {
                let naive_val = *result_naive.get(i, j).unwrap();
                let strassen_val = *result_strassen.get(i, j).unwrap();
                let diff = (naive_val - strassen_val).abs();
                assert!(diff < 1e-10,
                    "Mismatch at ({}, {}): {} vs {}", i, j, naive_val, strassen_val);
            }
        }
    }

    #[test]
    fn test_strassen_identity() {
        // Test Strassen with identity matrix
        let size = 128;
        let mut identity_data = vec![0; size * size];
        for i in 0..size {
            identity_data[i * size + i] = 1;
        }

        let a_data: Vec<i32> = (0..size * size).map(|i| (i % 50) as i32).collect();

        let a = Matrix::from_vec(size, size, a_data).unwrap();
        let identity = Matrix::from_vec(size, size, identity_data).unwrap();

        let result = a.multiply_strassen(&identity).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_strassen_dimension_mismatch() {
        let a: Matrix<i32> = Matrix::new(128, 256);
        let b: Matrix<i32> = Matrix::new(128, 256);

        assert!(a.multiply_strassen(&b).is_err());
    }
}

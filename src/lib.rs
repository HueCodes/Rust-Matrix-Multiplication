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

    /// Multiply matrices, storing the result in a pre-allocated matrix.
    ///
    /// This method allows reusing an existing matrix buffer, avoiding allocation
    /// when performing repeated multiplications with the same output dimensions.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Matrix dimensions don't match for multiplication
    /// - Result matrix dimensions don't match expected output
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let b = Matrix::from_vec(3, 2, vec![7, 8, 9, 10, 11, 12]).unwrap();
    /// let mut result: Matrix<i32> = Matrix::new(2, 2);
    ///
    /// a.multiply_into(&b, &mut result).unwrap();
    /// assert_eq!(result[(0, 0)], 58);
    /// assert_eq!(result[(1, 1)], 154);
    /// ```
    pub fn multiply_into(&self, other: &Matrix<T>, result: &mut Matrix<T>) -> Result<(), String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        if result.rows != self.rows || result.cols != other.cols {
            return Err(format!(
                "Result matrix dimensions {}x{} don't match expected {}x{}",
                result.rows, result.cols, self.rows, other.cols
            ));
        }

        let a_cols = self.cols;
        let b_cols = other.cols;

        // Zero out the result matrix
        for elem in &mut result.data {
            *elem = T::default();
        }

        for i in 0..self.rows {
            for j in 0..b_cols {
                let mut sum = T::default();
                let mut a_idx = i * a_cols;
                let mut b_idx = j;

                for _ in 0..a_cols {
                    let a_val = self.data[a_idx].clone();
                    let b_val = other.data[b_idx].clone();
                    sum = sum + (a_val * b_val);
                    a_idx += 1;
                    b_idx += b_cols;
                }

                result.data[i * b_cols + j] = sum;
            }
        }

        Ok(())
    }

    /// Multiply matrices, accumulating into a pre-allocated matrix.
    ///
    /// Unlike `multiply_into`, this adds to the existing values in the result
    /// matrix rather than overwriting them. Useful for implementing blocked
    /// algorithms or accumulating multiple products.
    ///
    /// # Errors
    /// Returns an error if matrix dimensions don't match.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![1, 0, 0, 1]).unwrap(); // Identity
    /// let mut result = Matrix::from_vec(2, 2, vec![10, 10, 10, 10]).unwrap();
    ///
    /// a.multiply_accumulate(&b, &mut result).unwrap();
    /// assert_eq!(result[(0, 0)], 11); // 10 + 1
    /// assert_eq!(result[(1, 1)], 14); // 10 + 4
    /// ```
    pub fn multiply_accumulate(&self, other: &Matrix<T>, result: &mut Matrix<T>) -> Result<(), String>
    where
        T: std::ops::AddAssign,
    {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        if result.rows != self.rows || result.cols != other.cols {
            return Err(format!(
                "Result matrix dimensions {}x{} don't match expected {}x{}",
                result.rows, result.cols, self.rows, other.cols
            ));
        }

        let a_cols = self.cols;
        let b_cols = other.cols;

        for i in 0..self.rows {
            for j in 0..b_cols {
                let mut sum = T::default();
                let mut a_idx = i * a_cols;
                let mut b_idx = j;

                for _ in 0..a_cols {
                    let a_val = self.data[a_idx].clone();
                    let b_val = other.data[b_idx].clone();
                    sum = sum + (a_val * b_val);
                    a_idx += 1;
                    b_idx += b_cols;
                }

                result.data[i * b_cols + j] += sum;
            }
        }

        Ok(())
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

// =============================================================================
// Index Traits - Enable matrix[(row, col)] syntax
// =============================================================================

impl<T> std::ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    /// Access matrix element using `matrix[(row, col)]` syntax.
    ///
    /// # Panics
    /// Panics if the indices are out of bounds.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// assert_eq!(m[(0, 0)], 1);
    /// assert_eq!(m[(1, 1)], 4);
    /// ```
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows && col < self.cols,
            "Index out of bounds: ({}, {}) for {}x{} matrix", row, col, self.rows, self.cols);
        &self.data[row * self.cols + col]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Matrix<T> {
    /// Mutably access matrix element using `matrix[(row, col)]` syntax.
    ///
    /// # Panics
    /// Panics if the indices are out of bounds.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// m[(0, 0)] = 10;
    /// assert_eq!(m[(0, 0)], 10);
    /// ```
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows && col < self.cols,
            "Index out of bounds: ({}, {}) for {}x{} matrix", row, col, self.rows, self.cols);
        &mut self.data[row * self.cols + col]
    }
}

// =============================================================================
// Display Trait - Pretty printing
// =============================================================================

impl<T: std::fmt::Display> std::fmt::Display for Matrix<T> {
    /// Format matrix for display with aligned columns.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// println!("{}", m);
    /// // Output:
    /// // [1, 2, 3]
    /// // [4, 5, 6]
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[i * self.cols + j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// =============================================================================
// From/Into Conversions
// =============================================================================

impl<T: Clone + Default> From<Vec<Vec<T>>> for Matrix<T> {
    /// Create a matrix from a 2D vector (Vec of rows).
    ///
    /// # Panics
    /// Panics if rows have inconsistent lengths.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let data = vec![
    ///     vec![1, 2, 3],
    ///     vec![4, 5, 6],
    /// ];
    /// let m: Matrix<i32> = data.into();
    /// assert_eq!(m.rows, 2);
    /// assert_eq!(m.cols, 3);
    /// assert_eq!(m[(0, 0)], 1);
    /// ```
    fn from(rows: Vec<Vec<T>>) -> Self {
        if rows.is_empty() {
            return Matrix {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            };
        }

        let num_rows = rows.len();
        let num_cols = rows[0].len();

        // Verify all rows have the same length
        for (i, row) in rows.iter().enumerate() {
            assert!(row.len() == num_cols,
                "Inconsistent row length at row {}: expected {}, got {}", i, num_cols, row.len());
        }

        let mut data = Vec::with_capacity(num_rows * num_cols);
        for row in rows {
            data.extend(row);
        }

        Matrix {
            rows: num_rows,
            cols: num_cols,
            data,
        }
    }
}

impl<T: Clone> From<Matrix<T>> for Vec<Vec<T>> {
    /// Convert a matrix back to a 2D vector (Vec of rows).
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let rows: Vec<Vec<i32>> = m.into();
    /// assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    /// ```
    fn from(matrix: Matrix<T>) -> Self {
        let mut result = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            let start = i * matrix.cols;
            let end = start + matrix.cols;
            result.push(matrix.data[start..end].to_vec());
        }
        result
    }
}

impl<T: Clone + Default, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    /// Create a matrix from a 2D array.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let arr = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ];
    /// let m: Matrix<i32> = arr.into();
    /// assert_eq!(m.rows, 2);
    /// assert_eq!(m.cols, 3);
    /// ```
    fn from(arr: [[T; C]; R]) -> Self {
        let mut data = Vec::with_capacity(R * C);
        for row in arr {
            data.extend(row);
        }
        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }
}

// =============================================================================
// Iterator Support
// =============================================================================

impl<T> Matrix<T> {
    /// Returns an iterator over references to all elements in row-major order.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// let sum: i32 = m.iter().sum();
    /// assert_eq!(sum, 10);
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns an iterator over mutable references to all elements in row-major order.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// for x in m.iter_mut() {
    ///     *x *= 2;
    /// }
    /// assert_eq!(m[(0, 0)], 2);
    /// assert_eq!(m[(1, 1)], 8);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Returns an iterator over a specific row.
    ///
    /// # Panics
    /// Panics if the row index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let row1: Vec<&i32> = m.row(1).collect();
    /// assert_eq!(row1, vec![&4, &5, &6]);
    /// ```
    #[inline]
    pub fn row(&self, row: usize) -> impl Iterator<Item = &T> {
        assert!(row < self.rows, "Row index {} out of bounds for {} rows", row, self.rows);
        let start = row * self.cols;
        self.data[start..start + self.cols].iter()
    }

    /// Returns an iterator over a specific column.
    ///
    /// # Panics
    /// Panics if the column index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let col1: Vec<&i32> = m.col(1).collect();
    /// assert_eq!(col1, vec![&2, &5]);
    /// ```
    #[inline]
    pub fn col(&self, col: usize) -> impl Iterator<Item = &T> + '_ {
        assert!(col < self.cols, "Column index {} out of bounds for {} cols", col, self.cols);
        (0..self.rows).map(move |row| &self.data[row * self.cols + col])
    }

    /// Returns an iterator over all rows as slices.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// for row in m.rows_iter() {
    ///     println!("{:?}", row);
    /// }
    /// ```
    #[inline]
    pub fn rows_iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols)
    }

    /// Returns the total number of elements in the matrix.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::new(3, 4);
    /// assert_eq!(m.len(), 12);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the matrix has no elements.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::new(0, 0);
    /// assert!(m.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the dimensions as a tuple (rows, cols).
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::new(3, 4);
    /// assert_eq!(m.shape(), (3, 4));
    /// ```
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns true if the matrix is square.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let square: Matrix<i32> = Matrix::new(3, 3);
    /// let rect: Matrix<i32> = Matrix::new(2, 3);
    /// assert!(square.is_square());
    /// assert!(!rect.is_square());
    /// ```
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    /// Consumes the matrix and returns an iterator over all elements.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// let v: Vec<i32> = m.into_iter().collect();
    /// assert_eq!(v, vec![1, 2, 3, 4]);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// =============================================================================
// Approximate Equality for Floating Point Matrices
// =============================================================================

impl Matrix<f32> {
    /// Check if two f32 matrices are approximately equal within a tolerance.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    /// assert!(a.approx_eq(&b, 1e-6));
    /// ```
    pub fn approx_eq(&self, other: &Matrix<f32>, tolerance: f32) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() <= tolerance)
    }

    /// Check if two f32 matrices are approximately equal with relative tolerance.
    ///
    /// For each pair of elements, checks if |a - b| <= tolerance * max(|a|, |b|).
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1000.0_f32, 2000.0, 3000.0, 4000.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![1000.1_f32, 2000.2, 3000.3, 4000.4]).unwrap();
    /// assert!(a.approx_eq_relative(&b, 1e-3));
    /// ```
    pub fn approx_eq_relative(&self, other: &Matrix<f32>, tolerance: f32) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter()).all(|(a, b)| {
            let max_abs = a.abs().max(b.abs());
            if max_abs == 0.0 {
                true
            } else {
                (a - b).abs() <= tolerance * max_abs
            }
        })
    }
}

impl Matrix<f64> {
    /// Check if two f64 matrices are approximately equal within a tolerance.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// assert!(a.approx_eq(&b, 1e-10));
    /// ```
    pub fn approx_eq(&self, other: &Matrix<f64>, tolerance: f64) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() <= tolerance)
    }

    /// Check if two f64 matrices are approximately equal with relative tolerance.
    ///
    /// For each pair of elements, checks if |a - b| <= tolerance * max(|a|, |b|).
    pub fn approx_eq_relative(&self, other: &Matrix<f64>, tolerance: f64) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter()).all(|(a, b)| {
            let max_abs = a.abs().max(b.abs());
            if max_abs == 0.0 {
                true
            } else {
                (a - b).abs() <= tolerance * max_abs
            }
        })
    }
}

// =============================================================================
// In-Place Operations (Memory Efficient)
// =============================================================================

impl<T: Clone> Matrix<T> {
    /// Transpose the matrix in-place (only for square matrices).
    ///
    /// This operation avoids allocating a new matrix by swapping elements.
    /// For non-square matrices, use `transpose()` which returns a new matrix.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(3, 3, vec![
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9,
    /// ]).unwrap();
    /// m.transpose_mut();
    /// assert_eq!(m[(0, 1)], 4);
    /// assert_eq!(m[(1, 0)], 2);
    /// ```
    pub fn transpose_mut(&mut self) {
        assert!(self.is_square(), "transpose_mut requires a square matrix");
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                self.data.swap(i * n + j, j * n + i);
            }
        }
    }
}

impl<T> Matrix<T>
where
    T: Clone + std::ops::AddAssign,
{
    /// Add another matrix to this one in-place (+=).
    ///
    /// This avoids allocating a new result matrix.
    ///
    /// # Errors
    /// Returns an error if the matrices have different dimensions.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut a = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![5, 6, 7, 8]).unwrap();
    /// a.add_assign(&b).unwrap();
    /// assert_eq!(a[(0, 0)], 6);
    /// assert_eq!(a[(1, 1)], 12);
    /// ```
    pub fn add_assign(&mut self, other: &Matrix<T>) -> Result<(), String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot add with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b.clone();
        }
        Ok(())
    }
}

impl<T> Matrix<T>
where
    T: Clone + std::ops::SubAssign,
{
    /// Subtract another matrix from this one in-place (-=).
    ///
    /// This avoids allocating a new result matrix.
    ///
    /// # Errors
    /// Returns an error if the matrices have different dimensions.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut a = Matrix::from_vec(2, 2, vec![10, 20, 30, 40]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// a.sub_assign(&b).unwrap();
    /// assert_eq!(a[(0, 0)], 9);
    /// assert_eq!(a[(1, 1)], 36);
    /// ```
    pub fn sub_assign(&mut self, other: &Matrix<T>) -> Result<(), String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot subtract {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b.clone();
        }
        Ok(())
    }
}

impl<T> Matrix<T>
where
    T: Clone + std::ops::MulAssign,
{
    /// Scale all elements in-place by a scalar value.
    ///
    /// This avoids allocating a new result matrix.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// m.scale_mut(3);
    /// assert_eq!(m[(0, 0)], 3);
    /// assert_eq!(m[(1, 1)], 12);
    /// ```
    pub fn scale_mut(&mut self, scalar: T) {
        for elem in &mut self.data {
            *elem *= scalar.clone();
        }
    }
}

// =============================================================================
// MatrixView - Zero-Copy Submatrix References
// =============================================================================

/// A view into a submatrix without copying data.
///
/// `MatrixView` provides read-only access to a rectangular region of a matrix
/// without allocating new memory. This is useful for algorithms that need to
/// work with submatrices, such as block matrix operations.
///
/// # Example
/// ```
/// use matrix_multiply::{Matrix, MatrixView};
///
/// let m = Matrix::from_vec(4, 4, (0..16).collect()).unwrap();
/// let view = MatrixView::new(&m, 1, 3, 1, 3); // 2x2 submatrix
/// assert_eq!(view.get(0, 0), Some(&5));
/// assert_eq!(view.get(1, 1), Some(&10));
/// ```
#[derive(Debug)]
pub struct MatrixView<'a, T> {
    matrix: &'a Matrix<T>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

impl<'a, T> MatrixView<'a, T> {
    /// Create a new view into a submatrix.
    ///
    /// # Arguments
    /// * `matrix` - The source matrix
    /// * `row_start` - Starting row (inclusive)
    /// * `row_end` - Ending row (exclusive)
    /// * `col_start` - Starting column (inclusive)
    /// * `col_end` - Ending column (exclusive)
    ///
    /// # Panics
    /// Panics if the bounds are invalid.
    pub fn new(
        matrix: &'a Matrix<T>,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        assert!(row_start <= row_end, "row_start must be <= row_end");
        assert!(col_start <= col_end, "col_start must be <= col_end");
        assert!(row_end <= matrix.rows, "row_end exceeds matrix rows");
        assert!(col_end <= matrix.cols, "col_end exceeds matrix cols");

        MatrixView {
            matrix,
            row_start,
            row_end,
            col_start,
            col_end,
        }
    }

    /// Returns the number of rows in the view.
    #[inline]
    pub fn rows(&self) -> usize {
        self.row_end - self.row_start
    }

    /// Returns the number of columns in the view.
    #[inline]
    pub fn cols(&self) -> usize {
        self.col_end - self.col_start
    }

    /// Returns the shape as (rows, cols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Get an element at position (row, col) relative to the view.
    ///
    /// Returns None if out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows() || col >= self.cols() {
            return None;
        }
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        Some(&self.matrix.data[actual_row * self.matrix.cols + actual_col])
    }

    /// Returns an iterator over elements in the view in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (self.row_start..self.row_end).flat_map(move |i| {
            (self.col_start..self.col_end)
                .map(move |j| &self.matrix.data[i * self.matrix.cols + j])
        })
    }

    /// Returns an iterator over a specific row in the view.
    pub fn row(&self, row: usize) -> impl Iterator<Item = &T> + '_ {
        assert!(row < self.rows(), "Row index out of bounds");
        let actual_row = self.row_start + row;
        let start = actual_row * self.matrix.cols + self.col_start;
        self.matrix.data[start..start + self.cols()].iter()
    }

    /// Returns an iterator over a specific column in the view.
    pub fn col(&self, col: usize) -> impl Iterator<Item = &T> + '_ {
        assert!(col < self.cols(), "Column index out of bounds");
        let actual_col = self.col_start + col;
        (self.row_start..self.row_end)
            .map(move |i| &self.matrix.data[i * self.matrix.cols + actual_col])
    }
}

impl<'a, T> std::ops::Index<(usize, usize)> for MatrixView<'a, T> {
    type Output = T;

    /// Access element using `view[(row, col)]` syntax.
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows() && col < self.cols(),
            "Index out of bounds: ({}, {}) for {}x{} view", row, col, self.rows(), self.cols());
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        &self.matrix.data[actual_row * self.matrix.cols + actual_col]
    }
}

impl<'a, T: Clone + Default> MatrixView<'a, T> {
    /// Convert the view to an owned Matrix.
    ///
    /// This creates a copy of the viewed data.
    pub fn to_matrix(&self) -> Matrix<T> {
        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for i in self.row_start..self.row_end {
            for j in self.col_start..self.col_end {
                data.push(self.matrix.data[i * self.matrix.cols + j].clone());
            }
        }
        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data,
        }
    }
}

/// A mutable view into a submatrix without copying data.
///
/// `MatrixViewMut` provides read-write access to a rectangular region of a matrix
/// without allocating new memory.
///
/// # Example
/// ```
/// use matrix_multiply::{Matrix, MatrixViewMut};
///
/// let mut m = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
/// let mut view = MatrixViewMut::new(&mut m, 0, 2, 0, 2);
/// view[(0, 0)] = 100;
/// assert_eq!(m[(0, 0)], 100);
/// ```
#[derive(Debug)]
pub struct MatrixViewMut<'a, T> {
    matrix: &'a mut Matrix<T>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// Create a new mutable view into a submatrix.
    pub fn new(
        matrix: &'a mut Matrix<T>,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        assert!(row_start <= row_end, "row_start must be <= row_end");
        assert!(col_start <= col_end, "col_start must be <= col_end");
        assert!(row_end <= matrix.rows, "row_end exceeds matrix rows");
        assert!(col_end <= matrix.cols, "col_end exceeds matrix cols");

        MatrixViewMut {
            matrix,
            row_start,
            row_end,
            col_start,
            col_end,
        }
    }

    /// Returns the number of rows in the view.
    #[inline]
    pub fn rows(&self) -> usize {
        self.row_end - self.row_start
    }

    /// Returns the number of columns in the view.
    #[inline]
    pub fn cols(&self) -> usize {
        self.col_end - self.col_start
    }

    /// Returns the shape as (rows, cols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Get an element at position (row, col) relative to the view.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows() || col >= self.cols() {
            return None;
        }
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        Some(&self.matrix.data[actual_row * self.matrix.cols + actual_col])
    }

    /// Get a mutable reference to an element at position (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows() || col >= self.cols() {
            return None;
        }
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        let cols = self.matrix.cols;
        Some(&mut self.matrix.data[actual_row * cols + actual_col])
    }

    /// Set an element at position (row, col) relative to the view.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), String> {
        if row >= self.rows() || col >= self.cols() {
            return Err(format!("Index out of bounds: ({}, {})", row, col));
        }
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        self.matrix.data[actual_row * self.matrix.cols + actual_col] = value;
        Ok(())
    }
}

impl<'a, T> std::ops::Index<(usize, usize)> for MatrixViewMut<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows() && col < self.cols(),
            "Index out of bounds: ({}, {}) for {}x{} view", row, col, self.rows(), self.cols());
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        &self.matrix.data[actual_row * self.matrix.cols + actual_col]
    }
}

impl<'a, T> std::ops::IndexMut<(usize, usize)> for MatrixViewMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows() && col < self.cols(),
            "Index out of bounds: ({}, {}) for {}x{} view", row, col, self.rows(), self.cols());
        let actual_row = self.row_start + row;
        let actual_col = self.col_start + col;
        &mut self.matrix.data[actual_row * self.matrix.cols + actual_col]
    }
}

impl<T> Matrix<T> {
    /// Create an immutable view into a submatrix.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(4, 4, (0..16).collect()).unwrap();
    /// let view = m.view(1, 3, 1, 3);
    /// assert_eq!(view.rows(), 2);
    /// assert_eq!(view.cols(), 2);
    /// ```
    pub fn view(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> MatrixView<'_, T> {
        MatrixView::new(self, row_start, row_end, col_start, col_end)
    }

    /// Create a mutable view into a submatrix.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    /// {
    ///     let mut view = m.view_mut(0, 2, 0, 2);
    ///     view[(0, 0)] = 100;
    /// }
    /// assert_eq!(m[(0, 0)], 100);
    /// ```
    pub fn view_mut(&mut self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> MatrixViewMut<'_, T> {
        MatrixViewMut::new(self, row_start, row_end, col_start, col_end)
    }
}

// =============================================================================
// Additional Utility Methods
// =============================================================================

impl<T: Clone + Default> Matrix<T> {
    /// Create an identity matrix of the given size.
    ///
    /// Note: Requires T to implement From<u8> for setting diagonal to 1.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let identity: Matrix<i32> = Matrix::identity(3);
    /// assert_eq!(identity[(0, 0)], 1);
    /// assert_eq!(identity[(0, 1)], 0);
    /// assert_eq!(identity[(1, 1)], 1);
    /// ```
    pub fn identity(size: usize) -> Self
    where
        T: From<u8>,
    {
        let mut m = Matrix::new(size, size);
        for i in 0..size {
            m.data[i * size + i] = T::from(1);
        }
        m
    }

    /// Create a matrix filled with a specific value.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::fill(2, 3, 42);
    /// assert!(m.iter().all(|&x| x == 42));
    /// ```
    pub fn fill(rows: usize, cols: usize, value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Create a matrix from a function that computes each element.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::from_fn(3, 3, |i, j| (i * 3 + j) as i32);
    /// assert_eq!(m[(0, 0)], 0);
    /// assert_eq!(m[(1, 1)], 4);
    /// assert_eq!(m[(2, 2)], 8);
    /// ```
    pub fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self
    where
        F: Fn(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
    }

    /// Map a function over all elements, returning a new matrix.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// let doubled = m.map(|x| x * 2);
    /// assert_eq!(doubled[(0, 0)], 2);
    /// assert_eq!(doubled[(1, 1)], 8);
    /// ```
    pub fn map<U, F>(&self, f: F) -> Matrix<U>
    where
        F: Fn(&T) -> U,
        U: Clone + Default,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect(),
        }
    }

    /// Apply a function to each element in place.
    ///
    /// # Example
    /// ```
    /// use matrix_multiply::Matrix;
    ///
    /// let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
    /// m.apply(|x| *x *= 2);
    /// assert_eq!(m[(0, 0)], 2);
    /// assert_eq!(m[(1, 1)], 8);
    /// ```
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(&mut T),
    {
        for elem in &mut self.data {
            f(elem);
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

    // =========================================================================
    // Ergonomics Tests
    // =========================================================================

    #[test]
    fn test_index_access() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 2)], 3);
        assert_eq!(m[(1, 0)], 4);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_index_mut_access() {
        let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        m[(0, 0)] = 10;
        m[(1, 1)] = 40;
        assert_eq!(m[(0, 0)], 10);
        assert_eq!(m[(1, 1)], 40);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let _ = m[(2, 0)];
    }

    #[test]
    fn test_display() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let s = format!("{}", m);
        assert!(s.contains("[1, 2]"));
        assert!(s.contains("[3, 4]"));
    }

    #[test]
    fn test_from_vec_vec() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];
        let m: Matrix<i32> = data.into();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_into_vec_vec() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let rows: Vec<Vec<i32>> = m.into();
        assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn test_from_array() {
        let arr = [
            [1, 2, 3],
            [4, 5, 6],
        ];
        let m: Matrix<i32> = arr.into();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m[(1, 1)], 5);
    }

    #[test]
    fn test_iter() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let sum: i32 = m.iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_iter_mut() {
        let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        for x in m.iter_mut() {
            *x *= 2;
        }
        assert_eq!(m[(0, 0)], 2);
        assert_eq!(m[(1, 1)], 8);
    }

    #[test]
    fn test_into_iter() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let v: Vec<i32> = m.into_iter().collect();
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_row_iterator() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let row0: Vec<&i32> = m.row(0).collect();
        let row1: Vec<&i32> = m.row(1).collect();
        assert_eq!(row0, vec![&1, &2, &3]);
        assert_eq!(row1, vec![&4, &5, &6]);
    }

    #[test]
    fn test_col_iterator() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let col0: Vec<&i32> = m.col(0).collect();
        let col1: Vec<&i32> = m.col(1).collect();
        assert_eq!(col0, vec![&1, &4]);
        assert_eq!(col1, vec![&2, &5]);
    }

    #[test]
    fn test_rows_iter() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let rows: Vec<&[i32]> = m.rows_iter().collect();
        assert_eq!(rows[0], &[1, 2, 3]);
        assert_eq!(rows[1], &[4, 5, 6]);
    }

    #[test]
    fn test_len_is_empty() {
        let m: Matrix<i32> = Matrix::new(3, 4);
        assert_eq!(m.len(), 12);
        assert!(!m.is_empty());

        let empty: Matrix<i32> = Matrix::new(0, 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_shape() {
        let m: Matrix<i32> = Matrix::new(3, 4);
        assert_eq!(m.shape(), (3, 4));
    }

    #[test]
    fn test_is_square() {
        let square: Matrix<i32> = Matrix::new(3, 3);
        let rect: Matrix<i32> = Matrix::new(2, 3);
        assert!(square.is_square());
        assert!(!rect.is_square());
    }

    #[test]
    fn test_approx_eq_f32() {
        let a = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let c = Matrix::from_vec(2, 2, vec![1.1_f32, 2.0, 3.0, 4.0]).unwrap();

        assert!(a.approx_eq(&b, 1e-6));
        assert!(!a.approx_eq(&c, 1e-6));
        assert!(a.approx_eq(&c, 0.2));
    }

    #[test]
    fn test_approx_eq_f64() {
        let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

        assert!(a.approx_eq(&b, 1e-10));
    }

    #[test]
    fn test_approx_eq_relative() {
        let a = Matrix::from_vec(2, 2, vec![1000.0_f32, 2000.0, 3000.0, 4000.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1000.1_f32, 2000.2, 3000.3, 4000.4]).unwrap();

        assert!(a.approx_eq_relative(&b, 1e-3));
        assert!(!a.approx_eq_relative(&b, 1e-5));
    }

    #[test]
    fn test_identity() {
        let identity: Matrix<i32> = Matrix::identity(3);
        assert_eq!(identity[(0, 0)], 1);
        assert_eq!(identity[(0, 1)], 0);
        assert_eq!(identity[(1, 1)], 1);
        assert_eq!(identity[(2, 2)], 1);
    }

    #[test]
    fn test_fill() {
        let m: Matrix<i32> = Matrix::fill(2, 3, 42);
        assert!(m.iter().all(|&x| x == 42));
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_from_fn() {
        let m: Matrix<i32> = Matrix::from_fn(3, 3, |i, j| (i * 3 + j) as i32);
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(1, 1)], 4);
        assert_eq!(m[(2, 2)], 8);
    }

    #[test]
    fn test_map() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let doubled = m.map(|x| x * 2);
        assert_eq!(doubled[(0, 0)], 2);
        assert_eq!(doubled[(1, 1)], 8);
    }

    #[test]
    fn test_apply() {
        let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        m.apply(|x| *x *= 2);
        assert_eq!(m[(0, 0)], 2);
        assert_eq!(m[(1, 1)], 8);
    }

    #[test]
    fn test_for_loop_iteration() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut sum = 0;
        for &x in &m {
            sum += x;
        }
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_for_loop_mut_iteration() {
        let mut m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        for x in &mut m {
            *x += 10;
        }
        assert_eq!(m[(0, 0)], 11);
        assert_eq!(m[(1, 1)], 14);
    }
}

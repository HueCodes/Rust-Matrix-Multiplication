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
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
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
    pub fn multiply_strassen(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }

        // For small matrices, use naive multiplication
        if self.rows <= 64 || self.cols <= 64 || other.cols <= 64 {
            return self.multiply(other);
        }

        // TODO: Implement Strassen's algorithm for large matrices
        // For now, fall back to naive implementation
        self.multiply(other)
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
}

use matrix_multiply::Matrix;

fn main() {
    println!("Matrix Multiplication Library - Basic Example\n");

    // Create two matrices
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

    println!("Matrix A (3x3):");
    print_matrix(&a);

    println!("\nMatrix B (3x3):");
    print_matrix(&b);

    // Multiply matrices
    let c = a.multiply(&b).unwrap();
    println!("\nMatrix C = A × B:");
    print_matrix(&c);

    // Transpose
    let a_t = a.transpose();
    println!("\nMatrix A^T (Transpose):");
    print_matrix(&a_t);

    // Addition
    let sum = a.add(&b).unwrap();
    println!("\nMatrix A + B:");
    print_matrix(&sum);
}

fn print_matrix<T: std::fmt::Display + Clone + Default>(matrix: &Matrix<T>) {
    for i in 0..matrix.rows {
        print!("  [");
        for j in 0..matrix.cols {
            print!(" {:3}", matrix.get(i, j).unwrap());
        }
        println!(" ]");
    }
}

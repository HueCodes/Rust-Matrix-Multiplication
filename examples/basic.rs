use matrix_multiply::Matrix;
use std::time::Instant;

fn main() {
    println!("Matrix Multiplication Library - Basic Example\n");
    println!("==============================================\n");

    // Create two small matrices for demonstration
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

    // Basic multiplication
    let c = a.multiply(&b).unwrap();
    println!("\nMatrix C = A × B (Standard Multiplication):");
    print_matrix(&c);

    // Transpose
    let a_t = a.transpose();
    println!("\nMatrix A^T (Transpose):");
    print_matrix(&a_t);

    // Addition
    let sum = a.add(&b).unwrap();
    println!("\nMatrix A + B:");
    print_matrix(&sum);

    // Subtraction
    let diff = a.subtract(&b).unwrap();
    println!("\nMatrix A - B:");
    print_matrix(&diff);

    // Demonstrate Strassen's algorithm with larger matrices
    println!("\n==============================================");
    println!("Strassen's Algorithm Demonstration");
    println!("==============================================\n");

    let size = 256;
    println!("Creating {}x{} matrices...", size, size);
    let large_a = Matrix::from_vec(size, size, vec![1; size * size]).unwrap();
    let large_b = Matrix::from_vec(size, size, vec![2; size * size]).unwrap();

    // Time standard multiplication
    println!("\nRunning standard multiplication...");
    let start = Instant::now();
    let _result_standard = large_a.multiply(&large_b).unwrap();
    let time_standard = start.elapsed();
    println!("  Time: {:?}", time_standard);

    // Time Strassen's algorithm
    println!("\nRunning Strassen's algorithm...");
    let start = Instant::now();
    let _result_strassen = large_a.multiply_strassen(&large_b).unwrap();
    let time_strassen = start.elapsed();
    println!("  Time: {:?}", time_strassen);

    println!("\nPerformance comparison:");
    println!("  Standard: {:?}", time_standard);
    println!("  Strassen: {:?}", time_strassen);
    if time_strassen < time_standard {
        let speedup = time_standard.as_secs_f64() / time_strassen.as_secs_f64();
        println!("  Strassen is {:.2}x faster!", speedup);
    } else {
        let slowdown = time_strassen.as_secs_f64() / time_standard.as_secs_f64();
        println!("  Strassen is {:.2}x slower (expected for this size)", slowdown);
    }

    println!("\nNote: Strassen's algorithm becomes faster than standard");
    println!("multiplication for very large matrices (typically > 512x512).");
    println!("For smaller matrices, the overhead outweighs the benefits.");
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

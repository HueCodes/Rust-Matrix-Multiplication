# Matrix Multiplication Library

A comprehensive matrix multiplication library written in Rust, providing efficient and type-safe matrix operations.

## Features

- **Generic Matrix Type**: Works with any numeric type (i32, f64, etc.)
- **Core Operations**:
  - Matrix multiplication (standard O(n³) and Strassen's O(n^2.807))
  - Matrix addition and subtraction
  - Matrix transposition
  - Element access and modification
- **Performance Optimizations**:
  - Tiled multiplication for cache efficiency
  - Strassen's algorithm for large matrices
  - Automatic algorithm selection based on matrix size
- **Type Safety**: Compile-time dimension checking where possible
- **Well-tested**: Comprehensive test suite included

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
matrix_multiply = "0.1.0"
```

## Usage

```rust
use matrix_multiply::Matrix;

fn main() {
    // Create matrices
    let a = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7, 8, 9, 10, 11, 12]).unwrap();

    // Multiply matrices
    let c = a.multiply(&b).unwrap();

    // Use Strassen's algorithm for large matrices
    let c_fast = a.multiply_strassen(&b).unwrap();

    // Access elements
    println!("Result at (0,0): {:?}", c.get(0, 0));

    // Transpose
    let a_t = a.transpose();

    // Addition and subtraction
    let d = Matrix::from_vec(2, 3, vec![1, 1, 1, 1, 1, 1]).unwrap();
    let sum = a.add(&d).unwrap();
    let diff = a.subtract(&d).unwrap();
}
```

## Examples

Run the basic example:

```bash
cargo run --example basic
```

## Benchmarks

Run benchmarks to compare performance:

```bash
cargo bench
```

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis and results.

## Testing

Run the test suite:

```bash
cargo test
```

See [TEST_RESULTS.md](TEST_RESULTS.md) for comprehensive test coverage details.

## Algorithm Details

### Strassen's Algorithm

Strassen's method reduces complexity from O(n³) to O(n^2.807) by using 7 recursive multiplications instead of 8. The algorithm:

1. Divides matrices into quadrants
2. Computes 7 products (M1-M7) using specific combinations
3. Combines products to form the result matrix

The implementation automatically uses Strassen's algorithm for matrices larger than 128x128 and handles non-power-of-2 dimensions through padding.

## Future Enhancements

- [x] Strassen's algorithm for large matrices
- [ ] Parallel multiplication using rayon
- [ ] SIMD optimizations
- [ ] Sparse matrix support
- [ ] Block matrix multiplication
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] More linear algebra operations (determinant, inverse, etc.)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

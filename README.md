# Matrix Multiplication Library

A comprehensive matrix multiplication library written in Rust, providing efficient and type-safe matrix operations.

## Features

- **Generic Matrix Type**: Works with any numeric type (i32, f64, etc.)
- **Core Operations**:
  - Matrix multiplication (naive O(n³) algorithm)
  - Matrix addition
  - Matrix transposition
  - Element access and modification
- **Type Safety**: Compile-time dimension checking where possible
- **Performance**: Optimized implementations with room for advanced algorithms
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
    
    // Access elements
    println!("Result at (0,0): {:?}", c.get(0, 0));
    
    // Transpose
    let a_t = a.transpose();
    
    // Addition
    let d = Matrix::from_vec(2, 3, vec![1, 1, 1, 1, 1, 1]).unwrap();
    let sum = a.add(&d).unwrap();
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

## Testing

Run the test suite:

```bash
cargo test
```

## Future Enhancements

- [ ] Strassen's algorithm for large matrices
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

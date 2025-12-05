# Benchmark Results

Performance comparison between standard O(n³) multiplication and Strassen's O(n^2.807) algorithm.

## Test Environment

- **Platform**: macOS (Darwin 25.1.0)
- **Compiler**: rustc with release optimizations
- **Build Profile**: release (optimized)
- **Date**: December 4, 2025

## Test Results Summary

All 20 tests passed successfully in 0.04 seconds.

## Performance Benchmarks

### Small Matrices (Below Threshold)

For matrices smaller than 128x128, both algorithms perform identically as Strassen falls back to standard multiplication.

| Size   | Standard (naive) | Strassen | Difference |
|--------|------------------|----------|------------|
| 16x16  | 1.99 µs          | 1.99 µs  | ~0%        |
| 32x32  | 14.89 µs         | 14.92 µs | ~0%        |
| 64x64  | 134.59 µs        | 134.69 µs| ~0%        |

**Conclusion**: For small matrices, overhead is negligible. Both algorithms have equivalent performance.

### Large Matrices (Power of 2 Dimensions)

Strassen's algorithm shows significant performance improvements for large matrices.

| Size      | Standard (naive) | Strassen | Speedup    |
|-----------|------------------|----------|------------|
| 128x128   | 1.16 ms          | 1.12 ms  | 1.04x      |
| 256x256   | 9.06 ms          | 7.03 ms  | **1.29x**  |
| 512x512   | 62.93 ms         | 49.76 ms | **1.26x**  |

**Conclusion**: Strassen achieves 25-30% performance improvement for large power-of-2 matrices.

### Non-Power-of-2 Matrices

Performance with automatic padding to nearest power of 2.

| Size      | Standard (naive) | Strassen | Speedup    | Padded Size |
|-----------|------------------|----------|------------|-------------|
| 100x100   | 500.77 µs        | 499.45 µs| 1.00x      | 128x128     |
| 300x300   | 12.67 ms         | 49.59 ms | **0.26x**  | 512x512     |
| 500x500   | 58.23 ms         | 49.62 ms | 1.17x      | 512x512     |

**Conclusion**: Padding overhead can be significant. At 300x300, padding to 512x512 causes 4x slowdown. The crossover point for non-power-of-2 matrices is higher.

### Rectangular Matrices

Performance with non-square matrices (automatically padded).

| Dimensions       | Standard (naive) | Strassen | Speedup    |
|------------------|------------------|----------|------------|
| 100x200 · 200x150| 1.41 ms          | 1.40 ms  | 1.01x      |
| 256x128 · 128x256| 3.85 ms          | 3.85 ms  | 1.00x      |
| 150x300 · 300x200| 4.16 ms          | 49.91 ms | **0.08x**  |

**Conclusion**: Rectangular matrices with dimensions far from powers of 2 suffer from severe padding overhead.

## Key Findings

### When to Use Strassen's Algorithm

✓ **Use Strassen for**:
- Square matrices larger than 256x256
- Power-of-2 dimensions (512x512, 1024x1024, etc.)
- Matrices close to power-of-2 sizes (480x480, 1000x1000)

✗ **Avoid Strassen for**:
- Small matrices (< 256x256)
- Dimensions far from power of 2 (e.g., 300x300 pads to 512x512)
- Very rectangular matrices where one dimension is much larger

### Performance Characteristics

1. **Optimal Case**: 512x512 matrices show 26% speedup
2. **Threshold**: Algorithm switches at 128x128 to balance overhead
3. **Padding Overhead**: Can dominate for matrices between powers of 2
4. **Asymptotic Benefit**: Improvement increases with larger matrices

### Algorithm Complexity

- **Standard Multiplication**: O(n³) = O(n^3.0)
- **Strassen's Algorithm**: O(n^2.807)
- **Theoretical Speedup**: ~1.3-1.4x for large n

### Memory Usage

- **Standard**: O(1) extra space
- **Strassen**: O(n²) extra space for temporary matrices

## Recommendations

For production use:

1. Use `multiply()` for general-purpose matrix multiplication
2. Use `multiply_strassen()` only when:
   - Matrix size > 512x512
   - Dimensions are power of 2 or close to it
   - Memory usage is not a constraint

The library's automatic threshold switching provides good default behavior for most use cases.

## Benchmark Configuration

Benchmarks use the Criterion.rs framework with:
- 100 samples per benchmark (10 for large matrices)
- 3-second warmup period
- Outlier detection and removal
- Statistical analysis with confidence intervals

All benchmarks measure end-to-end multiplication time including matrix creation overhead.

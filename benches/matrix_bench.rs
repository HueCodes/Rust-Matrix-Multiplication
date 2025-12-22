use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use matrix_multiply::{Matrix, simd_instruction_set, thread_count};

fn bench_matrix_multiply_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply_comparison");

    // Test small matrices (where naive should be faster)
    for size in [16, 32, 64].iter() {
        let a = Matrix::from_vec(*size, *size, vec![1; size * size]).unwrap();
        let b = Matrix::from_vec(*size, *size, vec![2; size * size]).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_large_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_matrices");
    group.sample_size(10); // Reduce sample size for large matrices

    // Test medium to large matrices (where Strassen may be faster)
    for size in [128, 256, 512].iter() {
        let a = Matrix::from_vec(*size, *size, vec![1; size * size]).unwrap();
        let b = Matrix::from_vec(*size, *size, vec![2; size * size]).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_non_power_of_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_power_of_2");

    // Test non-power-of-2 dimensions to measure padding overhead
    for size in [100, 300, 500].iter() {
        let a = Matrix::from_vec(*size, *size, vec![1; size * size]).unwrap();
        let b = Matrix::from_vec(*size, *size, vec![2; size * size]).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("rectangular_matrices");

    // Test rectangular matrices
    let test_cases = vec![
        (100, 200, 150),
        (256, 128, 256),
        (150, 300, 200),
    ];

    for (m, k, n) in test_cases {
        let a = Matrix::from_vec(m, k, vec![1; m * k]).unwrap();
        let b = Matrix::from_vec(k, n, vec![2; k * n]).unwrap();
        let label = format!("{}x{}.{}x{}", m, k, k, n);

        group.bench_with_input(BenchmarkId::new("naive", &label), &label, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", &label), &label, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

// =============================================================================
// SIMD Benchmarks
// =============================================================================

fn bench_simd_f32(c: &mut Criterion) {
    println!("SIMD Instruction Set: {}", simd_instruction_set());

    let mut group = c.benchmark_group("simd_f32");

    // Test various matrix sizes
    for size in [32, 64, 128, 256].iter() {
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_simd_f32_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_f32_large");
    group.sample_size(10);

    for size in [512, 1024].iter() {
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_simd_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_f64");

    for size in [32, 64, 128, 256].iter() {
        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 100) as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| ((i * 2) % 100) as f64).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_simd_f64_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_f64_large");
    group.sample_size(10);

    for size in [512, 1024].iter() {
        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 100) as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| ((i * 2) % 100) as f64).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_simd_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_rectangular");

    let test_cases = vec![
        (100, 200, 150),
        (256, 128, 256),
        (200, 300, 100),
    ];

    for (m, k, n) in test_cases {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(m, k, a_data).unwrap();
        let b = Matrix::from_vec(k, n, b_data).unwrap();
        let label = format!("{}x{}.{}x{}", m, k, k, n);

        group.bench_with_input(BenchmarkId::new("naive", &label), &label, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", &label), &label, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_all_algorithms_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_algorithms");
    group.sample_size(10);

    // Compare all three algorithms on 256x256 matrices
    let size = 256;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

    let a = Matrix::from_vec(size, size, a_data).unwrap();
    let b = Matrix::from_vec(size, size, b_data).unwrap();

    group.bench_function("naive_256x256", |bench| {
        bench.iter(|| {
            black_box(a.multiply(&b).unwrap());
        });
    });

    group.bench_function("simd_256x256", |bench| {
        bench.iter(|| {
            black_box(a.multiply_simd(&b).unwrap());
        });
    });

    group.bench_function("strassen_256x256", |bench| {
        bench.iter(|| {
            black_box(a.multiply_strassen(&b).unwrap());
        });
    });

    group.finish();
}

// =============================================================================
// Parallel Benchmarks
// =============================================================================

fn bench_parallel_f32(c: &mut Criterion) {
    println!("Thread count: {}", thread_count());
    println!("SIMD Instruction Set: {}", simd_instruction_set());

    let mut group = c.benchmark_group("parallel_f32");
    group.sample_size(10);

    for size in [128, 256, 512].iter() {
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel_simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_parallel_simd(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_parallel_f32_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_f32_large");
    group.sample_size(10);

    for size in [1024, 2048].iter() {
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel_simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_parallel_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_strassen(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_parallel_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_f64");
    group.sample_size(10);

    for size in [128, 256, 512].iter() {
        let a_data: Vec<f64> = (0..size * size).map(|i| (i % 100) as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| ((i * 2) % 100) as f64).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_simd(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel_simd", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_parallel_simd(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_parallel_generic(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_generic");
    group.sample_size(10);

    for size in [128, 256, 512].iter() {
        let a_data: Vec<i64> = (0..size * size).map(|i| (i % 100) as i64).collect();
        let b_data: Vec<i64> = (0..size * size).map(|i| ((i * 2) % 100) as i64).collect();

        let a = Matrix::from_vec(*size, *size, a_data).unwrap();
        let b = Matrix::from_vec(*size, *size, b_data).unwrap();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.multiply_parallel(&b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_all_methods_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_methods");
    group.sample_size(10);

    // Compare all methods on 512x512 matrices
    let size = 512;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i * 2) % 100) as f32).collect();

    let a = Matrix::from_vec(size, size, a_data).unwrap();
    let b = Matrix::from_vec(size, size, b_data).unwrap();

    group.bench_function("naive_512x512", |bench| {
        bench.iter(|| {
            black_box(a.multiply(&b).unwrap());
        });
    });

    group.bench_function("simd_512x512", |bench| {
        bench.iter(|| {
            black_box(a.multiply_simd(&b).unwrap());
        });
    });

    group.bench_function("parallel_simd_512x512", |bench| {
        bench.iter(|| {
            black_box(a.multiply_parallel_simd(&b).unwrap());
        });
    });

    group.bench_function("strassen_512x512", |bench| {
        bench.iter(|| {
            black_box(a.multiply_strassen(&b).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_multiply_comparison,
    bench_large_matrices,
    bench_non_power_of_2,
    bench_rectangular,
    bench_simd_f32,
    bench_simd_f32_large,
    bench_simd_f64,
    bench_simd_f64_large,
    bench_simd_rectangular,
    bench_all_algorithms_comparison,
    bench_parallel_f32,
    bench_parallel_f32_large,
    bench_parallel_f64,
    bench_parallel_generic,
    bench_all_methods_comparison
);
criterion_main!(benches);

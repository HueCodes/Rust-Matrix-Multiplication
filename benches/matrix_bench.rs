use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use matrix_multiply::Matrix;

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

criterion_group!(
    benches,
    bench_matrix_multiply_comparison,
    bench_large_matrices,
    bench_non_power_of_2,
    bench_rectangular
);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use matrix_multiply::Matrix;

fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply");
    
    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, &size| {
            let a = Matrix::from_vec(size, size, vec![1; size * size]).unwrap();
            let b = Matrix::from_vec(size, size, vec![2; size * size]).unwrap();
            b.iter(|| {
                black_box(a.multiply(&b).unwrap());
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_matrix_multiply);
criterion_main!(benches);

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(_c: &mut Criterion) {
    // CPU and GPU n-body benchmarks will go here (Days 12-13)
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{distributions::Uniform, Rng};

use itertools_num::linspace;
mod lib;

fn criterion_benchmark(c: &mut Criterion) {
    let n_samples = 1000;
    let freqs_iter = linspace::<f64>(0.0, 10.0, n_samples);
    let mut freqs: Vec<f64> = vec![];
    freqs.extend(freqs_iter);

    let b = 10.0;

    let range = Uniform::from(0.0..20.0);
    let amps: Vec<f64> = rand::thread_rng()
        .sample_iter(&range)
        .take(n_samples)
        .collect();

    c.bench_function("konno", |ben| {
        ben.iter(|| {
            lib::konnoohmachi_smooth(
                black_box(freqs.clone()),
                black_box(amps.clone()),
                black_box(b),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

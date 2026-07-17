use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use konnoohmachi::{konnoohmachi_smooth, Smoother};
use ndarray::Array1;

const BANDWIDTH: f64 = 40.0;

/// Frequency axis as it comes out of an rfft: linearly spaced, starting at zero.
fn frequencies(n_samples: usize) -> Array1<f64> {
    Array1::linspace(0.0, 100.0, n_samples)
}

/// Deterministic stand-in for a spectrum. The values do not affect runtime, only reproducibility.
fn amplitudes(n_samples: usize) -> Array1<f64> {
    Array1::from_iter((0..n_samples).map(|i| ((i * 37) % 13) as f64 + 1.0))
}

/// A single smoothing, where the cache cannot amortize and only has to not be slower.
fn bench_single_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_spectrum");

    for n_samples in [256, 1024, 4096] {
        let freqs = frequencies(n_samples);
        let amps = amplitudes(n_samples);

        group.bench_with_input(
            BenchmarkId::new("uncached", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    konnoohmachi_smooth(
                        black_box(freqs.view().into_dyn()),
                        black_box(amps.view().into_dyn()),
                        black_box(BANDWIDTH),
                    )
                })
            },
        );

        // Build + apply, i.e. what a caching user pays on a cache miss.
        group.bench_with_input(
            BenchmarkId::new("cached_cold", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let smoother = Smoother::new(black_box(freqs.view()), black_box(BANDWIDTH));
                    smoother.smooth(black_box(amps.view()))
                })
            },
        );
    }

    group.finish();
}

/// The case from issue #13: many spectra sharing one frequency axis.
fn bench_repeated_spectra(c: &mut Criterion) {
    let mut group = c.benchmark_group("repeated_spectra");
    let n_spectra = 10;

    for n_samples in [256, 1024, 4096] {
        let freqs = frequencies(n_samples);
        let spectra: Vec<Array1<f64>> = (0..n_spectra)
            .map(|offset| amplitudes(n_samples) + offset as f64)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("uncached", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    for amps in &spectra {
                        black_box(konnoohmachi_smooth(
                            black_box(freqs.view().into_dyn()),
                            black_box(amps.view().into_dyn()),
                            black_box(BANDWIDTH),
                        ));
                    }
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("cached", n_samples), &n_samples, |b, _| {
            b.iter(|| {
                let smoother = Smoother::new(black_box(freqs.view()), black_box(BANDWIDTH));
                for amps in &spectra {
                    black_box(smoother.smooth(black_box(amps.view())));
                }
            })
        });
    }

    group.finish();
}

/// Isolates the two halves of the cached path: building the windows vs. applying them.
fn bench_build_vs_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_vs_apply");

    for n_samples in [256, 1024, 4096] {
        let freqs = frequencies(n_samples);
        let amps = amplitudes(n_samples);
        let smoother = Smoother::new(freqs.view(), BANDWIDTH);

        group.bench_with_input(BenchmarkId::new("build", n_samples), &n_samples, |b, _| {
            b.iter(|| Smoother::new(black_box(freqs.view()), black_box(BANDWIDTH)))
        });

        group.bench_with_input(BenchmarkId::new("apply", n_samples), &n_samples, |b, _| {
            b.iter(|| smoother.smooth(black_box(amps.view())))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_spectrum,
    bench_repeated_spectra,
    bench_build_vs_apply
);
criterion_main!(benches);

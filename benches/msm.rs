//! This benchmarks Multi Scalar Multiplication (MSM).
//! Measurement on Bls12-381 G1.
//!
//! To run this benchmark:
//!
//!     cargo bench --bench msm
//!
//! To run the benchmark on halo2curve MSM version as well:
//!
//!     cargo bench --bench msm --features=h2c_compare

#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::PrimeField;
use group::Group;
use halo2curves::CurveAffine;
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::current_thread_index;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::time::SystemTime;
use rand_chacha::ChaCha20Rng;

const SAMPLE_SIZE: usize = 10;
const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

const MULTICORE_RANGE: &[u8] = &[8, 10, 12, 14, 16, 18, 20];

fn generate_curvepoints<C: CurveAffine>(k: u8) -> Vec<C> {
    let n: u64 = 1 << k;
    println!("Generating 2^{k} = {n} curve points..",);

    let timer = SystemTime::now();
    let bases = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| <C::CurveExt as Group>::random(rng).into(),
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} curve points took: {} sec.\n\n",
        end.as_secs()
    );
    bases
}

fn generate_coefficients<F: PrimeField>(k: u8) -> Vec<F> {
    let n = 1usize
        .checked_shl(k as u32)
        .expect("k çok büyük");
    // OS tabanlı seed ile ChaCha20Rng’i başlat
    let mut rng = ChaCha20Rng::from_entropy(); // requires rand_chacha + rand 0.8

    (0..n)
        .map(|_| F::random(&mut rng))
        .collect()
}

// Generates bases and coefficients for the given ranges and
// bit lenghts.
fn setup<C: CurveAffine>() -> (Vec<C>, Vec<C::ScalarExt>) {
    let max_k = *MULTICORE_RANGE.iter().max().unwrap_or(&16);
    assert!(max_k < 64);

    let bases = generate_curvepoints::<C>(max_k);
    let coeffs: Vec<_> = generate_coefficients(max_k);
      
    (bases, coeffs)
}

fn msm_blst(c: &mut Criterion) {
    let mut group = c.benchmark_group("Msm");
    group.significance_level(0.1).sample_size(SAMPLE_SIZE);

    let (bases, coeffs) = setup::<blstrs::G1Affine>();

    // Blstrs version.
    for k in MULTICORE_RANGE {
        let n: usize = 1 << k;
        let id = format!("blstrs_{k}");
        let points: Vec<blstrs::G1Projective> = bases.iter().map(Into::into).collect();
        group.bench_function(BenchmarkId::new("Blst", id), |b| {
            b.iter(|| blstrs::G1Projective::multi_exp(&points[..n], &coeffs[..n]))
        });
    }
    

    // Sppark version
    for k in MULTICORE_RANGE {
        let n: usize = 1 << k;
        let id = format!("gpu_{k}");
        group.bench_function(BenchmarkId::new("GPU", id), |b| {
            b.iter(|| {
                blstrs::msm_gpu(&bases[..n], &coeffs[..n])
            });
        });
    }


    #[cfg(feature = "h2c_compare")]
    // Halo2Curves version.
    for k in MULTICORE_RANGE {
        let n: usize = 1 << k;
        let id = format!("h2c_{k}");
        group.bench_function(BenchmarkId::new("halo2curves", id), |b| {
            b.iter(|| {
                halo2curves::msm::msm_best(&coeffs[..n], &bases[..n]);
            })
        });
    }
    

    group.finish();
}

criterion_group!(benches, msm_blst);
criterion_main!(benches);

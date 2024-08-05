use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::random;
use tracing::info_span;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// For testing the public values feature

pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.right, b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.right, next.left);

        // b' <- a + b
        when_transition.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: F, b: F, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace =
        RowMajorMatrix::new(vec![F::zero(); n * NUM_FIBONACCI_COLS], NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(a, b);

    for i in 1..n {
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

const NUM_FIBONACCI_COLS: usize = 2;

pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    const fn new(left: F, right: F) -> FibonacciRow<F> {
        FibonacciRow { left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

// RUST_LOG=debug RUSTFLAGS=-Ctarget-cpu=native cargo t --release bench_trace_commit -- --nocapture --exact
#[test]
fn bench_trace_commit() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Keccak256Hash {});

    type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let num_rows = 1 << 20;
    let num_cols = 32;
    let trace_vals: Vec<_> = (0..num_rows * num_cols)
        .into_par_iter()
        .map(|_| Val::from_wrapped_u32(random::<u32>()))
        .collect();

    let trace = RowMajorMatrix::new(trace_vals, num_cols);
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 103,
        proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = MyPcs::new(dft, val_mmcs, fri_config);

    assert_eq!(trace.height(), num_rows);
    let trace_domain = Pcs::<Challenge, Challenger>::natural_domain_for_degree(&pcs, num_rows);
    let (trace_commit, trace_data) = info_span!("commit to trace data")
        .in_scope(|| Pcs::<Challenge, Challenger>::commit(&pcs, vec![(trace_domain, trace)]));
}

// #[cfg(debug_assertions)]
// #[test]
// #[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
// fn test_incorrect_public_value() {
//     let perm = Perm::new_from_rng_128(
//         Poseidon2ExternalMatrixGeneral,
//         DiffusionMatrixBabyBear::default(),
//         &mut thread_rng(),
//     );
//     let hash = MyHash::new(perm.clone());
//     let compress = MyCompress::new(perm.clone());
//     let val_mmcs = ValMmcs::new(hash, compress);
//     let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
//     let dft = Dft {};
//     let fri_config = FriConfig {
//         log_blowup: 2,
//         num_queries: 28,
//         proof_of_work_bits: 8,
//         mmcs: challenge_mmcs,
//     };
//     let trace = generate_trace_rows::<Val>(Val::zero(), Val::one(), 1 << 3);
//     let pcs = Pcs::new(dft, val_mmcs, fri_config);
//     let config = MyConfig::new(pcs);
//     let mut challenger = Challenger::new(perm.clone());
//     let pis = vec![
//         BabyBear::from_canonical_u64(0),
//         BabyBear::from_canonical_u64(1),
//         BabyBear::from_canonical_u64(123_123), // incorrect result
//     ];
//     prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
// }

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const DATA_PATHS: &[&str] = &[
    "data/matrix_specs_main.json",
    "data/matrix_specs_4_perm.json",
    "data/matrix_specs_no_perm.json",
    "data/matrix_specs_gkr.json",
    "data/matrix_specs_gkr_adapter_separate.json",
    "data/matrix_specs_gkr_adapter_separate_group.json",
    "data/matrix_specs_main_quotient.json",
];

fn seeded_rng() -> impl Rng {
    SmallRng::seed_from_u64(0)
}

/// One round: a vector of (log_degree, width, num_openings) entries
type MatrixSpec = Vec<(usize, usize, usize)>;

fn load_matrix_specs_from_file(path: &str) -> Vec<MatrixSpec> {
    let file = File::open(path).expect("Could not open matrix spec JSON file");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("JSON was not well-formatted")
}

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn setup_pcs(rng: &mut impl Rng) -> (MyPcs, Challenger) {
    let perm = Perm::new_from_rng_128(rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_config = FriConfig {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_config);
    let challenger = Challenger::new(perm);

    (pcs, challenger)
}

fn make_pcs_input<P>(
    pcs: &P,
    challenger: &mut Challenger,
    matrix_specs_by_round: &[MatrixSpec],
    rng: &mut impl Rng,
) -> Vec<(P::ProverData, Vec<Vec<Challenge>>)>
where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    StandardUniform: Distribution<Val>,
{
    let (_, data_by_round): (Vec<_>, Vec<_>) =
        generate_random_traces(pcs, matrix_specs_by_round, rng)
            .iter()
            .map(|r| pcs.commit(r.clone()))
            .unzip();

    let zeta: Challenge = challenger.sample_algebra_element();

    let points_by_round = matrix_specs_by_round
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|(log_degree, _, num_openings)| {
                    let d = 1 << log_degree;
                    let domain = pcs.natural_domain_for_degree(d);

                    let mut opening_points = vec![zeta];
                    let mut cur_g = domain.next_point(Challenge::ONE).unwrap();
                    for _ in 1..*num_openings {
                        opening_points.push(cur_g * zeta);
                        cur_g *= cur_g
                    }
                    opening_points
                })
                .collect_vec()
        })
        .collect_vec();

    data_by_round
        .into_iter()
        .zip(points_by_round)
        .map(|(data, points)| (data, points))
        .collect_vec()
}

fn generate_random_traces<P>(
    pcs: &P,
    matrix_specs_by_round: &[MatrixSpec],
    mut rng: impl Rng,
) -> Vec<Vec<(P::Domain, RowMajorMatrix<Val>)>>
where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    StandardUniform: Distribution<Val>,
{
    let domains_and_polys_by_round = matrix_specs_by_round
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|&(log_degree, width, _)| {
                    let d = 1 << log_degree;
                    let domain = pcs.natural_domain_for_degree(d);
                    let matrix = RowMajorMatrix::rand(&mut rng, d, width);
                    (domain, matrix)
                })
                .collect_vec()
        })
        .collect_vec();
    domains_and_polys_by_round
}

fn bench_babybear_pcs_open(c: &mut Criterion) {
    let mut rng = seeded_rng();
    let (pcs, challenger) = setup_pcs(&mut rng);

    let matrix_specs_per_file: Vec<_> = DATA_PATHS
        .iter()
        .map(|path| load_matrix_specs_from_file(&path))
        .collect();

    let mut group = c.benchmark_group("pcs_open");
    group.sample_size(10);

    for (data_path, matrix_specs) in DATA_PATHS
        .into_iter()
        .zip(matrix_specs_per_file.into_iter())
    {
        let mut p_challenger = challenger.clone();
        let data_and_points = make_pcs_input(&pcs, &mut p_challenger, &matrix_specs, &mut rng);
        let data_and_points_view: Vec<_> = data_and_points
            .iter()
            .map(|(data, points)| (data, points.clone()))
            .collect();

        let label = Path::new(data_path).file_stem().unwrap().to_string_lossy();
        group.bench_function(label, |b| {
            b.iter(|| {
                let _ = pcs.open(data_and_points_view.clone(), &mut p_challenger);
            });
        });
    }

    group.finish();
}

fn bench_babybear_pcs_commit(c: &mut Criterion) {
    let mut rng = seeded_rng();
    let (pcs, _challenger) = setup_pcs(&mut rng);

    let matrix_specs_per_file: Vec<_> = DATA_PATHS
        .iter()
        .map(|path| load_matrix_specs_from_file(path))
        .collect();

    let mut group = c.benchmark_group("pcs_commit");
    group.sample_size(10);

    for (data_path, matrix_specs) in DATA_PATHS.iter().zip(matrix_specs_per_file.iter()) {
        let polys = generate_random_traces(&pcs, matrix_specs, &mut rng);
        let label = Path::new(data_path).file_stem().unwrap().to_string_lossy();

        group.bench_function(label, |b| {
            b.iter(|| {
                let _: Vec<_> = polys
                    .iter()
                    .map(|r| Pcs::<Challenge, Challenger>::commit(&pcs, r.clone()))
                    .collect();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_babybear_pcs_commit, bench_babybear_pcs_open);
criterion_main!(benches);

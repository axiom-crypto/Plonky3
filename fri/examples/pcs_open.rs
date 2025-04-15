use std::fs::File;
use std::io::BufReader;

use clap::Parser;
use itertools::Itertools;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Parser, Debug)]
#[command(
    name = "PCS Runner",
    about = "Runs PCS Open with matrix specs from JSON"
)]
struct Cli {
    /// Path to the matrix_specs JSON file
    #[arg(short, long, default_value = "matrix_specs.json")]
    input: String,
}

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

fn make_pcs_input<Val, Challenge, Challenger, P>(
    pcs: &P,
    challenger: &mut Challenger,
    matrix_specs_by_round: &[MatrixSpec],
) -> Vec<(P::ProverData, Vec<Vec<Challenge>>)>
where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field,
    Challenge: ExtensionField<Val>,
    StandardUniform: Distribution<Val>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
{
    let mut rng = seeded_rng();

    let domains_and_polys_by_round = matrix_specs_by_round
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|&(log_degree, width, _)| {
                    let d = 1 << log_degree;
                    let domain = pcs.natural_domain_for_degree(d);
                    let matrix = RowMajorMatrix::<Val>::rand(&mut rng, d, width);
                    (domain, matrix)
                })
                .collect_vec()
        })
        .collect_vec();

    let (_, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
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

fn main() {
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

    let perm = Perm::new_from_rng_128(&mut seeded_rng());
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
    let mut challenger = Challenger::new(perm);

    let cli = Cli::parse();
    let matrix_specs = load_matrix_specs_from_file(&cli.input);

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let data_and_points = make_pcs_input(&pcs, &mut challenger, &matrix_specs);
    let data_and_points_view: Vec<_> = data_and_points
        .iter()
        .map(|(data, points)| (data, points.clone()))
        .collect();

    pcs.open(data_and_points_view.clone(), &mut challenger);
}

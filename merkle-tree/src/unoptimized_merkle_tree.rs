use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// A binary Merkle tree for packed data. It has leaves of type `F` and digests of type
/// `[W; DIGEST_ELEMS]`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
#[derive(Debug)]
pub struct UnoptimizedMerkleTree<F, W, M, const DIGEST_ELEMS: usize> {
    pub(crate) leaves: Vec<M>,
    pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,
    _phantom: PhantomData<F>,
}

impl<F: Clone + Send + Sync, W: Clone, M: Matrix<F>, const DIGEST_ELEMS: usize>
    UnoptimizedMerkleTree<F, W, M, DIGEST_ELEMS>
{
    /// Assume matrix heights are powers of two.
    pub fn new<H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let first_digest_layer = (0..max_height)
            .map(|i| {
                let digest: [W; DIGEST_ELEMS] = h.hash_iter(
                    tallest_matrices
                        .iter()
                        .flat_map(|m| m.row_slice(i).to_vec()),
                );
                digest
            })
            .collect::<Vec<_>>();
        let mut digest_layers = vec![first_digest_layer];
        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() == 1 {
                break;
            }
            assert!(prev_layer.len().is_power_of_two());
            let next_layer_height = prev_layer.len() / 2;

            let prev_layer_compressed = prev_layer
                .chunks_exact(2)
                .map(|digest_pair| c.compress([digest_pair[0].clone(), digest_pair[1].clone()]))
                .collect::<Vec<_>>();

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height() == next_layer_height)
                .collect_vec();

            if matrices_to_inject.is_empty() {
                digest_layers.push(prev_layer_compressed);
                continue;
            }
            let layer_to_inject = (0..next_layer_height)
                .map(|i| {
                    let digest: [W; DIGEST_ELEMS] = h.hash_iter(
                        matrices_to_inject
                            .iter()
                            .flat_map(|m| m.row_slice(i).to_vec()),
                    );
                    digest
                })
                .collect::<Vec<_>>();
            let next_layer = prev_layer_compressed
                .into_iter()
                .zip_eq(layer_to_inject)
                .map(|(left, right)| c.compress([left, right]))
                .collect::<Vec<_>>();

            digest_layers.push(next_layer);
        }

        Self {
            leaves,
            digest_layers,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn root(&self) -> Hash<F, W, DIGEST_ELEMS>
    where
        W: Copy,
    {
        self.digest_layers.last().unwrap()[0].into()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    use p3_field::{Field, FieldAlgebra};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
    };
    use rand::distributions::Standard;
    use rand::rngs::StdRng;
    use rand::{thread_rng, Rng, SeedableRng};

    use crate::mmcs::MerkleTreeMmcs;
    use crate::unoptimized_merkle_tree::UnoptimizedMerkleTree;

    type F = BabyBear;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    // Actual plonky3 MMCS to test against:
    type MyMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;

    #[test]
    fn commit_single_1x8() {
        let mut rng = StdRng::seed_from_u64(0);
        let (rounds_f, rounds_p) = (8, 13);
        let external_constants = ExternalLayerConstants::new_from_rng(rounds_f, &mut rng);
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect();
        let perm = Poseidon2BabyBear::new(external_constants, internal_constants);

        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());

        // v = [2, 1, 2, 2, 0, 0, 1, 0]
        let v = vec![
            F::TWO,
            F::ONE,
            F::TWO,
            F::TWO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let (expected_commit, _) = mmcs.commit_vec(v.clone());
        let commit =
            UnoptimizedMerkleTree::new(&hash, &compress, vec![RowMajorMatrix::new_col(v)]).root();

        assert_eq!(commit, expected_commit);
    }

    #[test]
    fn commit_mixed() {
        let mut rng = StdRng::seed_from_u64(0);
        let (rounds_f, rounds_p) = (8, 13);
        let external_constants = ExternalLayerConstants::new_from_rng(rounds_f, &mut rng);
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect();
        let perm = Poseidon2BabyBear::new(external_constants, internal_constants);

        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
        let default_digest = [F::ZERO; 8];

        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        //   2 1
        //   2 2
        // ]
        let mat_1 = RowMajorMatrix::new(
            vec![
                F::ZERO,
                F::ONE,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
            ],
            2,
        );
        // mat_2 = [
        //   1 2 1
        //   0 2 2
        //   1 2 1
        // ]
        let mat_2 = RowMajorMatrix::new(
            vec![
                F::ONE,
                F::TWO,
                F::ONE,
                F::ZERO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::ONE,
            ],
            3,
        );

        let matrices = vec![mat_1, mat_2];
        let (expected_commit, _) = mmcs.commit(matrices.clone());
        let commit = UnoptimizedMerkleTree::new(&hash, &compress, matrices).root();
        assert_eq!(commit, expected_commit);
    }

    // .. see merkle-tree/src/mmcs.rs for more tests
}

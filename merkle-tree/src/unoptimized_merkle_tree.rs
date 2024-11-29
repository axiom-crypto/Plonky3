use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::PackedValue;
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
pub struct MerkleTree<F, W, M, const DIGEST_ELEMS: usize> {
    pub(crate) leaves: Vec<M>,
    pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,
    _phantom: PhantomData<F>,
}

impl<F: Clone, W: Clone, M: Matrix<F>, const DIGEST_ELEMS: usize>
    MerkleTree<F, W, M, DIGEST_ELEMS>
{
    /// Assume matrix heights are powers of two.
    pub fn new<H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        assert_eq!(P::WIDTH, PW::WIDTH, "Packing widths must match");

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
                        .flat_map(|m| m.vertically_packed_row(i)),
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
                .map(|digest_pair| c.compress([digest_pair[0], digest_pair[1]]))
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
                            .flat_map(|m| m.vertically_packed_row(i)),
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

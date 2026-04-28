"""Combined-kernel Boltzmann-sampling test.

Pins that `CustomCanonicalEnsemble` configured with both `PairSwap`
and `CyclicShift` produces the analytic Boltzmann distribution on
mchammer-pt's bundled fixture (4-site 1D chain, 2 Cu + 2 Au, NN-only
pair ECI, ΔE ≈ 3 kT at the test temperature).

The combined-kernel construction is the right test shape for
`CyclicShift` here because slide-row in isolation on a single
cyclic chain is exactly energy-preserving by translational
invariance: a unit shift moves the bond at site i to site i+1 and
the bond multiset is unchanged. So a single-chain fixture cannot
discriminate slide-row's correctness from a no-op kernel regardless
of chain length. The combined kernel reaches all six microstates
through `PairSwap` (which the framework's existing test already
validates as Boltzmann-sampling-correct on this fixture); slide-row's
contribution is its own correctness against the joint-kernel
stationary distribution. A two-chain analytic fixture would isolate
slide-row further but is deferred work.
"""

from __future__ import annotations

from mchammer_pt.testing import FIXTURE_CHAIN_INDICES, assert_boltzmann_sampling

from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap


def test_combined_pair_swap_and_cyclic_shift_samples_correct_boltzmann() -> None:
    """`CustomCanonicalEnsemble([PairSwap, CyclicShift])` matches analytic Boltzmann.

    Equal weights so both moves fire frequently; the bundled fixture's
    single 4-site chain is supplied to `CyclicShift` as the row argument.
    """
    chain = list(FIXTURE_CHAIN_INDICES[0])
    assert_boltzmann_sampling(
        CustomCanonicalEnsemble,
        ensemble_kwargs={
            "moves": [
                (PairSwap(sublattice_index=0), 1.0),
                (CyclicShift(cycles=[chain]), 1.0),
            ],
        },
    )

"""Boltzmann-sampling tests for `CustomCanonicalEnsemble`.

Pins that `CustomCanonicalEnsemble` produces the analytic Boltzmann
distribution on this package's local fixture (4-site 1D chain,
2 Cu + 2 Au, NN-only pair ECI, ΔE ≈ 3 kT at the test temperature)
under two configurations: `PairSwap` alone, and the combined
`PairSwap` + `CyclicShift` kernel. The fixture and the sampling
harness live in `tests/_boltzmann.py`.

There is no `CyclicShift`-alone test on this fixture: a unit cyclic
shift on a single chain moves the NN bond at site i to site i+1 and
leaves the bond multiset invariant, so the move is exactly
energy-preserving regardless of chain length. A single-chain fixture
cannot discriminate `CyclicShift`'s correctness from a no-op kernel.
The combined kernel reaches all six microstates through `PairSwap`
(pinned Boltzmann-correct by the `PairSwap`-alone test above) and
pins `CyclicShift`'s correctness against the joint-kernel stationary
distribution. A two-chain analytic fixture would isolate `CyclicShift`
further but is deferred work.
"""

from __future__ import annotations

from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap
from tests._boltzmann import FIXTURE_CHAIN_INDICES, assert_boltzmann_sampling


def test_pair_swap_alone_samples_correct_boltzmann() -> None:
    """`CustomCanonicalEnsemble([PairSwap])` matches analytic Boltzmann.

    Validates that running through `_do_trial_step` (with the dispatch
    + acceptance + counter machinery) reproduces the same stationary
    distribution as the framework's `CanonicalEnsemble` baseline. A
    sign flip on `potential_diff`, a miswired (sites, species), or any
    other bug in the trial-step path would surface here.
    """
    assert_boltzmann_sampling(
        CustomCanonicalEnsemble,
        ensemble_kwargs={
            "moves": [(PairSwap(sublattice_index=0), 1.0)],
        },
    )


def test_combined_pair_swap_and_cyclic_shift_samples_correct_boltzmann() -> None:
    """`CustomCanonicalEnsemble([PairSwap, CyclicShift])` matches analytic Boltzmann.

    Equal weights so both moves fire frequently; the bundled fixture's
    single 4-site chain is supplied to `CyclicShift`. Bugs in
    `CyclicShift.propose` (wrong cycle index, wrong direction
    semantics, wrong site ordering) would produce a stationary
    distribution mismatched against analytic Boltzmann here.
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

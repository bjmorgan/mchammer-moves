"""Boltzmann-sampling tests for `CustomCanonicalEnsemble`.

Pins that `CustomCanonicalEnsemble` produces the analytic Boltzmann
distribution on mchammer-pt's bundled fixture (4-site 1D chain,
2 Cu + 2 Au, NN-only pair ECI, ΔE ≈ 3 kT at the test temperature)
under two configurations: `PairSwap` alone, and the combined
`PairSwap` + `CyclicShift` kernel.

There is no `CyclicShift`-alone test on this fixture: a unit cyclic
shift on a single chain moves the NN bond at site i to site i+1 and
leaves the bond multiset invariant, so the move is exactly
energy-preserving regardless of chain length. A single-chain fixture
cannot discriminate `CyclicShift`'s correctness from a no-op kernel.
The combined kernel reaches all six microstates through `PairSwap`
(known Boltzmann-sampling-correct via mchammer-pt's framework test)
and pins `CyclicShift`'s correctness against the joint-kernel
stationary distribution. A two-chain analytic fixture would isolate
`CyclicShift` further but is deferred work.
"""

from __future__ import annotations

from mchammer_pt.testing import FIXTURE_CHAIN_INDICES, assert_boltzmann_sampling

from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap


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


def test_process_pool_propagates_per_move_acceptance(small_ising_setup) -> None:
    """End-to-end integration with `mchammer_pt.process_pool`.

    Pins the headline migration claim: `CustomCanonicalEnsemble`
    rides `mchammer_pt`'s native `ensemble_cls=` API through the
    spawn boundary, with per-move acceptance fields surviving in
    each replica's `BaseDataContainer`. The override of
    `_get_ensemble_data` exists specifically to make this work; this
    test exercises the pickle/spawn path that motivates it.

    A regression in `CustomCanonicalEnsemble.__init__` taking a
    non-picklable arg, or in mchammer-pt's spawn semantics, surfaces
    here rather than in production. Uses the local `small_ising_setup`
    fixture rather than mchammer-pt's analytic-Boltzmann fixture: the
    integration check only needs an MC-able CE+atoms, not a calibrated
    energy gap, so reaching into mchammer-pt's private fixture builder
    is unnecessary.
    """
    from mchammer_pt import CanonicalParallelTempering

    setup = small_ising_setup
    ce = setup["cluster_expansion"]
    atoms = setup["structure"]
    chain = list(range(len(atoms)))
    with CanonicalParallelTempering.process_pool(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[300.0, 600.0],
        block_size=20,
        random_seed=0,
        ensemble_cls=CustomCanonicalEnsemble,
        ensemble_kwargs={
            "moves": [
                (PairSwap(sublattice_index=0), 1.0),
                (CyclicShift(cycles=[chain]), 1.0),
            ],
            "ensemble_data_write_interval": 10,
        },
    ) as pt:
        pt.run(n_cycles=3)
        containers = pt.pool.data_containers()

    assert len(containers) == 2
    for dc in containers:
        cols = dc.data.columns
        assert "pair_swap_acceptance_rate" in cols
        assert "cyclic_shift_acceptance_rate" in cols
        # Final rows should have valid per-interval rates in [0, 1].
        for col in ("pair_swap_acceptance_rate", "cyclic_shift_acceptance_rate"):
            final = float(dc.data[col].iloc[-1])
            assert 0.0 <= final <= 1.0, (
                f"{col} per-interval rate {final} out of [0, 1] in worker output"
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

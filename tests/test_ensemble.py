"""Tests for :class:`CustomCanonicalEnsemble`.

Covers per-move accept/reject tracking, weight-based dispatch,
constructor validation, and compatibility with the base ensemble's
global counters.
"""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap


def test_per_move_counts_match_trial_step_returns(small_ising_setup):
    """Per-move accept counts must match what ``_do_trial_step`` returns.

    Calling ``_do_trial_step`` directly (without ``run()``, which
    periodically resets ``_accepted_trials`` at observation
    boundaries) lets us track the expected accept count locally and
    verify the per-move counters agree.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1500.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=11,
    )
    n = 3000
    expected_accepted = 0
    for _ in range(n):
        expected_accepted += ensemble._do_trial_step()

    rates = ensemble.acceptance_rates()
    total_proposed = sum(r.proposed for r in rates.values())
    total_accepted = sum(r.accepted for r in rates.values())
    assert total_proposed == n
    assert total_accepted == expected_accepted


def test_run_preserves_global_step_count(small_ising_setup):
    """Calling ``run(n)`` advances ``_step`` by ``n``.

    ``_accepted_trials`` is reset at each observer interval inside
    ``run`` (used to compute per-interval acceptance ratios in the data
    container), so it is not a cumulative counter — but ``_step`` is.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1500.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=11,
    )
    initial_step = ensemble._step
    n = 200
    ensemble.run(n)
    assert ensemble._step == initial_step + n
    rates = ensemble.acceptance_rates()
    assert sum(r.proposed for r in rates.values()) == n


def test_weight_based_dispatch(small_ising_setup):
    """Move selection probability tracks the configured weights.

    Configure two moves with a 4:1 weight ratio and verify the
    proposal counts are consistent with that ratio (chi-squared
    tolerance).
    """
    setup = small_ising_setup
    pair_swap_a = PairSwap(sublattice_index=0, name="swap_a")
    pair_swap_b = PairSwap(sublattice_index=0, name="swap_b")
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=2000.0,
        moves=[(pair_swap_a, 4.0), (pair_swap_b, 1.0)],
        random_seed=7,
    )
    n = 10_000
    for _ in range(n):
        ensemble._do_trial_step()
    rates = ensemble.acceptance_rates()
    assert rates["swap_a"].proposed + rates["swap_b"].proposed == n

    expected_a = n * 4 / 5
    se = np.sqrt(n * (4 / 5) * (1 / 5))
    z_a = abs(rates["swap_a"].proposed - expected_a) / se
    assert z_a < 4.0, (
        f"Weight dispatch off: swap_a proposed {rates['swap_a'].proposed}, "
        f"expected ~{expected_a}, z={z_a:.2f}"
    )


def test_reset_acceptance_counts(small_ising_setup):
    """``reset_acceptance_counts`` clears per-move counters but not globals."""
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=3,
    )
    ensemble.run(500)
    rates_before = ensemble.acceptance_rates()
    assert rates_before["pair_swap"].proposed == 500
    global_step_before = ensemble._step

    ensemble.reset_acceptance_counts()
    rates_after = ensemble.acceptance_rates()
    assert rates_after["pair_swap"].proposed == 0
    # The inherited cumulative step counter must NOT be reset
    assert ensemble._step == global_step_before


def test_constructor_validation(small_ising_setup):
    """Bad constructor inputs raise informative errors."""
    setup = small_ising_setup
    common = dict(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
    )

    with pytest.raises(ValueError, match="at least one"):
        CustomCanonicalEnsemble(moves=[], **common)

    with pytest.raises((TypeError, ValueError)):
        CustomCanonicalEnsemble(moves=[PairSwap(sublattice_index=0)], **common)

    with pytest.raises(ValueError, match="positive"):
        CustomCanonicalEnsemble(
            moves=[(PairSwap(sublattice_index=0), 0.0)], **common
        )

    with pytest.raises(ValueError, match="positive"):
        CustomCanonicalEnsemble(
            moves=[(PairSwap(sublattice_index=0), -1.0)], **common
        )

    with pytest.raises(ValueError, match="unique"):
        CustomCanonicalEnsemble(
            moves=[
                (PairSwap(sublattice_index=0, name="x"), 1.0),
                (PairSwap(sublattice_index=0, name="x"), 1.0),
            ],
            **common,
        )


def test_null_proposals_tracked_separately_from_metropolis_rejections(
    small_ising_setup,
):
    """A move that always returns ``None`` increments only the null
    counter, leaving accepted and (Metropolis-)rejected at zero.

    Achieved by forcing the configuration to a single-species
    occupation, on which `PairSwap.propose` always returns ``None``
    (no distinct-species pair to swap). The ensemble must increment
    `null_proposed`, not `rejected`, so that `MoveStats.null_rate`
    can diagnose the structurally-infeasible configuration without
    being conflated with energy rejections.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=0,
    )
    n_sites = len(ensemble.configuration.occupations)
    only_au = [79] * n_sites  # single species → no swap possible
    ensemble.update_occupations(list(range(n_sites)), only_au)

    n_trials = 500
    for _ in range(n_trials):
        ensemble._do_trial_step()

    stats = ensemble.acceptance_rates()["pair_swap"]
    assert stats.accepted == 0
    assert stats.rejected == 0
    assert stats.null_proposed == n_trials
    assert stats.proposed == n_trials
    assert stats.acceptance_rate == 0.0
    assert stats.null_rate == 1.0


def test_per_move_null_rate_in_data_container(small_ising_setup):
    """`<move>_null_rate` column appears alongside `<move>_acceptance_rate`
    and reports the per-interval null fraction.

    Forces a single-species configuration so that every PairSwap
    proposal returns ``None``; the data-container row for the
    interval must record `pair_swap_null_rate == 1.0` and
    `pair_swap_acceptance_rate == 0.0`.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=0,
        ensemble_data_write_interval=10,
    )
    n_sites = len(ensemble.configuration.occupations)
    ensemble.update_occupations(list(range(n_sites)), [79] * n_sites)
    ensemble.run(50)

    df = ensemble.data_container.data
    assert "pair_swap_acceptance_rate" in df.columns
    assert "pair_swap_null_rate" in df.columns
    last_acceptance = float(df["pair_swap_acceptance_rate"].iloc[-1])
    last_null = float(df["pair_swap_null_rate"].iloc[-1])
    assert last_acceptance == pytest.approx(0.0)
    assert last_null == pytest.approx(1.0)


def test_combined_pair_swap_and_cyclic_shift_runs(small_ising_setup):
    """Combined PairSwap + CyclicShift ensemble runs without error.

    Uses a synthetic single-row over the small system. The point is
    smoke-coverage of weight dispatch through both move types and
    correct behaviour of ``run`` and per-move tracking on a real CE.
    """
    setup = small_ising_setup
    n_sites = len(setup["structure"])
    rows = [list(range(n_sites))]  # one synthetic row containing every site
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=2000.0,
        moves=[
            (PairSwap(sublattice_index=0), 1.0),
            (CyclicShift(cycles=rows), 0.1),
        ],
        random_seed=42,
    )
    n = 1000
    for _ in range(n):
        ensemble._do_trial_step()
    rates = ensemble.acceptance_rates()
    assert set(rates.keys()) == {"pair_swap", "cyclic_shift"}
    assert rates["pair_swap"].proposed + rates["cyclic_shift"].proposed == n
    # Both moves should have non-zero proposal counts
    assert rates["pair_swap"].proposed > 0
    assert rates["cyclic_shift"].proposed > 0


def test_seeded_trial_step_sequence_is_reproducible(small_ising_factory):
    """Two ensembles seeded identically produce identical trajectories.

    Pins the load-bearing claim in `_do_trial_step`'s docstring:
    move-dispatch and `Move.propose` both draw from
    `_next_random_number`, so the entire trial-step sequence is
    determined by the seed. A future contributor adding an unrouted
    `numpy.random` or `random.choice` somewhere in the trial-step path
    would silently break per-replica RNG isolation under
    `mchammer_pt.ProcessPool`; this test catches it.

    mchammer drives MC from Python's global `random` module, so two
    co-resident ensembles share state. The test constructs and runs
    each ensemble sequentially — run-a-then-build-b-then-run-b — so
    that each ``CanonicalEnsemble.__init__`` ``random.seed(2024)``
    call lands the global state at the same starting point.
    """
    common_kwargs = dict(
        temperature=1500.0,
        moves=[
            (PairSwap(sublattice_index=0, name="swap_a"), 1.0),
            (PairSwap(sublattice_index=0, name="swap_b"), 1.0),
        ],
        random_seed=2024,
    )
    setup_a = small_ising_factory()
    ensemble_a = CustomCanonicalEnsemble(
        structure=setup_a["structure"],
        calculator=setup_a["calculator"],
        **common_kwargs,
    )
    ensemble_a.run(500)
    occ_a = ensemble_a.configuration.occupations.copy()
    rates_a = ensemble_a.acceptance_rates()

    setup_b = small_ising_factory()
    ensemble_b = CustomCanonicalEnsemble(
        structure=setup_b["structure"],
        calculator=setup_b["calculator"],
        **common_kwargs,
    )
    ensemble_b.run(500)
    np.testing.assert_array_equal(ensemble_b.configuration.occupations, occ_a)
    assert ensemble_b.acceptance_rates() == rates_a


def test_per_move_acceptance_in_data_container_is_per_interval(small_ising_setup):
    """Per-move acceptance fields are *per-interval*, not cumulative.

    The override of `_get_ensemble_data` is what makes per-move rates
    visible from a `mchammer_pt.ProcessPool` run, where the parent
    only ever sees the data container that's pickled back from the
    worker. The contract is: the data-container column reports the
    per-interval acceptance rate over the trials since the previous
    write — matching mchammer's `acceptance_ratio` convention so the
    per-move and global columns are directly comparable.

    Pinned by snapshotting the cumulative counts at two points in
    `acceptance_rates()` and verifying the data-container row for the
    interval between them equals the implied per-interval rate.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1500.0,
        moves=[
            (PairSwap(sublattice_index=0, name="swap_a"), 1.0),
            (PairSwap(sublattice_index=0, name="swap_b"), 1.0),
        ],
        random_seed=7,
        ensemble_data_write_interval=10,
    )
    ensemble.run(10)
    rates_after_first = ensemble.acceptance_rates()
    ensemble.run(10)
    rates_after_second = ensemble.acceptance_rates()

    df = ensemble.data_container.data
    assert "swap_a_acceptance_rate" in df.columns
    assert "swap_b_acceptance_rate" in df.columns

    for name in ("swap_a", "swap_b"):
        before = rates_after_first[name]
        after = rates_after_second[name]
        delta_accepted = after.accepted - before.accepted
        delta_proposed = after.proposed - before.proposed
        expected_interval_rate = (
            delta_accepted / delta_proposed if delta_proposed > 0 else 0.0
        )
        # The final data-container row records the second interval.
        column = f"{name}_acceptance_rate"
        observed = float(df[column].iloc[-1])
        assert observed == pytest.approx(expected_interval_rate, abs=1e-9), (
            f"Per-interval rate mismatch for {name}: "
            f"data-container={observed:.6f}, expected={expected_interval_rate:.6f}"
        )


def test_run_method_inherited(small_ising_setup):
    """``ensemble.run(n)`` should drive the custom trial step.

    The mchammer_pt wrapper calls ``ensemble.run(n_steps)``; ensure
    that path works end-to-end.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=21,
    )
    n = 500
    ensemble.run(n)
    rates = ensemble.acceptance_rates()
    assert rates["pair_swap"].proposed == n

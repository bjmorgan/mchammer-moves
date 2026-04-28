"""Tests for :class:`CustomCanonicalEnsemble`.

Covers per-move accept/reject tracking, weight-based dispatch,
constructor validation, and compatibility with the base ensemble's
global counters.
"""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_moves import CustomCanonicalEnsemble, PairSwap, SlideRow


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
    total_proposed = sum(r["proposed"] for r in rates.values())
    total_accepted = sum(r["accepted"] for r in rates.values())
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
    assert sum(r["proposed"] for r in rates.values()) == n


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
    assert rates["swap_a"]["proposed"] + rates["swap_b"]["proposed"] == n

    expected_a = n * 4 / 5
    se = np.sqrt(n * (4 / 5) * (1 / 5))
    z_a = abs(rates["swap_a"]["proposed"] - expected_a) / se
    assert z_a < 4.0, (
        f"Weight dispatch off: swap_a proposed {rates['swap_a']['proposed']}, "
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
    assert rates_before["pair_swap"]["proposed"] == 500
    global_step_before = ensemble._step

    ensemble.reset_acceptance_counts()
    rates_after = ensemble.acceptance_rates()
    assert rates_after["pair_swap"]["proposed"] == 0
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


def test_combined_pair_swap_and_slide_row_runs(small_ising_setup):
    """Combined PairSwap + SlideRow ensemble runs without error.

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
            (SlideRow(rows=rows), 0.1),
        ],
        random_seed=42,
    )
    n = 1000
    for _ in range(n):
        ensemble._do_trial_step()
    rates = ensemble.acceptance_rates()
    assert set(rates.keys()) == {"pair_swap", "slide_row"}
    assert rates["pair_swap"]["proposed"] + rates["slide_row"]["proposed"] == n
    # Both moves should have non-zero proposal counts
    assert rates["pair_swap"]["proposed"] > 0
    assert rates["slide_row"]["proposed"] > 0


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
    assert rates["pair_swap"]["proposed"] == n

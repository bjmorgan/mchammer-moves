"""Tests for :class:`CustomWangLandauEnsemble`.

Covers move dispatch, per-move counting, window/WL rejection
classification, data-container integration, constructor validation,
and reset.
"""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_moves import CustomWangLandauEnsemble, PairSwap


def test_wl_per_move_counts_match_trial_step_returns(small_ising_setup):
    """Per-move accept counts must match what _do_trial_step returns."""
    setup = small_ising_setup
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=11,
    )
    n = 500
    expected_accepted = 0
    for _ in range(n):
        expected_accepted += ensemble._do_trial_step()

    rates = ensemble.acceptance_rates()
    total_proposed = sum(r.proposed for r in rates.values())
    total_accepted = sum(r.accepted for r in rates.values())
    total_rejected = sum(r.rejected for r in rates.values())
    total_null = sum(r.null_proposed for r in rates.values())
    assert total_proposed == n
    assert total_accepted == expected_accepted
    assert total_accepted + total_rejected + total_null == n


def test_wl_null_proposals_tracked_separately(small_ising_setup):
    """A move that always returns None increments only the null counter."""
    setup = small_ising_setup
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=0,
    )
    n_sites = len(ensemble.configuration.occupations)
    ensemble.update_occupations(list(range(n_sites)), [79] * n_sites)

    n_trials = 200
    for _ in range(n_trials):
        ensemble._do_trial_step()

    stats = ensemble.acceptance_rates()["pair_swap"]
    assert stats.accepted == 0
    assert stats.rejected == 0
    assert stats.null_proposed == n_trials
    assert stats.null_rate == 1.0


def test_wl_weight_based_dispatch(small_ising_setup):
    """Move selection probability tracks the configured weights."""
    setup = small_ising_setup
    move_a = PairSwap(sublattice_index=0, name="swap_a")
    move_b = PairSwap(sublattice_index=0, name="swap_b")
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(move_a, 4.0), (move_b, 1.0)],
        random_seed=7,
    )
    n = 5000
    for _ in range(n):
        ensemble._do_trial_step()
    rates = ensemble.acceptance_rates()
    assert rates["swap_a"].proposed + rates["swap_b"].proposed == n

    expected_a = n * 4 / 5
    se = np.sqrt(n * (4 / 5) * (1 / 5))
    z = abs(rates["swap_a"].proposed - expected_a) / se
    assert z < 4.0, (
        f"Weight dispatch off: swap_a={rates['swap_a'].proposed}, "
        f"expected ~{expected_a:.0f}, z={z:.2f}"
    )

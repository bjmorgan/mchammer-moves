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


def test_wl_window_rejection_classification(small_ising_setup):
    """Rejections outside the energy window are classified as
    window-rejected, not WL-rejected.

    Uses tight energy bounds around the initial energy so that most
    swap proposals overshoot the window.
    """
    setup = small_ising_setup
    calculator = setup["calculator"]
    structure = setup["structure"]
    initial_energy = calculator.calculate_total(
        occupations=list(structure.get_atomic_numbers())
    )
    # Tight window: only a narrow band around the starting energy.
    ensemble = CustomWangLandauEnsemble(
        structure=structure,
        calculator=calculator,
        energy_spacing=0.01,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        energy_limit_left=initial_energy - 0.01,
        energy_limit_right=initial_energy + 0.01,
        random_seed=42,
    )
    n = 500
    for _ in range(n):
        ensemble._do_trial_step()

    breakdown = ensemble.rejection_breakdown()
    window_rej, wl_rej = breakdown["pair_swap"]
    stats = ensemble.acceptance_rates()["pair_swap"]
    # With a very tight window, most non-null proposals should be
    # window-rejected.
    assert window_rej > 0, "Expected some window rejections with tight bounds"
    # The initial configuration is inside the window, so
    # _reached_energy_window becomes True on the very first
    # _acceptance_condition call. All rejections are therefore
    # classified; no pre-window rejections can occur.
    assert window_rej + wl_rej == stats.rejected


def test_wl_data_container_columns(small_ising_setup):
    """Per-move acceptance, null, window-rejection, and WL-rejection
    columns appear in the data container.
    """
    setup = small_ising_setup
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=0,
        ensemble_data_write_interval=10,
    )
    ensemble.run(50)

    df = ensemble.data_container.data
    assert "pair_swap_acceptance_rate" in df.columns
    assert "pair_swap_null_rate" in df.columns
    assert "pair_swap_window_rejection_rate" in df.columns
    assert "pair_swap_wl_rejection_rate" in df.columns
    # All rates must be in [0, 1].
    for col in [
        "pair_swap_acceptance_rate",
        "pair_swap_null_rate",
        "pair_swap_window_rejection_rate",
        "pair_swap_wl_rejection_rate",
    ]:
        assert (df[col] >= 0.0).all(), f"{col} has negative values"
        assert (df[col] <= 1.0).all(), f"{col} has values > 1"


def test_wl_constructor_validation(small_ising_setup):
    """Bad constructor inputs raise informative errors."""
    setup = small_ising_setup
    common = dict(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
    )
    with pytest.raises(ValueError, match="at least one"):
        CustomWangLandauEnsemble(moves=[], **common)

    with pytest.raises(ValueError, match="positive"):
        CustomWangLandauEnsemble(
            moves=[(PairSwap(sublattice_index=0), -1.0)], **common
        )

    with pytest.raises(ValueError, match="unique"):
        CustomWangLandauEnsemble(
            moves=[
                (PairSwap(sublattice_index=0, name="x"), 1.0),
                (PairSwap(sublattice_index=0, name="x"), 1.0),
            ],
            **common,
        )


def test_wl_reset_clears_all_counters(small_ising_setup):
    """reset_acceptance_counts clears dispatcher and WL-specific counters."""
    setup = small_ising_setup
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=3,
    )
    for _ in range(200):
        ensemble._do_trial_step()

    ensemble.reset_acceptance_counts()

    stats = ensemble.acceptance_rates()["pair_swap"]
    assert stats.proposed == 0
    window_rej, wl_rej = ensemble.rejection_breakdown()["pair_swap"]
    assert window_rej == 0
    assert wl_rej == 0


def test_wl_run_method_works(small_ising_setup):
    """ensemble.run(n) drives the custom trial step."""
    setup = small_ising_setup
    ensemble = CustomWangLandauEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        energy_spacing=1.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=21,
    )
    n = 200
    ensemble.run(n)
    rates = ensemble.acceptance_rates()
    assert rates["pair_swap"].proposed == n


def test_wl_pre_window_rejections_not_classified(small_ising_setup):
    """Rejections during the pre-window search phase are not classified.

    Placing the energy window far above the starting configuration
    ensures the walker never reaches the window. The distance-penalty
    heuristic still rejects some proposals, but because
    ``_reached_energy_window`` stays ``False``, the window/WL rejection
    counters must remain at zero.
    """
    setup = small_ising_setup
    calculator = setup["calculator"]
    structure = setup["structure"]
    initial_energy = calculator.calculate_total(
        occupations=list(structure.get_atomic_numbers())
    )
    # Window placed far above the initial energy — unreachable in 300 steps.
    # energy_spacing=0.01 ensures proposals move between distinct bins so
    # that distance-penalty rejections actually occur.
    ensemble = CustomWangLandauEnsemble(
        structure=structure,
        calculator=calculator,
        energy_spacing=0.01,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        energy_limit_left=initial_energy + 500.0,
        energy_limit_right=initial_energy + 600.0,
        random_seed=99,
    )
    n = 300
    for _ in range(n):
        ensemble._do_trial_step()

    assert not ensemble._reached_energy_window, (
        "Walker should not have reached the window"
    )
    stats = ensemble.acceptance_rates()["pair_swap"]
    assert stats.rejected > 0, (
        "Expected some distance-penalty rejections during pre-window search"
    )
    window_rej, wl_rej = ensemble.rejection_breakdown()["pair_swap"]
    assert window_rej == 0, (
        "Pre-window rejections must not be classified as window-rejected"
    )
    assert wl_rej == 0, (
        "Pre-window rejections must not be classified as WL-rejected"
    )


def test_wl_post_reset_data_container_rates_non_negative(small_ising_setup):
    """Data-container rates are non-negative after a mid-run reset.

    Verifies that ``reset_acceptance_counts`` correctly clears the
    WL-specific snapshot baselines (``_last_recorded_window_reject`` and
    ``_last_recorded_wl_reject``) alongside the cumulative counters, so
    that per-interval deltas cannot go negative after the reset.
    """
    setup = small_ising_setup
    calculator = setup["calculator"]
    structure = setup["structure"]
    initial_energy = calculator.calculate_total(
        occupations=list(structure.get_atomic_numbers())
    )
    # Tight window so window/WL rejections accumulate quickly.
    ensemble = CustomWangLandauEnsemble(
        structure=structure,
        calculator=calculator,
        energy_spacing=0.01,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        energy_limit_left=initial_energy - 0.01,
        energy_limit_right=initial_energy + 0.01,
        random_seed=77,
    )
    # Phase 1: accumulate counts and advance the snapshot baselines.
    for _ in range(100):
        ensemble._do_trial_step()
    ensemble._get_ensemble_data()

    # Reset all counters (cumulative and snapshot baselines).
    ensemble.reset_acceptance_counts()

    # Phase 2: run more steps.
    for _ in range(100):
        ensemble._do_trial_step()

    # _get_ensemble_data must not produce negative rates — a negative
    # rate would indicate that a snapshot baseline was not cleared
    # alongside its cumulative counter during reset.
    data = ensemble._get_ensemble_data()
    assert data["pair_swap_window_rejection_rate"] >= 0.0
    assert data["pair_swap_wl_rejection_rate"] >= 0.0

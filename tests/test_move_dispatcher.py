"""Tests for :class:`MoveDispatcher`.

Covers constructor validation, weighted selection, per-move counter
bookkeeping, per-interval rate computation, and reset.
"""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_moves import MoveDispatcher, PairSwap
from tests.conftest import seeded_uniform


def test_dispatcher_rejects_empty_moves_list():
    with pytest.raises(ValueError, match="at least one"):
        MoveDispatcher(moves=[])


def test_dispatcher_rejects_non_tuple_entry():
    with pytest.raises((TypeError, ValueError)):
        MoveDispatcher(moves=[PairSwap(sublattice_index=0)])


def test_dispatcher_rejects_zero_weight():
    with pytest.raises(ValueError, match="positive"):
        MoveDispatcher(moves=[(PairSwap(sublattice_index=0), 0.0)])


def test_dispatcher_rejects_negative_weight():
    with pytest.raises(ValueError, match="positive"):
        MoveDispatcher(moves=[(PairSwap(sublattice_index=0), -1.0)])


def test_dispatcher_rejects_duplicate_move_names():
    with pytest.raises(ValueError, match="unique"):
        MoveDispatcher(
            moves=[
                (PairSwap(sublattice_index=0, name="x"), 1.0),
                (PairSwap(sublattice_index=0, name="x"), 1.0),
            ]
        )


def test_choose_respects_weights():
    """Weighted selection probability tracks configured weights."""
    move_a = PairSwap(sublattice_index=0, name="a")
    move_b = PairSwap(sublattice_index=0, name="b")
    dispatcher = MoveDispatcher(
        moves=[(move_a, 4.0), (move_b, 1.0)]
    )
    rng = seeded_uniform(42)
    counts: dict[str, int] = {"a": 0, "b": 0}
    n = 10_000
    for _ in range(n):
        chosen = dispatcher.choose(rng)
        counts[chosen.name] += 1

    expected_a = n * 4 / 5
    se = np.sqrt(n * (4 / 5) * (1 / 5))
    z = abs(counts["a"] - expected_a) / se
    assert z < 4.0, (
        f"Weight dispatch off: a={counts['a']}, "
        f"expected ~{expected_a:.0f}, z={z:.2f}"
    )


def test_record_and_acceptance_rates():
    """record_* increments are reflected in acceptance_rates()."""
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])

    dispatcher.record_accept("ps")
    dispatcher.record_accept("ps")
    dispatcher.record_reject("ps")
    dispatcher.record_null("ps")

    stats = dispatcher.acceptance_rates()["ps"]
    assert stats.accepted == 2
    assert stats.rejected == 1
    assert stats.null_proposed == 1
    assert stats.proposed == 4
    assert stats.acceptance_rate == pytest.approx(0.5)
    assert stats.null_rate == pytest.approx(0.25)


def test_get_interval_data_per_interval_semantics():
    """get_interval_data returns per-interval rates and updates snapshots."""
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])

    # First interval: 3 accepts, 1 reject
    for _ in range(3):
        dispatcher.record_accept("ps")
    dispatcher.record_reject("ps")

    data1 = dispatcher.get_interval_data()
    assert data1["ps_acceptance_rate"] == pytest.approx(3 / 4)
    assert data1["ps_null_rate"] == pytest.approx(0.0)

    # Second interval: 1 accept, 1 null
    dispatcher.record_accept("ps")
    dispatcher.record_null("ps")

    data2 = dispatcher.get_interval_data()
    assert data2["ps_acceptance_rate"] == pytest.approx(1 / 2)
    assert data2["ps_null_rate"] == pytest.approx(1 / 2)


def test_get_interval_data_raises_on_negative_interval():
    """Negative interval delta triggers RuntimeError.

    Simulated by manually tampering with a snapshot counter to be
    higher than the cumulative counter.
    """
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])
    dispatcher.record_accept("ps")
    # Corrupt: set the snapshot ahead of the cumulative counter.
    dispatcher._last_recorded_accept["ps"] = 999
    with pytest.raises(RuntimeError, match="negative"):
        dispatcher.get_interval_data()


def test_reset_clears_all_counters():
    """reset() zeroes lifetime and snapshot counters."""
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])

    dispatcher.record_accept("ps")
    dispatcher.record_reject("ps")
    dispatcher.record_null("ps")
    dispatcher.get_interval_data()  # populate snapshots

    dispatcher.reset()

    stats = dispatcher.acceptance_rates()["ps"]
    assert stats.proposed == 0

    # After reset, get_interval_data should return zeros
    # (no trials in this interval).
    data = dispatcher.get_interval_data()
    assert data["ps_acceptance_rate"] == 0.0
    assert data["ps_null_rate"] == 0.0


def test_moves_property_returns_copy():
    """Mutating the returned list does not affect the dispatcher."""
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])
    moves_copy = dispatcher.moves
    moves_copy.clear()
    assert len(dispatcher.moves) == 1


def test_move_weights_property_returns_copy():
    """Mutating the returned list does not affect the dispatcher."""
    move = PairSwap(sublattice_index=0, name="ps")
    dispatcher = MoveDispatcher(moves=[(move, 1.0)])
    weights_copy = dispatcher.move_weights
    weights_copy.clear()
    assert len(dispatcher.move_weights) == 1

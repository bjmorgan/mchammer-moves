"""Tests for :class:`MoveDispatcher`.

Covers constructor validation.
"""

from __future__ import annotations

import pytest

from mchammer_moves import MoveDispatcher, PairSwap


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

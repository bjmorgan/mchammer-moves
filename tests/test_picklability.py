"""Picklability of move objects across a process spawn.

`mchammer-pt`'s `ProcessPool` pickles the ensemble and its registered
moves to spawn workers (see the multiprocess note in
`mchammer_moves.moves.base`). A move that silently became unpicklable
would break a multiprocess parallel-tempering run rather than fail in
an obvious place. These tests pin that the built-in moves used across
that spawn boundary round-trip through `pickle` with their
configuration intact.
"""

from __future__ import annotations

import pickle

from mchammer_moves import CyclicShift, PairSwap


def test_pair_swap_pickle_round_trip() -> None:
    """`PairSwap` survives a `pickle` round-trip with its state intact."""
    move = PairSwap(sublattice_index=0)
    restored = pickle.loads(pickle.dumps(move))
    assert isinstance(restored, PairSwap)
    assert restored.name == move.name
    assert restored.sublattice_index == move.sublattice_index
    assert restored.allowed_species == move.allowed_species
    assert restored.allowed_sites == move.allowed_sites


def test_cyclic_shift_pickle_round_trip() -> None:
    """`CyclicShift` survives a `pickle` round-trip with its state intact."""
    move = CyclicShift(cycles=[[0, 1, 2, 3]])
    restored = pickle.loads(pickle.dumps(move))
    assert isinstance(restored, CyclicShift)
    assert restored.name == move.name
    assert restored.cycles == move.cycles

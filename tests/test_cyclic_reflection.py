"""Tests for :class:`CyclicReflection`.

Covers:

1. Structural correctness — known input pattern reflects to the
   formula-predicted result for chosen pivots, including periodic
   wrap-around.
2. Identity-skip — uniform-species cycle and palindromic cycle
   reflected around its symmetry axis return ``None``.
3. Detailed balance — ``P(A → B) = P(B → A)`` over a small enumerable
   state space.
4. Constructor validation.
"""

from __future__ import annotations

from itertools import permutations
from unittest.mock import MagicMock

import numpy as np
import pytest

from mchammer_moves import CyclicReflection
from tests.conftest import seeded_uniform


def _make_fake_configuration(occupations: list[int]):
    """Return a minimal mock supplying the ``occupations`` attribute."""
    config = MagicMock()
    config.occupations = np.array(occupations, dtype=int)
    return config


def _fixed_rng(values: list[float]):
    """Callable returning a fixed sequence of uniform draws.

    Used to force `CyclicReflection.propose` down a specific
    (cycle, pivot) branch in deterministic structural tests.
    """
    seq = iter(values)

    def draw() -> float:
        return next(seq)

    return draw


def test_cyclic_reflection_pivot_0_on_length_4_swaps_off_axis_pair():
    """For cycle ``[10, 11, 12, 13]`` with species ``[A, B, C, D]``,
    a reflection at pivot ``0`` fixes positions 0 and 2 (the pivot
    and its antipode under cyclic L=4) and swaps positions 1 and 3.
    """
    cycle = [10, 11, 12, 13]
    occupations = [0] * 20
    occupations[10] = 100  # A
    occupations[11] = 101  # B
    occupations[12] = 102  # C
    occupations[13] = 103  # D
    config = _make_fake_configuration(occupations)

    move = CyclicReflection(cycles=[cycle])
    # First draw selects cycle 0 (0.0 → cycle 0); second draw, scaled
    # to L=4, gives pivot = int(0.0 * 4) = 0.
    sites, species = move.propose(config, _fixed_rng([0.0, 0.0]))

    assert list(sites) == cycle
    assert species == [100, 103, 102, 101]


def test_cyclic_reflection_pivot_in_middle_fixes_centre_for_odd_length():
    """For cycle ``[0, 1, 2, 3, 4]`` with species ``[A, B, C, D, E]``,
    a reflection at pivot ``2`` fixes position 2 (the pivot) and
    swaps the pairs (1, 3) and (0, 4).
    """
    cycle = [0, 1, 2, 3, 4]
    occupations = [100, 101, 102, 103, 104]
    config = _make_fake_configuration(occupations)
    move = CyclicReflection(cycles=[cycle])
    # Cycle 0; pivot = int(0.5 * 5) = 2.
    sites, species = move.propose(config, _fixed_rng([0.0, 0.5]))
    assert list(sites) == cycle
    assert species == [104, 103, 102, 101, 100]


def test_cyclic_reflection_dispatches_across_cycles_and_pivots():
    """Both cycles must be reachable, and a reflection on one cycle
    must not mutate sites in the other cycle.
    """
    cycle_a = [0, 1, 2]
    cycle_b = [10, 11, 12, 13, 14]
    occupations = [0] * 20
    for i, idx in enumerate(cycle_a):
        occupations[idx] = 100 + i  # 100, 101, 102
    for i, idx in enumerate(cycle_b):
        occupations[idx] = 200 + i  # 200..204
    config = _make_fake_configuration(occupations)
    move = CyclicReflection(cycles=[cycle_a, cycle_b])

    seen_a = 0
    seen_b = 0
    pivots_a: set[tuple[int, int, int]] = set()
    pivots_b: set[tuple[int, ...]] = set()
    for seed in range(400):
        result = move.propose(config, seeded_uniform(seed))
        if result is None:
            continue
        sites, species = result
        sites_set = set(sites)
        if sites_set == set(cycle_a):
            seen_a += 1
            pivots_a.add(tuple(species))
            assert set(species) == {100, 101, 102}
        elif sites_set == set(cycle_b):
            seen_b += 1
            pivots_b.add(tuple(species))
            assert set(species) == {200, 201, 202, 203, 204}
        else:
            raise AssertionError(f"Proposal sites {sites} match neither cycle")
    assert seen_a > 0 and seen_b > 0, (
        f"Both cycles must be reachable; got seen_a={seen_a}, seen_b={seen_b}"
    )
    # On a non-uniform cycle, multiple pivots produce distinct
    # reflected patterns; the set should have more than one element.
    assert len(pivots_a) > 1, f"Only one cycle_a pattern observed: {pivots_a}"
    assert len(pivots_b) > 1, f"Only one cycle_b pattern observed: {pivots_b}"


def test_cyclic_reflection_returns_none_on_uniform_species_cycle():
    """A cycle whose every site shares one species reflects to itself
    under any pivot; the move returns ``None`` so the per-move
    acceptance rate is not inflated by trivial accepts.
    """
    cycle = [0, 1, 2, 3]
    occupations = [42, 42, 42, 42]
    config = _make_fake_configuration(occupations)
    move = CyclicReflection(cycles=[cycle])
    for seed in range(20):
        assert move.propose(config, seeded_uniform(seed)) is None


def test_cyclic_reflection_returns_none_on_palindrome_at_symmetry_pivot():
    """A palindromic cycle reflected around its symmetry axis returns
    ``None``. For ``[A, B, C, B, A]`` the symmetry axis sits on
    position 2; a pivot=2 reflection is the identity.
    """
    cycle = [0, 1, 2, 3, 4]
    occupations = [100, 101, 102, 101, 100]
    config = _make_fake_configuration(occupations)
    move = CyclicReflection(cycles=[cycle])
    # Cycle 0; pivot = int(0.5 * 5) = 2.
    assert move.propose(config, _fixed_rng([0.0, 0.5])) is None


def test_cyclic_reflection_returns_proposal_for_non_palindromic_pivot():
    """Same palindromic cycle, but a non-symmetry pivot still yields a
    valid (non-identity) proposal.
    """
    cycle = [0, 1, 2, 3, 4]
    occupations = [100, 101, 102, 101, 100]  # palindrome at pivot 2
    config = _make_fake_configuration(occupations)
    move = CyclicReflection(cycles=[cycle])
    # Cycle 0; pivot = int(0.0 * 5) = 0. Reflection at 0 swaps (1,4)
    # and (2,3); the pattern is asymmetric at pivot 0.
    sites, species = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert sites == cycle
    # Position 0: from (0-0)%5=0 → 100; pos 1: from 4 → 100;
    # pos 2: from 3 → 101; pos 3: from 2 → 102; pos 4: from 1 → 101.
    assert species == [100, 100, 101, 102, 101]


def test_cyclic_reflection_rejects_invalid_cycles():
    with pytest.raises(ValueError, match="at least one cycle"):
        CyclicReflection(cycles=[])
    with pytest.raises(ValueError, match="empty"):
        CyclicReflection(cycles=[[]])
    with pytest.raises(ValueError, match="length 1"):
        CyclicReflection(cycles=[[0]])
    with pytest.raises(ValueError, match="length 2"):
        CyclicReflection(cycles=[[0, 1]])


def test_cyclic_reflection_rejects_within_cycle_duplicate_sites():
    with pytest.raises(ValueError, match="duplicate site indices"):
        CyclicReflection(cycles=[[3, 5, 3]])


def test_cyclic_reflection_rejects_overlapping_cycles():
    with pytest.raises(ValueError, match="shares site"):
        CyclicReflection(cycles=[[0, 1, 2], [2, 3, 4]])


def test_cyclic_reflection_accepts_range_object_as_cycle():
    """`range(N)` is a valid `Sequence[int]`; the constructor accepts it."""
    move = CyclicReflection(cycles=[range(4)])
    assert move.cycles == [(0, 1, 2, 3)]


def test_cyclic_reflection_detailed_balance_on_chain():
    """Detailed balance for `CyclicReflection` on a single 6-site chain.

    At composition (3, 3) the canonical state space has
    ``C(6, 3) = 20`` distinct states. Reflection transitions form
    pairs of involutions; the empirical transition matrix must be
    symmetric within ~4 sigma.
    """
    cycle = list(range(6))
    move = CyclicReflection(cycles=[cycle])
    sp_a, sp_b = 100, 101

    states = sorted(set(permutations([0, 0, 0, 1, 1, 1])))
    state_index = {s: i for i, s in enumerate(states)}

    def to_atomic(local: tuple[int, ...]) -> list[int]:
        return [sp_b if b == 1 else sp_a for b in local]

    def from_atomic(occ) -> tuple[int, ...]:
        return tuple(0 if int(z) == sp_a else 1 for z in occ)

    n_per = 6000
    transitions = np.zeros((len(states), len(states)), dtype=int)
    rng = seeded_uniform(24680)
    for src in states:
        config = _make_fake_configuration(to_atomic(src))
        for _ in range(n_per):
            result = move.propose(config, rng)
            if result is None:
                transitions[state_index[src], state_index[src]] += 1
                continue
            sites, species = result
            cand = list(config.occupations)
            for s, z in zip(sites, species, strict=True):
                cand[s] = z
            transitions[state_index[src], state_index[from_atomic(cand)]] += 1

    failures = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            a, b = transitions[i, j], transitions[j, i]
            if a + b < 30:
                continue
            mean = (a + b) / 2
            std = np.sqrt((a + b) / 4)
            z = abs(a - mean) / std
            if z > 4.0:
                failures.append((i, j, a, b, z))
    assert not failures, (
        "CyclicReflection detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )

    # Sanity: must have observed at least some transitions.
    assert transitions.sum() == n_per * len(states)

"""Tests for :class:`CyclicShift`.

Covers:

1. Structural correctness — a known input pattern produces the expected
   output for ``+1`` and ``-1`` shifts, including periodic wrap-around.
2. Detailed balance — ``P(A -> B) = P(B -> A)`` over a small enumerable
   state space.
3. Constructor validation.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from unittest.mock import MagicMock

import numpy as np
import pytest

from mchammer_moves import CyclicShift
from tests.conftest import seeded_uniform


def _make_fake_configuration(occupations: list[int]):
    """Return a minimal mock supplying the ``occupations`` attribute."""
    config = MagicMock()
    config.occupations = np.array(occupations, dtype=int)
    return config


def _fixed_rng(values: list[float]):
    """Callable returning a fixed sequence of uniform draws.

    Used to force `CyclicShift.propose` down a specific (cycle,
    direction) branch in deterministic structural tests. The callable
    cycles through the supplied values so a single-proposal test only
    needs to specify two draws (cycle index, direction).
    """
    seq = iter(values)

    def draw() -> float:
        return next(seq)

    return draw


def test_cyclic_shift_plus_one_shifts_pattern_forward():
    """Direction ``+1`` shifts each species one site forward along the row.

    For row ``[10, 11, 12, 13]`` with species ``[A, B, C, D]``, a ``+1``
    slide must produce ``[D, A, B, C]`` at sites ``[10, 11, 12, 13]``
    (the last species wraps to the front).
    """
    row = [10, 11, 12, 13]
    occupations = [0] * 20
    occupations[10] = 100  # A
    occupations[11] = 101  # B
    occupations[12] = 102  # C
    occupations[13] = 103  # D
    config = _make_fake_configuration(occupations)

    move = CyclicShift(cycles=[row])
    # First draw picks the row (0.0 → row 0); second draw picks the
    # direction (< 0.5 → +1).
    sites, species = move.propose(config, _fixed_rng([0.0, 0.0]))

    assert list(sites) == row
    assert species == [103, 100, 101, 102]


def test_cyclic_shift_minus_one_shifts_pattern_backward():
    """Direction ``-1`` shifts each species one site backward along the row."""
    row = [0, 1, 2, 3]
    occupations = [10, 11, 12, 13]
    config = _make_fake_configuration(occupations)

    move = CyclicShift(cycles=[row])
    # First draw picks the row; second draw picks the direction
    # (>= 0.5 → -1).
    sites, species = move.propose(config, _fixed_rng([0.0, 0.9]))

    assert list(sites) == row
    assert species == [11, 12, 13, 10]


def test_cyclic_shift_period_3_pattern_one_step_breaks_period():
    """A length-6 period-3 pattern produces both forward and backward
    shifts over a range of RNG seeds.

    For the period-3 pattern ``[A, B, B, A, B, B]`` on a length-6 row,
    a single forward slide gives ``[B, A, B, B, A, B]`` and a single
    backward slide gives ``[B, B, A, B, B, A]``. Both must appear when
    sampling many seeds.
    """
    row = [0, 1, 2, 3, 4, 5]
    pattern = [100, 101, 101, 100, 101, 101]
    config = _make_fake_configuration(pattern)
    move = CyclicShift(cycles=[row])

    seen = Counter()
    for seed in range(200):
        _, species = move.propose(config, seeded_uniform(seed))
        seen[tuple(species)] += 1

    # Forward slide:  pattern[(i - 1) % 6] -> [101, 100, 101, 101, 100, 101]
    # Backward slide: pattern[(i + 1) % 6] -> [101, 101, 100, 101, 101, 100]
    expected = {
        (101, 100, 101, 101, 100, 101),
        (101, 101, 100, 101, 101, 100),
    }
    assert set(seen.keys()) == expected


def test_cyclic_shift_rejects_empty_cycles():
    with pytest.raises(ValueError, match="at least one cycle"):
        CyclicShift(cycles=[])
    with pytest.raises(ValueError, match="empty"):
        CyclicShift(cycles=[[]])
    with pytest.raises(ValueError, match="length 1"):
        CyclicShift(cycles=[[0]])


def test_cyclic_shift_returns_none_on_identity_proposal():
    """A cycle whose every site shares one species produces an identity
    shift; the move returns ``None`` so the per-move acceptance rate
    in the data container is not inflated by trivial accepts.
    """
    cycle = [0, 1, 2, 3]
    occupations = [42, 42, 42, 42]
    config = _make_fake_configuration(occupations)
    move = CyclicShift(cycles=[cycle])

    # Any RNG draws produce an identity shift on a uniform cycle.
    assert move.propose(config, _fixed_rng([0.0, 0.0])) is None
    assert move.propose(config, _fixed_rng([0.0, 0.9])) is None


def test_cyclic_shift_returns_proposal_when_cycle_is_not_uniform():
    """A non-uniform cycle produces a non-identity proposal."""
    cycle = [0, 1, 2, 3]
    occupations = [10, 20, 10, 20]  # alternating, so any shift differs
    config = _make_fake_configuration(occupations)
    move = CyclicShift(cycles=[cycle])
    proposal = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert proposal is not None


def test_cyclic_shift_accepts_range_object_as_cycle():
    """`range(N)` is a valid `Sequence[int]`; the constructor accepts it."""
    move = CyclicShift(cycles=[range(4)])
    assert move.cycles == [(0, 1, 2, 3)]


def test_cyclic_shift_rejects_within_cycle_duplicate_sites():
    """A cycle with a repeated index would propose multiple species
    values for the duplicated site — caught up-front rather than
    producing undefined behaviour at `update_occupations`.
    """
    with pytest.raises(ValueError, match="duplicate site indices"):
        CyclicShift(cycles=[[3, 5, 3]])


def test_cyclic_shift_rejects_overlapping_cycles():
    """Cycles that share sites silently over-sample the shared sites.

    The likely real-world cause is a chain-construction bug (e.g. an
    off-by-one in a meshgrid) producing two cycles that contain the
    same global index. Caught at the constructor.
    """
    with pytest.raises(ValueError, match="shares site"):
        CyclicShift(cycles=[[0, 1, 2], [2, 3, 4]])


def test_cyclic_shift_dispatches_across_multiple_cycles():
    """Both cycles must be reachable, and a slide on one cycle must
    not mutate sites in the other cycle.

    Two cycles of different lengths are configured; over many
    proposals (a) each cycle is selected at least once with high
    probability, (b) every proposal touches only sites of one cycle,
    and (c) the species values produced are consistent with the
    chosen cycle's length-specific wrap.
    """
    cycle_a = [0, 1, 2]      # length 3
    cycle_b = [10, 11, 12, 13, 14]  # length 5
    occupations = [0] * 20
    for i, idx in enumerate(cycle_a):
        occupations[idx] = 100 + i
    for i, idx in enumerate(cycle_b):
        occupations[idx] = 200 + i
    config = _make_fake_configuration(occupations)
    move = CyclicShift(cycles=[cycle_a, cycle_b])

    seen_a = 0
    seen_b = 0
    for seed in range(400):
        sites, species = move.propose(config, seeded_uniform(seed))
        sites_set = set(sites)
        if sites_set == set(cycle_a):
            seen_a += 1
            assert len(species) == len(cycle_a)
            # Species values must be a cyclic permutation of cycle_a's species.
            assert set(species) == {100, 101, 102}
        elif sites_set == set(cycle_b):
            seen_b += 1
            assert len(species) == len(cycle_b)
            assert set(species) == {200, 201, 202, 203, 204}
        else:
            raise AssertionError(
                f"Proposal sites {sites} match neither cycle"
            )
    assert seen_a > 0 and seen_b > 0, (
        f"Both cycles must be reachable; got seen_a={seen_a}, seen_b={seen_b}"
    )


def test_cyclic_shift_detailed_balance_on_chain():
    """Detailed balance for CyclicShift on a single 6-site chain.

    With a single row of length 6 and binary species, the full canonical
    state space at composition (3, 3) has C(6, 3) = 20 states. Slide
    transitions form orbits under cyclic shifts; within each orbit, all
    transitions are between distinct states. We check that the empirical
    transition matrix is symmetric.
    """
    row = list(range(6))
    move = CyclicShift(cycles=[row])
    sp_a, sp_b = 100, 101

    # Enumerate distinct binary occupations of length 6 with three 1s.
    states = sorted(set(permutations([0, 0, 0, 1, 1, 1])))
    state_index = {s: i for i, s in enumerate(states)}

    def to_atomic(local: tuple[int, ...]) -> list[int]:
        return [sp_b if b == 1 else sp_a for b in local]

    def from_atomic(occ) -> tuple[int, ...]:
        return tuple(0 if int(z) == sp_a else 1 for z in occ)

    n_per = 6000
    transitions = np.zeros((len(states), len(states)), dtype=int)
    rng = seeded_uniform(13579)
    for src in states:
        config = _make_fake_configuration(to_atomic(src))
        for _ in range(n_per):
            sites, species = move.propose(config, rng)
            cand = list(config.occupations)
            for s, z in zip(sites, species, strict=True):
                cand[s] = z
            dst = from_atomic(cand)
            transitions[state_index[src], state_index[dst]] += 1

    # Symmetry check: for each pair (i, j) with sufficient counts, the
    # forward and reverse counts must agree within statistical tolerance.
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
        "CyclicShift detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )

    # Sanity: must have observed at least some transitions.
    assert transitions.sum() == n_per * len(states)
    # Diagonal: a slide of a non-uniform pattern reaches itself only via
    # the identity orbit (period | 1 = 1). For all our states the chain
    # is not uniform, so the diagonal should be small. Just confirm
    # proposals actually move the system most of the time.
    diag_frac = np.trace(transitions) / transitions.sum()
    assert diag_frac < 0.5, (
        f"Most slides should change state, got diag_frac={diag_frac}"
    )

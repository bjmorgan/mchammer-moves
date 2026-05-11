"""Tests for :class:`IndexSetSwap`.

Covers:

1. Basic structural correctness — a known input produces the expected
   sites + species update.
2. Constructor validation (count, length, emptiness, duplicates,
   overlap).
3. Composition-mismatch and identity-swap rejection (return ``None``).
4. ``allowed_species`` filtering.
5. Detailed balance — transition matrix symmetric to ~4 sigma over a
   small enumerable canonical state space.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from unittest.mock import MagicMock

import numpy as np
import pytest

from mchammer_moves import IndexSetSwap
from tests.conftest import seeded_uniform


def _make_fake_configuration(occupations: list[int]):
    """Return a minimal mock supplying the ``occupations`` attribute.

    `IndexSetSwap` does not consult the sublattice machinery (the
    index sets are passed in at construction), so a single
    ``.occupations`` attribute suffices.
    """
    config = MagicMock()
    config.occupations = np.array(occupations, dtype=int)
    return config


def _fixed_rng(values: list[float]):
    """Callable returning a fixed sequence of uniform draws."""
    seq = iter(values)

    def draw() -> float:
        return next(seq)

    return draw


def test_index_set_swap_basic_proposal():
    """A swap between two equal-composition index sets returns the
    expected concatenated sites and swapped occupations.
    """
    g1 = [0, 1, 2]
    g2 = [10, 11, 12]
    occupations = [0] * 20
    occupations[0], occupations[1], occupations[2] = 100, 101, 100
    occupations[10], occupations[11], occupations[12] = 100, 101, 100
    # g1 and g2 have the same composition (2x100, 1x101). g1's pattern
    # is (100, 101, 100); g2's is (100, 101, 100) -> identity. Vary g2
    # to make the swap non-identity.
    occupations[10], occupations[11], occupations[12] = 101, 100, 100

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2])
    # First draw selects index 0 (g1); second draw, scaled to (n-1)=1,
    # selects index 0 -> j=0, then j>=i so j+=1 -> j=1 (g2).
    proposal = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert proposal is not None
    sites, species = proposal
    assert sites == [0, 1, 2, 10, 11, 12]
    assert species == [101, 100, 100, 100, 101, 100]


def test_index_set_swap_accepts_composition_mismatch_by_default():
    """Without ``require_matching_composition``, two sets with differing
    species multisets give a valid swap proposal. The swap moves
    composition between the two groups.
    """
    g1 = [0, 1, 2]
    g2 = [10, 11, 12]
    occupations = [0] * 20
    occupations[0], occupations[1], occupations[2] = 100, 100, 100  # all 100
    occupations[10], occupations[11], occupations[12] = 100, 101, 100  # mixed

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2])
    proposal = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert proposal is not None
    sites, species = proposal
    assert sites == g1 + g2
    # g1 receives g2's contents; g2 receives g1's.
    assert species == [100, 101, 100, 100, 100, 100]


def test_index_set_swap_rejects_composition_mismatch_when_required():
    """With ``require_matching_composition=True``, two sets with
    differing species multisets give a ``None`` proposal.
    """
    g1 = [0, 1, 2]
    g2 = [10, 11, 12]
    occupations = [0] * 20
    occupations[0], occupations[1], occupations[2] = 100, 100, 100  # all 100
    occupations[10], occupations[11], occupations[12] = 100, 101, 100  # mixed

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2], require_matching_composition=True)
    assert move.propose(config, _fixed_rng([0.0, 0.0])) is None


def test_index_set_swap_returns_none_on_identity():
    """Two sets currently holding identical occupation patterns give
    a ``None`` proposal — counting these as accepts would inflate the
    per-move acceptance rate without meaningful change.
    """
    g1 = [0, 1, 2]
    g2 = [10, 11, 12]
    occupations = [0] * 20
    pattern = [100, 101, 100]
    for s, z in zip(g1, pattern, strict=True):
        occupations[s] = z
    for s, z in zip(g2, pattern, strict=True):
        occupations[s] = z

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2])
    assert move.propose(config, _fixed_rng([0.0, 0.0])) is None


def test_index_set_swap_allowed_species_filters():
    """A draw whose either set contains a species outside
    ``allowed_species`` yields a ``None`` proposal.
    """
    g1 = [0, 1, 2]
    g2 = [10, 11, 12]
    occupations = [0] * 20
    occupations[0], occupations[1], occupations[2] = 100, 101, 100
    occupations[10], occupations[11], occupations[12] = 100, 101, 102  # 102 disallowed

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2], allowed_species=[100, 101])
    assert move.propose(config, _fixed_rng([0.0, 0.0])) is None


def test_index_set_swap_dispatches_uniformly_across_pairs():
    """Over many proposals, every pair of distinct index sets is
    selected with frequency consistent with uniform sampling.
    """
    g1, g2, g3 = [0, 1], [2, 3], [4, 5]
    # Set up occupations so every pair is a valid (composition-matching,
    # non-identity) swap.
    occupations = [100, 101, 100, 101, 100, 101]
    # All three sets have composition {100, 101}. g1 = (100, 101);
    # g2 = (100, 101); g3 = (100, 101) -> all identical patterns. To
    # avoid identity rejections, perturb.
    occupations = [100, 101, 101, 100, 100, 101]
    # Now g1=(100,101), g2=(101,100), g3=(100,101). g1==g3 (identity);
    # g1!=g2; g2!=g3. Two of the three pairs are non-identity.

    config = _make_fake_configuration(occupations)
    move = IndexSetSwap(index_sets=[g1, g2, g3])

    pair_counts: Counter[tuple[int, int]] = Counter()
    n = 6000
    rng = seeded_uniform(99)
    for _ in range(n):
        result = move.propose(config, rng)
        if result is None:
            # identity-pair case (g1, g3)
            pair_counts[("identity",)] += 1
            continue
        sites, _ = result
        # Decode which two groups were drawn from the returned sites.
        first_two = tuple(sites[:2])
        last_two = tuple(sites[2:])
        groups = (g1, g2, g3)
        i = next(idx for idx, g in enumerate(groups) if tuple(g) == first_two)
        j = next(idx for idx, g in enumerate(groups) if tuple(g) == last_two)
        pair_counts[tuple(sorted((i, j)))] += 1

    # Expect ~ n/3 of each of the three unordered pairs.
    expected = n / 3
    for pair in [(0, 1), (0, 2), (1, 2)]:
        if pair == (0, 2):
            # identity rejection — comes back as ("identity",)
            obs = pair_counts.get(("identity",), 0)
        else:
            obs = pair_counts.get(pair, 0)
        # 5-sigma binomial bound
        assert abs(obs - expected) < 5 * np.sqrt(expected), (
            f"Pair {pair} count {obs} deviates >5 sigma from {expected}"
        )


def test_index_set_swap_detailed_balance():
    """Empirical detailed balance for `IndexSetSwap` on three length-3
    index sets at fixed within-set composition (2x100, 1x101).

    Three index sets, each of length 3, each holding two 100s and one
    101. Within-set composition is preserved by every swap; the
    canonical state space restricted to (2, 1) within each set has
    ``3^3 = 27`` configurations. Run many proposals from each starting
    state and assert the transition matrix is symmetric within ~4
    sigma.
    """
    g1 = (0, 1, 2)
    g2 = (3, 4, 5)
    g3 = (6, 7, 8)
    sets = (g1, g2, g3)
    sp_a, sp_b = 100, 101

    # Per-set state: which of the three positions holds the 101.
    # Joint state: tuple of three positions, one per set.
    states = [(p1, p2, p3) for p1 in range(3) for p2 in range(3) for p3 in range(3)]
    state_index = {s: i for i, s in enumerate(states)}

    def to_occ(state: tuple[int, int, int]) -> list[int]:
        occ = [0] * 9
        for set_idx, pos in enumerate(state):
            for k, site in enumerate(sets[set_idx]):
                occ[site] = sp_b if k == pos else sp_a
        return occ

    def from_occ(occ: list[int]) -> tuple[int, int, int]:
        result = []
        for s in sets:
            for k, site in enumerate(s):
                if int(occ[site]) == sp_b:
                    result.append(k)
                    break
        return tuple(result)  # type: ignore[return-value]

    move = IndexSetSwap(index_sets=[list(g1), list(g2), list(g3)])
    n_per = 4000
    transitions = np.zeros((len(states), len(states)), dtype=int)
    rng = seeded_uniform(2026)
    for src in states:
        config = _make_fake_configuration(to_occ(src))
        for _ in range(n_per):
            result = move.propose(config, rng)
            if result is None:
                # identity-swap: state unchanged
                transitions[state_index[src], state_index[src]] += 1
                continue
            sites, species = result
            cand = list(config.occupations)
            for s, z in zip(sites, species, strict=True):
                cand[s] = z
            transitions[state_index[src], state_index[from_occ(cand)]] += 1

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
        "IndexSetSwap detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )


def test_index_set_swap_detailed_balance_default_mode():
    """Empirical detailed balance for the default mode
    (``require_matching_composition=False``) on three length-2 index
    sets.

    Total composition is fixed at 3 of ``sp_a`` and 3 of ``sp_b``;
    per-group composition spans (2, 0), (1, 1), and (0, 2) across
    states. The state space is ``C(6, 3) = 20`` placements of the
    three ``sp_b`` sites across six sites, and includes
    configurations where the three groups have mismatched
    compositions. Composition-mixing swaps are exercised; the
    transition-count matrix must remain symmetric within ~4 sigma.
    The sibling detailed-balance test uses fixed per-group
    composition, so this case adds the move set the default mode
    introduces.
    """
    g1 = (0, 1)
    g2 = (2, 3)
    g3 = (4, 5)
    sp_a, sp_b = 100, 101

    states = sorted(set(permutations([0, 0, 0, 1, 1, 1])))
    state_index = {s: i for i, s in enumerate(states)}

    def to_occ(state: tuple[int, ...]) -> list[int]:
        return [sp_b if b == 1 else sp_a for b in state]

    def from_occ(occ: list[int]) -> tuple[int, ...]:
        return tuple(0 if int(z) == sp_a else 1 for z in occ)

    move = IndexSetSwap(index_sets=[list(g1), list(g2), list(g3)])
    n_per = 4000
    transitions = np.zeros((len(states), len(states)), dtype=int)
    rng = seeded_uniform(31415)
    for src in states:
        config = _make_fake_configuration(to_occ(src))
        for _ in range(n_per):
            result = move.propose(config, rng)
            if result is None:
                transitions[state_index[src], state_index[src]] += 1
                continue
            sites, species = result
            cand = list(config.occupations)
            for s, z in zip(sites, species, strict=True):
                cand[s] = z
            transitions[state_index[src], state_index[from_occ(cand)]] += 1
    # The transitions must include at least some across-composition
    # moves — otherwise this test is no different from the
    # require_matching_composition=True test and isn't exercising the
    # default mode's added move set.
    assert sum(
        transitions[i, j]
        for i, src in enumerate(states)
        for j, dst in enumerate(states)
        if i != j
        and Counter(to_occ(src)[:2]) != Counter(to_occ(dst)[:2])
    ) > 0, "default-mode test never exercises a composition-changing transition"

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
        "IndexSetSwap default-mode detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )


def test_index_set_swap_rejects_too_few_sets():
    with pytest.raises(ValueError, match="at least two sets"):
        IndexSetSwap(index_sets=[[0, 1, 2]])
    with pytest.raises(ValueError, match="at least two sets"):
        IndexSetSwap(index_sets=[])


def test_index_set_swap_rejects_empty_set():
    with pytest.raises(ValueError, match="empty"):
        IndexSetSwap(index_sets=[[0, 1, 2], []])


def test_index_set_swap_rejects_internal_duplicates():
    with pytest.raises(ValueError, match="duplicate site indices"):
        IndexSetSwap(index_sets=[[0, 1, 0], [2, 3, 4]])


def test_index_set_swap_rejects_mixed_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        IndexSetSwap(index_sets=[[0, 1, 2], [3, 4]])


def test_index_set_swap_rejects_overlapping_sets():
    """Overlapping sets would produce ill-defined swaps (which species
    wins for the shared site?). Caught at construction.
    """
    with pytest.raises(ValueError, match="shares site"):
        IndexSetSwap(index_sets=[[0, 1, 2], [2, 3, 4]])


def test_index_set_swap_rejects_empty_allowed_species():
    """An empty `allowed_species` would silently filter out every
    species and make every proposal return `None`. The intent gap
    (caller meant `None`) is caught at construction.
    """
    with pytest.raises(ValueError, match="allowed_species.*empty"):
        IndexSetSwap(index_sets=[[0, 1, 2], [3, 4, 5]], allowed_species=[])


def test_index_set_swap_accepts_range_objects():
    """`range(N)` is a valid `Sequence[int]`; the constructor accepts it."""
    move = IndexSetSwap(index_sets=[range(0, 3), range(3, 6)])
    assert move.index_sets == [(0, 1, 2), (3, 4, 5)]

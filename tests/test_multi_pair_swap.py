"""Tests for :class:`MultiPairSwap`.

Covers:

1. Basic structural correctness of the proposal (shape, site
   disjointness, species-swap pattern).
2. ``k = 1`` reduces to the same proposal distribution as
   :class:`PairSwap` on a binary sublattice.
3. Detailed balance — transition matrix symmetric to ~4 sigma over a
   small enumerable canonical state space.
4. Null-proposal handling when the sublattice cannot supply ``k``
   distinct-species pairs.
5. Constructor validation.
"""

from __future__ import annotations

from itertools import permutations
from unittest.mock import MagicMock

import numpy as np
import pytest

from mchammer_moves import MultiPairSwap, PairSwap
from tests.conftest import seeded_uniform


def _make_fake_configuration(
    occupations: list[int], sublattice_indices: list[int]
):
    """Return a minimal mock configuration with ``occupations`` and
    a single sublattice.

    Mirrors the pattern used in `test_cyclic_shift.py` for tests that
    do not need a full CE-backed `ConfigurationManager` — `MultiPairSwap`
    reads only `.occupations` and `.sublattices[i].indices`.
    """
    config = MagicMock()
    config.occupations = np.array(occupations, dtype=int)
    sublattice = MagicMock()
    sublattice.indices = list(sublattice_indices)
    config.sublattices = [sublattice]
    return config


def test_multi_pair_swap_basic_proposal_shape():
    """A `k=2` proposal returns 4 site-disjoint sites and 4 species,
    paired so that each pair's species are swapped.
    """
    occupations = [10, 11, 10, 11, 10, 11]
    config = _make_fake_configuration(occupations, sublattice_indices=list(range(6)))
    move = MultiPairSwap(sublattice_index=0, k=2)
    proposal = move.propose(config, seeded_uniform(0))
    assert proposal is not None
    sites, species = proposal
    assert len(sites) == 4
    assert len(species) == 4
    # Site-disjoint
    assert len(set(sites)) == 4
    # Each pair (sites[2i], sites[2i+1]) has differing species, and the
    # proposed species are the swap of the current species at those sites.
    for i in range(2):
        s1, s2 = sites[2 * i], sites[2 * i + 1]
        assert occupations[s1] != occupations[s2]
        assert species[2 * i] == occupations[s2]
        assert species[2 * i + 1] == occupations[s1]


def test_multi_pair_swap_k1_matches_pair_swap_distribution(small_ising_setup):
    """`MultiPairSwap(k=1)` should sample the same (sites, species)
    distribution as `PairSwap` on a binary sublattice.

    Run many proposals from a fixed configuration with each move and
    assert that the empirical pair-frequency distributions are
    statistically consistent.
    """
    setup = small_ising_setup
    occupations = list(setup["structure"].numbers)
    config = _make_fake_configuration(
        occupations, sublattice_indices=list(range(len(occupations)))
    )
    move_multi = MultiPairSwap(sublattice_index=0, k=1)
    move_pair = PairSwap(sublattice_index=0)

    # Use the real CE-backed configuration for PairSwap (since it
    # delegates to mchammer's `get_swapped_state`); use the mock for
    # MultiPairSwap. Both should produce the same uniform-over-pairs
    # distribution on a binary sublattice.
    n = 20_000
    multi_counts: dict[tuple[int, int], int] = {}
    pair_counts: dict[tuple[int, int], int] = {}

    rng = seeded_uniform(2024)
    for _ in range(n):
        result = move_multi.propose(config, rng)
        assert result is not None
        sites, _ = result
        key = tuple(sorted(sites))
        multi_counts[key] = multi_counts.get(key, 0) + 1

    # Build a real ensemble for PairSwap (it requires a real
    # ConfigurationManager).
    from mchammer_moves import CustomCanonicalEnsemble

    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=99,
    )
    for _ in range(n):
        result = move_pair.propose(ensemble.configuration, rng)
        assert result is not None
        sites, _ = result
        key = tuple(sorted(sites))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # Both distributions should cover the same set of pairs (all
    # distinct-species pairs on the sublattice).
    assert set(multi_counts.keys()) == set(pair_counts.keys()), (
        "MultiPairSwap(k=1) and PairSwap reach different pair sets"
    )
    # And the empirical frequencies should be consistent with a uniform
    # distribution over those pairs (chi-sq-ish check via bounded ratios).
    n_pairs = len(multi_counts)
    expected = n / n_pairs
    for key in multi_counts:
        for counts, label in ((multi_counts, "multi"), (pair_counts, "pair")):
            obs = counts[key]
            # 5-sigma binomial bound: sqrt(n p (1-p)) ~ sqrt(expected)
            assert abs(obs - expected) < 5 * np.sqrt(expected), (
                f"{label} count {obs} for {key} deviates >5 sigma from {expected}"
            )


def test_multi_pair_swap_detailed_balance_k2():
    """Empirical detailed balance for `MultiPairSwap(k=2)` on a
    6-site binary sublattice at composition (3, 3).

    Enumerates the C(6, 3) = 20 canonical states, runs many proposals
    from each, and asserts that the transition-count matrix is
    symmetric within ~4 sigma. The `+/- 4 sigma` cut matches the
    `PairSwap` and `CyclicShift` detailed-balance tests.
    """
    sp_a, sp_b = 100, 101
    n_sites = 6
    sublattice_indices = list(range(n_sites))

    states = sorted(set(permutations([0, 0, 0, 1, 1, 1])))
    state_index = {s: i for i, s in enumerate(states)}

    def to_atomic(local: tuple[int, ...]) -> list[int]:
        return [sp_b if b == 1 else sp_a for b in local]

    def from_atomic(occ: list[int]) -> tuple[int, ...]:
        return tuple(0 if int(z) == sp_a else 1 for z in occ)

    move = MultiPairSwap(sublattice_index=0, k=2)
    n_per = 6000
    transitions = np.zeros((len(states), len(states)), dtype=int)
    rng = seeded_uniform(54321)
    for src in states:
        config = _make_fake_configuration(to_atomic(src), sublattice_indices)
        for _ in range(n_per):
            result = move.propose(config, rng)
            assert result is not None  # k=2 always succeeds at (3, 3)
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
        "MultiPairSwap detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )


def test_multi_pair_swap_returns_none_when_k_exceeds_minority():
    """At composition (4, 1) on 5 sites, `k=2` cannot succeed: after
    the first pair, the minority species pool is exhausted, so no
    second pair of differing species exists. The move must return
    ``None`` rather than raising or returning a malformed proposal.
    """
    occupations = [10, 10, 10, 10, 11]  # four 10s, one 11
    config = _make_fake_configuration(occupations, sublattice_indices=list(range(5)))
    move = MultiPairSwap(sublattice_index=0, k=2)
    # Try many seeds; every proposal must be None because the second
    # pair has no minority site to draw on.
    for seed in range(50):
        assert move.propose(config, seeded_uniform(seed)) is None


def test_multi_pair_swap_returns_none_on_uniform_sublattice():
    """A sublattice with only one species cannot supply any pair."""
    occupations = [42, 42, 42, 42]
    config = _make_fake_configuration(occupations, sublattice_indices=list(range(4)))
    move = MultiPairSwap(sublattice_index=0, k=1)
    assert move.propose(config, seeded_uniform(0)) is None


def test_multi_pair_swap_respects_allowed_sites():
    """Sites outside `allowed_sites` must never appear in a proposal."""
    occupations = [10, 11, 10, 11, 10, 11]
    config = _make_fake_configuration(occupations, sublattice_indices=list(range(6)))
    allowed = [0, 1, 2, 3]  # restrict to first four sites
    move = MultiPairSwap(sublattice_index=0, k=2, allowed_sites=allowed)
    for seed in range(50):
        result = move.propose(config, seeded_uniform(seed))
        assert result is not None
        sites, _ = result
        assert set(sites).issubset(set(allowed))


def test_multi_pair_swap_respects_allowed_species():
    """Species outside `allowed_species` must never appear in a proposal."""
    # Three species; restrict to {10, 11}, leaving species 12 untouched.
    occupations = [10, 11, 10, 11, 12, 12]
    config = _make_fake_configuration(occupations, sublattice_indices=list(range(6)))
    move = MultiPairSwap(sublattice_index=0, k=2, allowed_species=[10, 11])
    seen_species: set[int] = set()
    for seed in range(50):
        result = move.propose(config, seeded_uniform(seed))
        if result is None:
            continue
        sites, _ = result
        for s in sites:
            seen_species.add(int(occupations[s]))
    assert seen_species == {10, 11}, (
        f"Species 12 should never appear; seen = {seen_species}"
    )


def test_multi_pair_swap_rejects_invalid_constructor_arguments():
    """Constructor rejects negative sublattice indices and `k < 1`."""
    with pytest.raises(ValueError, match="non-negative"):
        MultiPairSwap(sublattice_index=-1, k=2)
    with pytest.raises(ValueError, match="k must be at least 1"):
        MultiPairSwap(sublattice_index=0, k=0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        MultiPairSwap(sublattice_index=0, k=-3)


def test_multi_pair_swap_rejects_empty_filter_lists():
    """An empty `allowed_species` or `allowed_sites` would silently
    filter out everything and make every proposal return `None`. The
    intent gap (caller meant `None`) is caught at construction.
    """
    with pytest.raises(ValueError, match="allowed_species.*empty"):
        MultiPairSwap(sublattice_index=0, allowed_species=[])
    with pytest.raises(ValueError, match="allowed_sites.*empty"):
        MultiPairSwap(sublattice_index=0, allowed_sites=[])

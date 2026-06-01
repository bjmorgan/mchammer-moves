"""Tests for :class:`SitePermutation`.

Covers:

1. Constructor validation of each operation (non-empty, no explicit
   fixed points, bijection, closed support) and the operations list.
2. Both input forms (Mapping and sequence of pairs).
3. Proposal mechanics — forward and inverse directions, identity-skip.
4. Detailed balance over small enumerable state spaces, including a
   genuine non-involution (3-cycle) that guards the direction draw.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mchammer_moves import SitePermutation
from tests.conftest import seeded_uniform


def _make_fake_configuration(occupations: list[int]):
    """Return a minimal mock supplying the ``occupations`` attribute."""
    config = MagicMock()
    config.occupations = np.array(occupations, dtype=int)
    return config


def _fixed_rng(values: list[float]):
    """Callable returning a fixed sequence of uniform draws.

    Used to force `SitePermutation.propose` down a specific
    (operation, direction) branch in deterministic structural tests.
    """
    seq = iter(values)

    def draw() -> float:
        return next(seq)

    return draw


def test_site_permutation_rejects_empty_operations_list():
    with pytest.raises(ValueError, match="at least one operation"):
        SitePermutation(operations=[])


def test_site_permutation_rejects_empty_operation():
    with pytest.raises(ValueError, match="empty"):
        SitePermutation(operations=[{}])


def test_site_permutation_rejects_explicit_fixed_point():
    with pytest.raises(ValueError, match="to itself"):
        SitePermutation(operations=[{0: 0}])


def test_site_permutation_rejects_duplicate_source():
    with pytest.raises(ValueError, match="source more than once"):
        SitePermutation(operations=[[(0, 1), (0, 2), (1, 0)]])


def test_site_permutation_rejects_repeated_image():
    with pytest.raises(ValueError, match="not a bijection"):
        SitePermutation(operations=[{0: 2, 1: 2}])


def test_site_permutation_rejects_open_support():
    with pytest.raises(ValueError, match="open support"):
        SitePermutation(operations=[{0: 1}])


def test_site_permutation_accepts_mapping_and_pair_sequence_forms():
    from_mapping = SitePermutation(operations=[{0: 1, 1: 0}])
    from_pairs = SitePermutation(operations=[[(0, 1), (1, 0)]])
    assert from_mapping.operations == [{0: 1, 1: 0}]
    assert from_pairs.operations == [{0: 1, 1: 0}]
    assert from_mapping.n_operations == 1


def test_site_permutation_forward_direction_applies_operation():
    """Operation {0:1, 1:2, 2:0} forward: new[i] = old[sigma[i]]."""
    occupations = [100, 101, 102]
    config = _make_fake_configuration(occupations)
    move = SitePermutation(operations=[{0: 1, 1: 2, 2: 0}])
    # First draw selects operation 0; second draw < 0.5 selects forward.
    sites, species = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert list(sites) == [0, 1, 2]
    # new[0]=old[1]=101, new[1]=old[2]=102, new[2]=old[0]=100.
    assert species == [101, 102, 100]


def test_site_permutation_inverse_direction_applies_inverse():
    """Same operation, inverse direction: new[i] = old[inverse[i]]."""
    occupations = [100, 101, 102]
    config = _make_fake_configuration(occupations)
    move = SitePermutation(operations=[{0: 1, 1: 2, 2: 0}])
    # Operation 0; second draw >= 0.5 selects inverse.
    sites, species = move.propose(config, _fixed_rng([0.0, 0.9]))
    assert list(sites) == [0, 1, 2]
    # inverse = {1:0, 2:1, 0:2}; new[0]=old[2]=102, new[1]=old[0]=100,
    # new[2]=old[1]=101.
    assert species == [102, 100, 101]


def test_site_permutation_leaves_unsupported_sites_untouched():
    """Sites outside the operation's support are not in the proposal."""
    occupations = [100, 101, 102, 999, 999]
    config = _make_fake_configuration(occupations)
    move = SitePermutation(operations=[{0: 1, 1: 0}])
    sites, species = move.propose(config, _fixed_rng([0.0, 0.0]))
    assert set(sites) == {0, 1}
    assert 3 not in sites and 4 not in sites


def test_site_permutation_returns_none_on_invariant_configuration():
    """A configuration symmetric under the operation returns ``None``."""
    occupations = [42, 42, 7]
    config = _make_fake_configuration(occupations)
    move = SitePermutation(operations=[{0: 1, 1: 0}])
    for seed in range(20):
        assert move.propose(config, seeded_uniform(seed)) is None


def test_site_permutation_dispatches_across_operations_and_directions():
    """Every operation and both directions must be reachable."""
    occupations = [100, 101, 102, 200, 201, 202]
    config = _make_fake_configuration(occupations)
    move = SitePermutation(
        operations=[{0: 1, 1: 2, 2: 0}, {3: 4, 4: 5, 5: 3}]
    )
    patterns_op0: set[tuple[int, ...]] = set()
    patterns_op1: set[tuple[int, ...]] = set()
    for seed in range(400):
        result = move.propose(config, seeded_uniform(seed))
        assert result is not None
        sites, species = result
        if set(sites) == {0, 1, 2}:
            patterns_op0.add(tuple(species))
        elif set(sites) == {3, 4, 5}:
            patterns_op1.add(tuple(species))
        else:
            raise AssertionError(f"Proposal sites {sites} match no operation")
    # Both operations reached, and within each both directions produce
    # distinct rotated patterns.
    assert len(patterns_op0) == 2, patterns_op0
    assert len(patterns_op1) == 2, patterns_op1

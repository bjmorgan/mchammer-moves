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

"""Shared pytest fixtures.

The tests build a small Ising-like cluster expansion on an FCC ``Au`` host
with two allowed species (``Ag``/``Au``). This is the same toy CE used in
the mchammer documentation; it is fast to evaluate and gives a non-trivial
energy landscape suitable for exercising sampling correctness.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator


def seeded_uniform(seed: int) -> Callable[[], float]:
    """Return a deterministic ``next_random_number``-style callable.

    Wraps a per-instance ``random.Random(seed)`` so successive calls
    produce a reproducible stream of uniform ``[0, 1)`` floats without
    touching Python's global RNG. Tests that previously seeded
    ``random.seed(N)`` then called ``move.propose(config)`` now use
    ``move.propose(config, seeded_uniform(N))`` instead.
    """
    return random.Random(seed).random


def _build_small_ising_setup() -> dict:
    """Return a fresh structure + CE + calculator for a 2x2x2 FCC Ising CE."""
    prim = bulk("Au", a=4.0)
    cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=["Ag", "Au"])
    parameters = [0.0, 0.0, 0.1, -0.02]
    ce = ClusterExpansion(cs, parameters)

    structure = prim.repeat(2)
    for k in range(len(structure) // 2):
        structure[k].symbol = "Ag"

    calc = ClusterExpansionCalculator(structure, ce)
    return {
        "structure": structure,
        "cluster_expansion": ce,
        "calculator": calc,
        "sublattice_index": 0,
    }


@pytest.fixture
def small_ising_setup():
    """Build a fresh Ising CE setup for a single test.

    The calculator is bound to the ``structure`` it's built on, so each
    ensemble needs its own calculator. This fixture returns a fresh
    setup on each invocation.
    """
    return _build_small_ising_setup()


@pytest.fixture
def small_ising_factory():
    """Factory yielding fresh Ising CE setups (one per call).

    Used by tests that compare two ensembles on independent copies of
    the same CE.
    """
    return _build_small_ising_setup

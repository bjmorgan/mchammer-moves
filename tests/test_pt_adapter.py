"""Tests for the ``mchammer_pt`` adapter layer.

These exercise the :class:`CustomReplica` duck-typed interface and
verify that a ``SerialPool`` built via :func:`make_serial_pool` can
drive ``CanonicalParallelTempering`` end-to-end.
"""

from __future__ import annotations

import pytest

from mchammer_moves import PairSwap


def test_custom_replica_quacks_like_replica(small_ising_setup):
    """``CustomReplica`` must expose the methods ``SerialPool`` calls."""
    pytest.importorskip("mchammer_pt")
    from mchammer_moves.pt_adapter import CustomReplica

    setup = small_ising_setup
    replica = CustomReplica(
        cluster_expansion=setup["cluster_expansion"],
        atoms=setup["structure"],
        temperature=600.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=99,
    )
    # Required attributes
    assert isinstance(replica.temperature, float)
    occ0 = replica.current_occupations()
    replica.advance(50)
    occ1 = replica.current_occupations()
    assert occ0.shape == occ1.shape
    # set_occupations round-trip
    replica.set_occupations(occ0)
    assert (replica.current_occupations() == occ0).all()


def test_make_serial_pool_drives_parallel_tempering(small_ising_setup):
    """End-to-end: build a pool, hand it to PT, run a few cycles."""
    pytest.importorskip("mchammer_pt")
    from mchammer_pt import CanonicalParallelTempering

    from mchammer_moves.pt_adapter import make_serial_pool

    setup = small_ising_setup
    temperatures = [100.0, 300.0, 900.0]
    pool = make_serial_pool(
        cluster_expansion=setup["cluster_expansion"],
        atoms=setup["structure"],
        temperatures=temperatures,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=2024,
    )
    pt = CanonicalParallelTempering(
        cluster_expansion=setup["cluster_expansion"],
        atoms=setup["structure"],
        temperatures=temperatures,
        block_size=20,
        random_seed=7,
        pool=pool,
    )
    history = pt.run(n_cycles=3)

    # Energies snapshot has expected shape; one per replica.
    assert pool.current_energies().shape == (len(temperatures),)
    # Per-cycle energy log has shape (n_cycles+1, n_replicas).
    assert history.energies_per_cycle.shape == (4, len(temperatures))

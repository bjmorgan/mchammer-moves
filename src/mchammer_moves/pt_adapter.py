"""Adapter layer for using :class:`CustomCanonicalEnsemble` with ``mchammer_pt``.

The default ``mchammer_pt.replica.Replica`` constructs an
``mchammer.CanonicalEnsemble`` directly inside its initialiser, which
prevents drop-in replacement of the ensemble class. ``CanonicalParallelTempering``,
however, accepts an arbitrary ``ReplicaPool`` via its ``pool`` argument.
The path that does not require modifying ``mchammer_pt`` is therefore to
build a ``Replica``-shaped wrapper around :class:`CustomCanonicalEnsemble`,
collect those wrappers into a ``SerialPool``, and pass the pool through.

This module provides :class:`CustomReplica` (the wrapper) and
:func:`make_serial_pool` (the convenience constructor).

``mchammer_pt`` is imported lazily — it is not a required dependency of
the rest of the package.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator

from mchammer_moves.ensemble import CustomCanonicalEnsemble

if TYPE_CHECKING:
    from mchammer_moves.moves.base import Move


class CustomReplica:
    """Replica-shaped wrapper around :class:`CustomCanonicalEnsemble`.

    Implements the duck-typed contract that
    ``mchammer_pt.parallel.serial.SerialPool`` expects of its replica
    objects: ``temperature`` (property), ``advance``, ``current_energy``,
    ``current_occupations``, ``set_occupations``,
    ``attach_mchammer_observer``, and ``data_container``.

    Each replica owns a private RNG state snapshot; ``advance`` swaps
    the global ``random`` state in and out around the inner
    ``ensemble.run`` call so that two replicas in the same process do
    not contaminate each other's draws. This mirrors the behaviour of
    ``mchammer_pt.replica.Replica``.

    Parameters
    ----------
    cluster_expansion
        icet :class:`ClusterExpansion` defining the energy.
    atoms
        Starting structure (copied, not mutated).
    temperature
        Replica temperature in kelvin.
    moves
        List of ``(Move, weight)`` tuples for
        :class:`CustomCanonicalEnsemble`.
    random_seed
        Seed for this replica's RNG stream.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        temperature: float,
        moves: list[tuple["Move", float]],
        random_seed: int,
    ) -> None:
        self._temperature = float(temperature)
        atoms_copy: Atoms = atoms.copy()
        calculator = ClusterExpansionCalculator(atoms_copy, cluster_expansion)
        caller_state = random.getstate()
        try:
            self._ensemble = CustomCanonicalEnsemble(
                structure=atoms_copy,
                calculator=calculator,
                temperature=self._temperature,
                moves=moves,
                random_seed=int(random_seed),
            )
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def ensemble(self) -> CustomCanonicalEnsemble:
        """The underlying :class:`CustomCanonicalEnsemble`."""
        return self._ensemble

    def advance(self, n_steps: int) -> None:
        previous_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble.run(int(n_steps))
            self._rng_state = random.getstate()
        finally:
            random.setstate(previous_state)

    def current_energy(self) -> float:
        return float(
            self._ensemble.calculator.calculate_total(
                occupations=self._ensemble.configuration.occupations
            )
        )

    def current_occupations(self) -> np.ndarray:
        return self._ensemble.configuration.occupations.copy()

    def set_occupations(self, occupations: np.ndarray) -> None:
        occ = np.asarray(occupations, dtype=int)
        self._ensemble.update_occupations(
            sites=list(range(len(occ))), species=list(occ)
        )

    def attach_mchammer_observer(self, observer) -> None:
        self._ensemble.attach_observer(observer)

    def data_container(self):
        return self._ensemble.data_container


def make_serial_pool(
    cluster_expansion: ClusterExpansion,
    atoms: Atoms,
    temperatures: Sequence[float],
    moves: list[tuple["Move", float]],
    random_seed: int,
):
    """Build a ``SerialPool`` of :class:`CustomReplica` instances.

    Convenience wrapper that mirrors the per-replica seeding logic in
    ``mchammer_pt.canonical.CanonicalParallelTempering``: spawns one
    child seed per replica and one master seed, returning a pool ready
    to pass as the ``pool=`` argument to
    ``CanonicalParallelTempering``.

    Parameters
    ----------
    cluster_expansion
        icet :class:`ClusterExpansion`.
    atoms
        Starting structure; each replica is constructed from a copy.
    temperatures
        Non-decreasing temperature ladder.
    moves
        ``(Move, weight)`` list applied to every replica.
    random_seed
        Master seed; per-replica seeds are derived deterministically.

    Returns
    -------
    mchammer_pt.parallel.serial.SerialPool
        Pool of :class:`CustomReplica` objects, in temperature order.
    """
    from mchammer_pt.parallel.serial import SerialPool

    seed_sequence = np.random.SeedSequence(int(random_seed))
    child_seeds = seed_sequence.spawn(len(temperatures) + 1)
    replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]
    replicas = [
        CustomReplica(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            temperature=T,
            moves=moves,
            random_seed=seed,
        )
        for T, seed in zip(temperatures, replica_seeds, strict=True)
    ]
    return SerialPool(replicas)

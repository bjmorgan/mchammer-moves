"""Local Boltzmann-sampling harness for `CanonicalEnsemble` subclasses.

`assert_boltzmann_sampling` anchors a custom `CanonicalEnsemble`
subclass's stationarity against an analytic Boltzmann fixture: a
four-site one-dimensional chain (2 Cu + 2 Au, NN-only pair ECI,
dE ~ 3 kT at 1000 K). It builds the fixture, drives the ensemble
directly, and asserts that the empirical energy-class populations
match the analytic Boltzmann probabilities within a binomial-sigma
tolerance.

The function-level docstring on `assert_boltzmann_sampling` describes
the fixture in full. The public surface is one function plus one
constant (`FIXTURE_CHAIN_INDICES`) for callers whose ensemble kwargs
depend on the fixture's chain geometry.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import kB
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble

FIXTURE_CHAIN_INDICES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3),)
"""Site indices of the chains in the fixture, in geometric order.

The fixture is a single one-dimensional chain of four sites. Moves
that take chain definitions can use this constant to build
fixture-aware ``ensemble_kwargs``, converting to ``list[list[int]]``
for consumers that type their chain argument as a nested list. A unit
`CyclicShift` on the one-chain fixture is energy-preserving and so
non-ergodic; pair it with a `PairSwap` before sampling. Example::

    from mchammer_moves import CyclicShift, PairSwap

    chains = [list(c) for c in FIXTURE_CHAIN_INDICES]
    moves = [
        (PairSwap(sublattice_index=0), 1.0),
        (CyclicShift(cycles=chains), 1.0),
    ]
"""

_FIXTURE_N_SITES = 4
_FIXTURE_TEMPERATURE = 1000.0
_TARGET_GAP_KT = 3.0


def _build_chain_ce_and_atoms() -> tuple[ClusterExpansion, Atoms]:
    """Build a four-site 1D-chain CE calibrated to dE ~ 3 kT at fixture T.

    Cell length is 4 Angstrom along x with isolated y, z. NN cutoff
    1.5 Angstrom captures only x-direction nearest neighbours; the
    second-NN distance along x is 2.0 Angstrom, outside the cutoff.
    """
    primitive = Atoms(
        "Cu",
        positions=[(0.0, 0.0, 0.0)],
        cell=[(1.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)],
        pbc=True,
    )
    cs = ClusterSpace(
        structure=primitive,
        cutoffs=[1.5],
        chemical_symbols=["Cu", "Au"],
    )
    parameters = np.zeros(len(cs), dtype=float)
    # Calibrate the pair ECI so the gap dE = E_alt - E_clust is ~3 kT
    # at the fixture temperature. Energy is linear in the pair ECI so
    # we compute the gap once with a placeholder value, then scale.
    parameters[-1] = 1.0
    ce_probe = ClusterExpansion(cluster_space=cs, parameters=parameters)
    atoms: Atoms = primitive.repeat((_FIXTURE_N_SITES, 1, 1))
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])
    e_clust_unit = float(ce_probe.predict(atoms)) * len(atoms)
    atoms.set_chemical_symbols(["Cu", "Au", "Cu", "Au"])
    e_alt_unit = float(ce_probe.predict(atoms)) * len(atoms)
    gap_unit = e_alt_unit - e_clust_unit
    if abs(gap_unit) < 1e-12:
        raise RuntimeError(
            "ECI probe produced a degenerate energy gap; check ClusterSpace"
        )
    target_gap = _TARGET_GAP_KT * kB * _FIXTURE_TEMPERATURE
    parameters[-1] = target_gap / gap_unit
    ce = ClusterExpansion(cluster_space=cs, parameters=parameters)
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])
    return ce, atoms


def _enumerate_two_cu_microstates() -> list[list[str]]:
    """All six (2 Cu, 2 Au) symbol assignments on the 4 fixture sites."""
    configs: list[list[str]] = []
    for cu_indices in itertools.combinations(range(_FIXTURE_N_SITES), 2):
        symbols = ["Au"] * _FIXTURE_N_SITES
        for i in cu_indices:
            symbols[i] = "Cu"
        configs.append(symbols)
    return configs


def _classify_by_energy(
    ce: ClusterExpansion,
    atoms: Atoms,
    configs: list[list[str]],
) -> dict[float, int]:
    """Group microstates by total CE energy; return {energy: multiplicity}.

    Operates on a copy of `atoms` so the caller's symbol assignment is
    preserved.
    """
    work: Atoms = atoms.copy()
    multiplicities: dict[float, int] = {}
    for symbols in configs:
        work.set_chemical_symbols(symbols)
        e_total = float(ce.predict(work)) * len(work)
        for e_existing in list(multiplicities):
            if abs(e_total - e_existing) < 1e-9:
                multiplicities[e_existing] += 1
                break
        else:
            multiplicities[e_total] = 1
    return multiplicities


def _analytic_class_probabilities(
    multiplicities: dict[float, int], temperature: float
) -> dict[float, float]:
    """Boltzmann probability per energy class: P(E) ~ g(E) exp(-beta E)."""
    beta = 1.0 / (kB * temperature)
    weights = {e: g * np.exp(-e * beta) for e, g in multiplicities.items()}
    Z = sum(weights.values())
    return {e: float(w / Z) for e, w in weights.items()}


def assert_boltzmann_sampling(
    ensemble_cls: type[CanonicalEnsemble],
    *,
    ensemble_kwargs: Mapping[str, Any] | None = None,
    n_samples: int = 10_000,
    sample_interval: int = 50,
    burn_in: int = 5_000,
    seed: int = 0,
    sigma_tolerance: float = 4.0,
) -> None:
    """Assert that `ensemble_cls` samples the local Boltzmann fixture.

    Fixture: a four-site one-dimensional chain (orthorhombic cell with
    periodic length 4 along x, isolated along y and z) with two Cu and
    two Au and an NN-only pair ECI. The six microstates of (2 Cu, 2 Au)
    split into four "clustered" (1 CuCu + 2 CuAu + 1 AuAu NN bonds) and
    two "alternating" (0 CuCu + 4 CuAu + 0 AuAu) configurations, giving
    two distinct CE energies. The pair ECI is calibrated so the energy
    gap is approximately 3 kT at T = 1000 K, producing analytic class
    populations of roughly 0.98 (clustered) and 0.02 (alternating) --
    far enough from a uniform 4:2 stationary distribution that a kernel
    stationary at the wrong distribution is detected at default
    tolerance.

    The function builds a `ClusterExpansionCalculator` and constructs
    `ensemble_cls` directly, advances it for `burn_in` trial steps, then
    collects `n_samples` samples at `sample_interval` step intervals.
    The ensemble's own constructor seeds Python's global RNG from
    `seed`, and successive `run` calls form one continuous stream.
    Empirical class proportions are compared to the analytic Boltzmann
    probabilities; an ``AssertionError`` is raised if any class deviates
    by more than `sigma_tolerance` standard errors of the binomial.

    Parameters
    ----------
    ensemble_cls
        A `CanonicalEnsemble` or subclass.
    ensemble_kwargs
        Extra keyword arguments forwarded to ``ensemble_cls(...)``.
        Cannot include `structure`, `calculator`, `temperature`, or
        `random_seed` (set by the harness). For ensembles whose
        construction depends on fixture geometry, use the
        `FIXTURE_CHAIN_INDICES` constant.
    n_samples
        Number of samples to collect after burn-in.
    sample_interval
        MC trial steps between samples.
    burn_in
        MC trial steps to advance before sampling.
    seed
        Random seed for the ensemble's RNG stream.
    sigma_tolerance
        Maximum binomial-sigma deviation per class before the assertion
        fires.

    Raises
    ------
    AssertionError
        If a sampled energy matches none of the enumerated classes --
        most often a move that breaks canonical composition, taking the
        configuration outside the fixture's (2 Cu, 2 Au) state space --
        or if any class's empirical population deviates from the
        analytic Boltzmann probability by more than `sigma_tolerance`
        sigma.
    RuntimeError
        If the fixture itself produces an unexpected number of energy
        classes (a fixture-invariant failure rather than a stationarity
        assertion failure).
    """
    ce, atoms = _build_chain_ce_and_atoms()
    multiplicities = _classify_by_energy(ce, atoms, _enumerate_two_cu_microstates())
    if len(multiplicities) != 2:
        raise RuntimeError(
            f"Fixture invariant violated: expected exactly 2 distinct "
            f"energy classes for the 4-site chain with NN-only pair ECI; "
            f"got {len(multiplicities)}"
        )
    p_analytic = _analytic_class_probabilities(multiplicities, _FIXTURE_TEMPERATURE)
    class_energies = sorted(multiplicities.keys())

    calculator = ClusterExpansionCalculator(atoms, ce)
    ensemble = ensemble_cls(
        structure=atoms,
        calculator=calculator,
        temperature=_FIXTURE_TEMPERATURE,
        random_seed=seed,
        **(dict(ensemble_kwargs) if ensemble_kwargs else {}),
    )
    ensemble.run(burn_in)
    counts = dict.fromkeys(class_energies, 0)
    for _ in range(n_samples):
        ensemble.run(sample_interval)
        e = float(
            ensemble.calculator.calculate_total(
                occupations=ensemble.configuration.occupations
            )
        )
        for e_class in class_energies:
            if abs(e - e_class) < 1e-6:
                counts[e_class] += 1
                break
        else:
            raise AssertionError(
                f"Sample energy {e!r} matches no enumerated class "
                f"{class_energies!r}"
            )

    for e_class, count in counts.items():
        p_emp = count / n_samples
        p_an = p_analytic[e_class]
        sigma = np.sqrt(p_an * (1.0 - p_an) / n_samples)
        delta = abs(p_emp - p_an)
        assert delta < sigma_tolerance * sigma, (
            f"Class E={e_class:.4f} eV (multiplicity "
            f"{multiplicities[e_class]}): empirical {p_emp:.4f} vs "
            f"analytic {p_an:.4f}, |delta|={delta:.4f} > "
            f"{sigma_tolerance} sigma={sigma_tolerance * sigma:.4f}"
        )

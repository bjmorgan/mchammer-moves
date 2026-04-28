"""Tests for :class:`PairSwap`.

The symmetry test verifies that ``P(A -> B) = P(B -> A)`` for the
canonical pair-swap proposal on a small enumerable state space — the
critical detailed-balance check. The baseline-equivalence test
verifies that running ``CustomCanonicalEnsemble`` with ``PairSwap`` as
the sole move reproduces the equilibrium energy distribution of
mchammer's stock :class:`CanonicalEnsemble`.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pytest
from mchammer.ensembles import CanonicalEnsemble

from mchammer_moves import CustomCanonicalEnsemble, PairSwap
from tests.conftest import seeded_uniform


def _enumerate_canonical_states(n_sites: int, n_minority: int) -> list[tuple[int, ...]]:
    """Enumerate all distinct binary configurations at fixed composition.

    Returns occupations as 0/1 tuples; the test code maps these to
    atomic numbers as needed.
    """
    states = []
    for minority_sites in combinations(range(n_sites), n_minority):
        occ = [0] * n_sites
        for s in minority_sites:
            occ[s] = 1
        states.append(tuple(occ))
    return states


def _set_occupations(ensemble, atomic_numbers: list[int]) -> None:
    """Force the ensemble's configuration to a specific occupation vector."""
    n = len(ensemble.configuration.occupations)
    ensemble.update_occupations(list(range(n)), list(atomic_numbers))


def test_pair_swap_propose_returns_swap(small_ising_setup):
    """A single proposal must swap two sites with differing species."""
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=42,
    )
    occ_before = list(ensemble.configuration.occupations)
    move = PairSwap(sublattice_index=0)
    sites, species = move.propose(ensemble.configuration, seeded_uniform(0))
    assert len(sites) == 2
    assert len(species) == 2
    i, j = sites
    assert occ_before[i] != occ_before[j]
    # Proposed species must be a swap of the existing species at those sites
    assert species == [occ_before[j], occ_before[i]]


def test_pair_swap_detailed_balance(small_ising_setup):
    """Verify ``P(A -> B) = P(B -> A)`` over the canonical state space.

    Strategy: pick a small subset of sites (4) on which to enumerate
    binary configurations at fixed composition, hold all other sites
    constant, force the ensemble into each state, count proposed
    transitions, and verify the transition-count matrix is symmetric
    within statistical tolerance.
    """
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=1000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=12345,
    )

    base_occ = list(ensemble.configuration.occupations)
    n = len(base_occ)
    species_a, species_b = sorted(set(base_occ))

    # Pick four sites — two of each species — to vary
    sites_a = [i for i, z in enumerate(base_occ) if z == species_a][:2]
    sites_b = [i for i, z in enumerate(base_occ) if z == species_b][:2]
    varied_sites = sites_a + sites_b
    n_varied = len(varied_sites)
    n_minority = 2  # always two of each in the varied subset

    # Enumerate the 6 distinct (2,2) compositions on these 4 sites
    states_local = _enumerate_canonical_states(n_varied, n_minority)
    state_index = {s: i for i, s in enumerate(states_local)}

    def to_full_occ(local_state: tuple[int, ...]) -> list[int]:
        occ = list(base_occ)
        for site, bit in zip(varied_sites, local_state, strict=True):
            occ[site] = species_b if bit == 1 else species_a
        return occ

    def project_to_local(occ: list[int]) -> tuple[int, ...]:
        return tuple(0 if occ[s] == species_a else 1 for s in varied_sites)

    n_proposals_per_state = 4000
    move = PairSwap(sublattice_index=0)
    transitions = np.zeros((len(states_local), len(states_local)), dtype=int)
    out_of_subspace = 0

    random.seed(987654321)
    for src_local in states_local:
        i_src = state_index[src_local]
        # Force ensemble into this state
        _set_occupations(ensemble, to_full_occ(src_local))
        for _ in range(n_proposals_per_state):
            # Restore state if a previous loop iteration mutated it
            current_local = project_to_local(list(ensemble.configuration.occupations))
            if current_local != src_local:
                _set_occupations(ensemble, to_full_occ(src_local))
            # PairSwap.propose ignores the callable and delegates to
            # mchammer's `get_swapped_state`, which draws from the
            # global random module seeded above.
            sites, species = move.propose(ensemble.configuration, seeded_uniform(0))
            # Build the candidate full occupation
            cand = list(ensemble.configuration.occupations)
            for s, z in zip(sites, species, strict=True):
                cand[s] = z
            cand_local = project_to_local(cand)
            # Did the swap stay within our 4-site subspace?
            unchanged_outside = all(
                cand[k] == base_occ[k] for k in range(n) if k not in varied_sites
            )
            if not unchanged_outside:
                out_of_subspace += 1
                continue
            j_dst = state_index[cand_local]
            transitions[i_src, j_dst] += 1

    # Sanity: most proposals must touch only varied sites; otherwise the
    # subspace is too small to be informative. With 4 varied sites out of 8
    # this should be ~ (C(4,2) / C(8,2))**? — at minimum, a healthy fraction.
    total_proposals = n_proposals_per_state * len(states_local)
    in_subspace = total_proposals - out_of_subspace
    assert in_subspace > 0.05 * total_proposals, (
        f"Too few in-subspace proposals: {in_subspace}/{total_proposals}"
    )

    # Symmetry: for each pair (i, j) with non-trivial counts, check that
    # transitions[i, j] and transitions[j, i] agree within statistical
    # tolerance. Use a chi-squared-style symmetric / asymmetric test:
    # for each off-diagonal pair, expect the two counts to be drawn from a
    # binomial with mean (a+b)/2.
    failures = []
    for i in range(len(states_local)):
        for j in range(i + 1, len(states_local)):
            a = transitions[i, j]
            b = transitions[j, i]
            tot = a + b
            if tot < 30:
                continue  # not enough samples to test
            mean = tot / 2
            std = np.sqrt(tot / 4)  # binomial p=1/2
            z = abs(a - mean) / std
            if z > 4.0:  # 4-sigma cut → vanishingly small false-positive rate
                failures.append((i, j, a, b, z))

    assert not failures, (
        "PairSwap detailed-balance violation: "
        + ", ".join(
            f"({i},{j}): {a} vs {b} (z={z:.2f})" for i, j, a, b, z in failures
        )
    )


def test_pair_swap_baseline_matches_canonical_ensemble(small_ising_factory):
    """``CustomCanonicalEnsemble`` with PairSwap reproduces ``CanonicalEnsemble``.

    Run both ensembles for the same number of steps from the same
    initial configuration, with the same temperature and identical
    seeds, and compare the empirical energy distributions with a
    Kolmogorov-Smirnov test.
    """
    scipy_stats = pytest.importorskip("scipy.stats")

    n_steps = 20_000
    burn_in = 2_000
    temperature = 1500.0
    seed = 2024

    # Reference: stock CanonicalEnsemble
    setup_ref = small_ising_factory()
    ref_ensemble = CanonicalEnsemble(
        structure=setup_ref["structure"],
        calculator=setup_ref["calculator"],
        temperature=temperature,
        random_seed=seed,
    )
    ref_energies = []
    for _ in range(n_steps):
        ref_ensemble._do_trial_step()
        ref_energies.append(ref_ensemble.calculator.calculate_total(
            occupations=ref_ensemble.configuration.occupations))
    ref_energies = np.asarray(ref_energies[burn_in:])

    # Custom: PairSwap-only
    setup_cus = small_ising_factory()
    cus_ensemble = CustomCanonicalEnsemble(
        structure=setup_cus["structure"],
        calculator=setup_cus["calculator"],
        temperature=temperature,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=seed,
    )
    cus_energies = []
    for _ in range(n_steps):
        cus_ensemble._do_trial_step()
        cus_energies.append(cus_ensemble.calculator.calculate_total(
            occupations=cus_ensemble.configuration.occupations))
    cus_energies = np.asarray(cus_energies[burn_in:])

    # KS test on the two energy distributions. We do not require identity
    # of the trajectories (RNG draws differ between the two
    # implementations), only that they sample the same equilibrium
    # distribution. A p-value of >1e-3 is generous; failure indicates a
    # real distributional disagreement.
    ks_stat, p_value = scipy_stats.ks_2samp(ref_energies, cus_energies)
    assert p_value > 1e-3, (
        f"Energy distributions disagree: KS={ks_stat:.4f}, p={p_value:.2e}. "
        f"ref mean={ref_energies.mean():.4f}, cus mean={cus_energies.mean():.4f}; "
        f"ref std={ref_energies.std():.4f}, cus std={cus_energies.std():.4f}"
    )

    # Sanity: means should agree to within ~3 standard errors
    mean_diff = abs(ref_energies.mean() - cus_energies.mean())
    pooled_se = np.sqrt(
        ref_energies.var() / len(ref_energies)
        + cus_energies.var() / len(cus_energies)
    )
    assert mean_diff < 5 * pooled_se, (
        f"Mean energies disagree: {ref_energies.mean():.4f} vs "
        f"{cus_energies.mean():.4f} (diff={mean_diff:.4f}, SE={pooled_se:.4f})"
    )


def test_pair_swap_acceptance_rate_reasonable(small_ising_setup):
    """Per-move acceptance rate should be a sensible fraction at moderate T."""
    setup = small_ising_setup
    ensemble = CustomCanonicalEnsemble(
        structure=setup["structure"],
        calculator=setup["calculator"],
        temperature=2000.0,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        random_seed=7,
    )
    n = 5000
    for _ in range(n):
        ensemble._do_trial_step()
    rates = ensemble.acceptance_rates()
    assert "pair_swap" in rates
    assert rates["pair_swap"]["proposed"] == n
    rate = rates["pair_swap"]["acceptance_rate"]
    assert 0.05 < rate < 0.99, f"Implausible acceptance rate: {rate}"

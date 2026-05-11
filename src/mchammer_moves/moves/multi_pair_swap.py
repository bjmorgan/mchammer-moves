"""Multi-pair canonical swap trial move."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class MultiPairSwap(Move):
    """``k`` site-disjoint pair swaps applied as one atomic proposal.

    Generalises :class:`PairSwap` to ``k`` simultaneous swaps. At each
    proposal, picks ``k`` site-disjoint pairs of differing-species
    sites on the sublattice (each pair selected as :class:`PairSwap`
    would: site 1 uniformly from the available sites, site 2 uniformly
    from the available sites of different species), and proposes
    swapping the species at all ``2k`` sites in a single update.

    Useful when single-pair swaps are kinetically blocked between
    adjacent minima in deep basins — for example, in the deepest-energy
    Wang-Landau windows of an order-disorder transition where every
    one-swap neighbour of the local minimum is itself a local minimum.
    Larger ``k`` gives larger jumps in configuration space at lower
    per-move acceptance.

    For ``k = 1`` the move is exactly :class:`PairSwap`: same proposal
    distribution, same null-proposal handling.

    Parameters
    ----------
    sublattice_index
        Index of the sublattice on which to swap, as defined by the
        ensemble's underlying :class:`Sublattices` object.
    k
        Number of site-disjoint pair swaps to apply per proposal.
        ``k = 1`` reduces to :class:`PairSwap`; ``k >= 2`` gives
        multi-pair jumps.
    name
        Identifier used for per-move acceptance tracking.
    allowed_species
        Optional list of atomic numbers restricting the swap to a
        subset of species. Sites currently holding a species not in
        this list are excluded from both endpoints of every pair.
    allowed_sites
        Optional list of site indices restricting the swap to a subset
        of sites within the sublattice.

    Raises
    ------
    ValueError
        If ``sublattice_index`` is negative or ``k < 1``.

    Notes
    -----
    Detailed balance: at fixed canonical composition, the species
    counts ``n_X`` on the sublattice are invariant under any sequence
    of valid pair swaps. Each pair's selection probability — pick site
    1 uniformly from non-used sites, pick site 2 uniformly from
    non-used sites of different species — depends only on those
    composition counts and on which sites have already been used in
    earlier pairs of the same proposal. Summed over the ``k!``
    orderings of pair selection that yield the same final state, the
    forward and reverse proposal probabilities for any site-disjoint
    pair-set are equal. Standard Metropolis acceptance therefore
    preserves detailed balance.

    The move returns ``None`` when fewer than ``k`` site-disjoint
    distinct-species pairs are available — for example, near edge
    compositions where the minority-species count drops below ``k``.
    The ensemble counts a ``None`` proposal as a rejection without an
    energy evaluation.
    """

    def __init__(
        self,
        sublattice_index: int = 0,
        k: int = 2,
        name: str = "multi_pair_swap",
        allowed_species: list[int] | None = None,
        allowed_sites: list[int] | None = None,
    ) -> None:
        if sublattice_index < 0:
            raise ValueError(
                f"sublattice_index must be non-negative; got {sublattice_index}. "
                "mchammer's `Sublattices` indexes positively from 0; negative "
                "values silently end-index into the sublattice list and "
                "produce a working but wrong sublattice."
            )
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}.")
        if allowed_species is not None and len(allowed_species) == 0:
            raise ValueError(
                "`allowed_species` is an empty list, which would filter out "
                "every species and make every proposal return `None`. Pass "
                "`None` to apply no species filter."
            )
        if allowed_sites is not None and len(allowed_sites) == 0:
            raise ValueError(
                "`allowed_sites` is an empty list, which would filter out "
                "every site and make every proposal return `None`. Pass "
                "`None` to apply no site filter."
            )
        super().__init__(name)
        self.sublattice_index = sublattice_index
        self.k = k
        self.allowed_species = allowed_species
        self.allowed_sites = allowed_sites

    def propose(
        self,
        configuration: ConfigurationManager,
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose ``k`` site-disjoint pair swaps as one combined update.

        At each of the ``k`` steps, samples a distinct-species pair
        without replacement from the sublattice sites not yet used in
        earlier steps of the same proposal. Returns the concatenated
        ``(sites, species)`` update for all ``k`` swaps, or ``None`` if
        insufficient distinct-species pairs remain at any step.
        """
        sublattice_sites = list(
            configuration.sublattices[self.sublattice_index].indices
        )
        if self.allowed_sites is not None:
            allowed_site_set = set(self.allowed_sites)
            sublattice_sites = [s for s in sublattice_sites if s in allowed_site_set]

        occupations = configuration.occupations
        allowed_species = (
            set(self.allowed_species) if self.allowed_species is not None else None
        )

        used: set[int] = set()
        all_sites: list[int] = []
        all_species: list[int] = []

        for _ in range(self.k):
            available = [
                s
                for s in sublattice_sites
                if s not in used
                and (allowed_species is None or int(occupations[s]) in allowed_species)
            ]
            if len(available) < 2:
                return None

            i1 = int(next_random_number() * len(available))
            site1 = available[i1]
            z1 = int(occupations[site1])

            candidates2 = [s for s in available if int(occupations[s]) != z1]
            if not candidates2:
                return None
            i2 = int(next_random_number() * len(candidates2))
            site2 = candidates2[i2]
            z2 = int(occupations[site2])

            used.add(site1)
            used.add(site2)
            all_sites.extend([site1, site2])
            all_species.extend([z2, z1])

        return all_sites, all_species

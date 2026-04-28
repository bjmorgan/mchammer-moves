"""Standard canonical pair-swap trial move."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mchammer.configuration_manager import SwapNotPossibleError

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class PairSwap(Move):
    """Two-site canonical swap on a single sublattice.

    Selects two sites of differing species from the configured
    sublattice and proposes exchanging their occupations. Implementation
    delegates to :meth:`ConfigurationManager.get_swapped_state`, which
    is the same routine used by mchammer's :class:`CanonicalEnsemble`.

    For a fixed-composition canonical ensemble, the number of
    distinct-species site pairs on a sublattice is composition-invariant,
    so the proposal probability of any specific (sites, species) change
    depends only on lattice geometry and is symmetric in the forward and
    reverse directions. Detailed balance is therefore satisfied under
    standard Metropolis acceptance.

    Parameters
    ----------
    sublattice_index
        Index of the sublattice on which to swap, as defined by the
        ensemble's underlying :class:`Sublattices` object.
    name
        Identifier used for per-move acceptance tracking.
    allowed_species
        Optional list of atomic numbers restricting the swap to a
        subset of species. Passed through to ``get_swapped_state``.
    allowed_sites
        Optional list of site indices restricting the swap to a subset
        of sites. Passed through to ``get_swapped_state``.
    """

    def __init__(
        self,
        sublattice_index: int = 0,
        name: str = "pair_swap",
        allowed_species: list[int] | None = None,
        allowed_sites: list[int] | None = None,
    ) -> None:
        self.name = name
        self.sublattice_index = sublattice_index
        self.allowed_species = allowed_species
        self.allowed_sites = allowed_sites

    def propose(
        self, configuration: "ConfigurationManager"
    ) -> tuple[list[int], list[int]] | None:
        """Propose a swap of two sites with differing species."""
        try:
            sites, species = configuration.get_swapped_state(
                sublattice_index=self.sublattice_index,
                allowed_species=self.allowed_species,
                allowed_sites=self.allowed_sites,
            )
        except SwapNotPossibleError:
            return None
        return list(sites), list(species)

"""Cyclic-shift trial move on an arbitrary set of site-index cycles.

A *cycle* is an ordered list of site indices; the :class:`CyclicShift`
move shifts the species pattern along a single cycle by :math:`\\pm 1`
position with periodic boundaries within the cycle. The cycle's
indices need not be geometrically related — the move operates on
indices only — but typical use cases supply geometrically
collinear indices to implement row translations, ring rotations, or
analogous neighbourhood-preserving moves on lattice systems with
chain-like or ring-like sublattices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class CyclicShift(Move):
    """Shift the species pattern along an index cycle by :math:`\\pm 1`.

    At each proposal, picks one cycle uniformly at random from the
    configured list, picks a direction uniformly from ``{+1, -1}``, and
    proposes assigning each cycle position the species at its neighbour
    in that direction along the cycle (with periodic boundaries within
    the cycle).

    The move treats the supplied indices as opaque labels. Whether the
    cycle corresponds to a physical chain, a ring, or some abstract
    set of related sites is the caller's choice; the detailed-balance
    argument holds either way.

    Detailed balance: the proposal probability for "cycle :math:`c`,
    direction :math:`d`" is :math:`1/(2 N_\\text{cycles})`, depending only
    on the cycle count, not on the configuration. The reverse of a
    :math:`+1` shift along cycle :math:`c` is a :math:`-1` shift along
    the same cycle, with the same selection probability. Standard
    Metropolis acceptance therefore preserves detailed balance.

    Parameters
    ----------
    cycles
        Sequence of cycles, where each cycle is a sequence of site
        indices in the order along which species are to be shifted.
        Cycles may be of different lengths. Site indices are not
        validated against any particular sublattice — that is the
        caller's responsibility.
    name
        Identifier used for per-move acceptance tracking.

    Raises
    ------
    ValueError
        If ``cycles`` is empty, contains an empty cycle, or contains
        a cycle of length 1 (for which a shift is the identity and
        contains no information).
    """

    def __init__(
        self,
        cycles: Sequence[Sequence[int]],
        name: str = "cyclic_shift",
    ) -> None:
        if not cycles:
            raise ValueError("`cycles` must contain at least one cycle.")
        seen: set[int] = set()
        materialised: list[tuple[int, ...]] = []
        for c, cycle in enumerate(cycles):
            cycle_tuple = tuple(cycle)
            if len(cycle_tuple) == 0:
                raise ValueError(f"Cycle {c} is empty.")
            if len(cycle_tuple) == 1:
                raise ValueError(
                    f"Cycle {c} has length 1; a shift on a single-site cycle "
                    "is the identity and not a useful move."
                )
            if len(set(cycle_tuple)) != len(cycle_tuple):
                raise ValueError(
                    f"Cycle {c} contains duplicate site indices ({list(cycle_tuple)}); "
                    "a shift would propose multiple species values for the "
                    "duplicated site."
                )
            overlap = seen & set(cycle_tuple)
            if overlap:
                raise ValueError(
                    f"Cycle {c} shares site(s) {sorted(overlap)} with an "
                    "earlier cycle; overlapping cycles silently over-sample "
                    "shared sites. Likely a chain-construction bug."
                )
            seen.update(cycle_tuple)
            materialised.append(cycle_tuple)
        super().__init__(name)
        self._cycles: list[tuple[int, ...]] = materialised

    @property
    def cycles(self) -> list[tuple[int, ...]]:
        """Copy of the configured cycles.

        Mutating the returned list does not affect the move's internal
        cycle list.
        """
        return list(self._cycles)

    @property
    def n_cycles(self) -> int:
        """Number of cycles configured."""
        return len(self._cycles)

    def propose(
        self,
        configuration: ConfigurationManager,
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose a single-step shift along one cycle.

        Returns all sites in the chosen cycle together with the species
        they would carry after the shift, including any sites whose
        species are unchanged by the shift.
        """
        # Uniform [0, n) integer; the floating-point bias is negligible
        # for n far below 2^52, which holds for any realistic cycle count.
        cycle_index = int(next_random_number() * len(self._cycles))
        direction = 1 if next_random_number() < 0.5 else -1
        cycle = self._cycles[cycle_index]
        occupations = configuration.occupations
        L = len(cycle)

        # Direction +1: each position i in the cycle receives the species
        # of cycle[(i - 1) % L]. Direction -1: cycle[(i + 1) % L].
        offset = -1 if direction == 1 else 1
        new_species = [int(occupations[cycle[(i + offset) % L]]) for i in range(L)]
        # Skip identity proposals (uniform-species cycles) so they do not
        # inflate the per-move acceptance rate. An identity is reversible
        # at zero cost, so treating it as a null proposal does not bias
        # detailed balance; counting it as accepted would simply make
        # uniform-cycle channels look 100% healthy when no useful work
        # is happening.
        current_species = [int(occupations[s]) for s in cycle]
        if new_species == current_species:
            return None
        return list(cycle), new_species

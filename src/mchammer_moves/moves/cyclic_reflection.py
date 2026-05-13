"""Cyclic-reflection trial move on an arbitrary set of site-index cycles.

A *cycle* is an ordered list of site indices; the
:class:`CyclicReflection` move reflects the species pattern along a
single cycle around a chosen pivot, with periodic boundaries within
the cycle. The cycle's indices need not be geometrically related —
the move operates on indices only — but typical use cases supply
geometrically collinear indices to implement long-range reflection
moves on lattice systems with chain-like or ring-like sublattices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class CyclicReflection(Move):
    """Reflect the species pattern along an index cycle around a pivot.

    At each proposal, picks one cycle uniformly at random from the
    configured list, picks an integer pivot uniformly from
    ``{0, 1, …, L - 1}`` where ``L`` is the chosen cycle's length, and
    proposes the cyclic reflection: site at cycle position ``i``
    receives the species currently at cycle position
    ``(2 · pivot - i) mod L``.

    Useful as a long-range complement to :class:`CyclicShift`'s
    nearest-neighbour shifts: a species at one end of a chain can hop
    to the other end in a single accepted move, opening up sampling
    pathways through reflection-symmetric configurations that
    single-step shifts and pair swaps cannot reach efficiently.

    The move treats the supplied indices as opaque labels. Whether the
    cycle corresponds to a physical chain, a ring, or some abstract
    set of related sites is the caller's choice; the detailed-balance
    argument holds either way.

    Detailed balance: cyclic reflection is an involution — applying
    the same ``(cycle, pivot)`` draw twice returns the original
    configuration. The cycle is picked uniformly from the configured
    list, then the pivot is picked uniformly from
    :math:`\\{0, \\ldots, L_c - 1\\}` for the chosen cycle's length
    :math:`L_c` (so different cycles draw pivots from
    cycle-specific ranges, with no rejection sampling). The joint
    selection probability for "cycle :math:`c`, pivot :math:`p`" is
    :math:`1 / (N_\\text{cycles} \\cdot L_c)`, depending only on the
    fixed list of cycles, not on the configuration. The reverse of
    any reflection along ``(c, p)`` is the same reflection along
    ``(c, p)``, with the same selection probability. Standard
    Metropolis acceptance therefore preserves detailed balance.

    Parameters
    ----------
    cycles
        Sequence of cycles, where each cycle is a sequence of site
        indices in the order along which species are to be reflected.
        Cycles may be of different lengths but must each have length
        at least 3 (see Raises). Site indices are not validated
        against any particular sublattice — that is the caller's
        responsibility.
    name
        Identifier used for per-move acceptance tracking.

    Raises
    ------
    ValueError
        If ``cycles`` is empty, contains an empty cycle, contains a
        cycle of length 1 or 2, contains within-cycle duplicate site
        indices, or contains cycles that share site indices with each
        other.

    Notes
    -----
    Length-1 cycles are rejected because a reflection on a single
    site is the identity. Length-2 cycles are rejected because, with
    integer pivots and cyclic semantics, both sites are fixed under
    every pivot (the pivot itself plus its antipode, which for
    ``L = 2`` is the only other site), so every reflection is the
    identity. Length :math:`\\geq 3` admits at least one
    non-identity reflection.

    Identity-skip: cycles whose current species pattern is invariant
    under the chosen reflection (uniform-species cycles, or
    palindromic cycles reflected around their symmetry axis) return
    ``None`` so the per-move acceptance rate stays meaningful and the
    null is recorded in ``MoveStats.null_rate``. Identity is
    reversible at zero cost, so dropping these as nulls does not bias
    detailed balance.
    """

    def __init__(
        self,
        cycles: Sequence[Sequence[int]],
        name: str = "cyclic_reflection",
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
                    f"Cycle {c} has length 1; a reflection on a single-site "
                    "cycle is the identity and not a useful move."
                )
            if len(cycle_tuple) == 2:
                raise ValueError(
                    f"Cycle {c} has length 2; with integer pivots and cyclic "
                    "semantics both sites are fixed under every reflection, "
                    "so every proposal would be the identity."
                )
            if len(set(cycle_tuple)) != len(cycle_tuple):
                raise ValueError(
                    f"Cycle {c} contains duplicate site indices ({list(cycle_tuple)}); "
                    "a reflection would propose multiple species values for the "
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
        """Propose a reflection of one cycle around a randomly-chosen pivot.

        Returns all sites in the chosen cycle together with the species
        they would carry after the reflection, including any sites whose
        species are unchanged. ``None`` is returned when the chosen
        ``(cycle, pivot)`` produces an identity proposal — uniform-species
        cycles always do, and palindromic cycles do for the pivot at the
        symmetry axis.
        """
        # Uniform [0, n) integer; the floating-point bias is negligible
        # for n far below 2^52, which holds for any realistic cycle count.
        cycle_index = int(next_random_number() * len(self._cycles))
        cycle = self._cycles[cycle_index]
        L = len(cycle)
        pivot = int(next_random_number() * L)

        occupations = configuration.occupations
        new_species = [
            int(occupations[cycle[(2 * pivot - i) % L]]) for i in range(L)
        ]
        current_species = [int(occupations[s]) for s in cycle]
        if new_species == current_species:
            return None
        return list(cycle), new_species

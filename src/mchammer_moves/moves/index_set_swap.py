"""Group-occupation swap trial move on a fixed list of index sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class IndexSetSwap(Move):
    """Swap occupations between two equal-length index sets.

    A generic group-permutation primitive: given a fixed list of
    index sets ``[g_0, g_1, …]`` of common length ``L``, picks two
    distinct sets uniformly at random and proposes exchanging the
    occupations site-by-site between them. The two sets must currently
    hold the same multiset of species — otherwise the swap would
    change each set's internal composition, which is typically not
    the intended kinetic move and is rejected as a null proposal.

    Useful as the primitive underneath project-specific group moves —
    chain swaps on a regular cubic anion sublattice, motif swaps in an
    intermetallic, layer swaps in a slab, and so on. The caller
    supplies the index sets; this class is geometry-agnostic.

    Parameters
    ----------
    index_sets
        Sequence of index sets. All sets must have the same length and
        be pairwise site-disjoint. The site indices are opaque labels;
        no sublattice or geometry validation is performed.
    name
        Identifier used for per-move acceptance tracking.
    allowed_species
        Optional list of atomic numbers. If supplied, proposals are
        rejected (``None`` returned) when either of the two drawn sets
        currently holds a species outside the allowed list.

    Raises
    ------
    ValueError
        If ``index_sets`` contains fewer than two sets, contains an
        empty set, contains a set with duplicated indices, contains
        sets of differing length, or contains sets that share any
        site index.

    Notes
    -----
    Detailed balance: the unordered pair ``{g_i, g_j}`` is drawn
    uniformly at random from the ``C(N, 2)`` distinct pairs at each
    proposal, giving a selection probability that depends only on the
    fixed list of index sets, not on the configuration. Swapping
    ``(g_i, g_j)`` exchanges the two sets' entire contents, so each
    set's composition (and the joint composition) is preserved by the
    move. Any pair valid in the forward direction is therefore also
    valid in the reverse direction with the same selection
    probability. Standard Metropolis acceptance preserves detailed
    balance.

    The composition-match check rejects pairs whose current
    occupations have differing species multisets — those proposals
    would change each set's internal composition, which the move's
    detailed-balance argument does not address (and which is rarely
    the intended kinetic move). The identity-skip rejects proposals
    where the two sets already hold identical occupation patterns;
    counting these as rejections rather than accepts keeps the
    per-move acceptance rate meaningful, and the rejected proposals
    do not bias detailed balance because the identity is reversible
    at zero cost.
    """

    def __init__(
        self,
        index_sets: Sequence[Sequence[int]],
        name: str = "index_set_swap",
        allowed_species: list[int] | None = None,
    ) -> None:
        if len(index_sets) < 2:
            raise ValueError(
                f"`index_sets` must contain at least two sets; got {len(index_sets)}."
            )
        materialised: list[tuple[int, ...]] = []
        seen: set[int] = set()
        first_length: int | None = None
        for i, group in enumerate(index_sets):
            group_tuple = tuple(group)
            if len(group_tuple) == 0:
                raise ValueError(f"Index set {i} is empty.")
            if len(set(group_tuple)) != len(group_tuple):
                raise ValueError(
                    f"Index set {i} contains duplicate site indices "
                    f"({list(group_tuple)})."
                )
            if first_length is None:
                first_length = len(group_tuple)
            elif len(group_tuple) != first_length:
                raise ValueError(
                    f"Index set {i} has length {len(group_tuple)}; expected "
                    f"{first_length}. All index sets must have the same length."
                )
            overlap = seen & set(group_tuple)
            if overlap:
                raise ValueError(
                    f"Index set {i} shares site(s) {sorted(overlap)} with an "
                    "earlier set; index sets must be pairwise disjoint."
                )
            seen.update(group_tuple)
            materialised.append(group_tuple)
        super().__init__(name)
        self._index_sets: list[tuple[int, ...]] = materialised
        self.allowed_species = allowed_species

    @property
    def index_sets(self) -> list[tuple[int, ...]]:
        """Copy of the configured index sets.

        Mutating the returned list does not affect the move's internal
        list.
        """
        return list(self._index_sets)

    @property
    def n_index_sets(self) -> int:
        """Number of index sets configured."""
        return len(self._index_sets)

    def propose(
        self,
        configuration: ConfigurationManager,
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose an occupation swap between two randomly chosen sets.

        Picks two distinct index sets uniformly at random from the
        configured list, reads their current occupations, and proposes
        the site-by-site swap. Returns ``None`` if either set holds a
        species outside ``allowed_species``, if the two sets have
        different species multisets (composition mismatch), or if their
        current occupation patterns are already identical (identity
        swap).
        """
        n = len(self._index_sets)
        # Draw two distinct uniform indices in [0, n) without replacement.
        i = int(next_random_number() * n)
        j = int(next_random_number() * (n - 1))
        if j >= i:
            j += 1
        g1, g2 = self._index_sets[i], self._index_sets[j]

        occupations = configuration.occupations
        occ_g1 = [int(occupations[s]) for s in g1]
        occ_g2 = [int(occupations[s]) for s in g2]

        if self.allowed_species is not None:
            allowed = set(self.allowed_species)
            if not allowed.issuperset(occ_g1) or not allowed.issuperset(occ_g2):
                return None

        if Counter(occ_g1) != Counter(occ_g2):
            return None

        if occ_g1 == occ_g2:
            return None

        return list(g1) + list(g2), occ_g2 + occ_g1

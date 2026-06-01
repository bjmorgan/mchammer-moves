"""Site-permutation trial move applying a fixed index permutation.

A *site permutation* is a bijection of site indices supplied by the
caller as a sparse mapping in which each entry ``i: j`` means site ``i``
takes the occupation currently on site ``j`` (so the proposal sets
``new[i] = old[j]``); sites not named are fixed points. The
:class:`SitePermutation` move applies one of a configured list of such
permutations to the current occupations, proposing a single correlated
jump along that operation. The indices are opaque labels: the move
neither knows nor checks whether a permutation corresponds to a lattice
symmetry. Typical callers reduce a geometric operation (a reflection
across a ``<100>`` plane, a point inversion, a rotation) to an index
mapping and pass it in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager

OperationInput = Mapping[int, int] | Sequence[tuple[int, int]]


class _Operation(NamedTuple):
    """A validated operation and its precomputed inverse.

    ``support`` is the tuple of moved site indices (the mapping's keys,
    which equal the set of its values for a valid permutation).
    ``forward`` is the operation as supplied — entry ``i: j`` means site
    ``i`` takes the occupation of site ``j`` (``new[i] = old[j]``);
    ``inverse`` is the reverse mapping, applied when the inverse
    direction is drawn. Both directions are stored so
    :meth:`SitePermutation.propose` can apply either without recomputing
    or risking the two getting out of step.
    """

    support: tuple[int, ...]
    forward: dict[int, int]
    inverse: dict[int, int]


class SitePermutation(Move):
    """Apply a fixed permutation of site occupations.

    At each proposal, picks one operation uniformly at random from the
    configured list, then applies that operation or its inverse, each
    with probability one half, to the current occupations: the site at
    index ``i`` receives the species currently at index ``sigma[i]``,
    where ``sigma`` is the chosen operation (forward) or its inverse.
    Sites outside the operation's support are unchanged.

    The move is the most general *occupation-permuting* move with a
    detailed-balance guarantee enforced by the move itself. It is
    composition-preserving by construction — permuting occupations
    cannot change the multiset of species. It pairs naturally with
    :class:`PairSwap`: random pair swaps generate the full symmetric
    group and do the local ergodic mixing, while ``SitePermutation``
    makes large correlated jumps along a chosen operation but reaches
    only the orbit its supplied operations generate.

    Each operation is a sparse mapping in which entry ``i: j`` means
    site ``i`` takes the occupation currently on site ``j``. The mapping
    must be a bijection on its support — the set of sites it writes (the
    keys) must equal the set of sites it reads (the values) — so that
    every moved site both gives up and receives an occupation. Sites not
    named in an operation are fixed points and are left untouched. For an
    involution such as a reflection or point inversion the read/write
    distinction is immaterial (paired sites simply exchange occupations);
    it only fixes the direction convention for non-involutions.

    Detailed balance: selection of an ``(operation, direction)`` pair
    has probability :math:`1 / (2K)` for :math:`K` operations,
    independent of the configuration. The multiset of applied
    permutations :math:`\\{\\sigma_k, \\sigma_k^{-1}\\}` is closed
    under inversion with equal weights by construction, and
    :math:`\\sigma \\cdot A = B` iff :math:`\\sigma^{-1} \\cdot B = A`,
    so the total selection weight for transitions :math:`A \\to B`
    equals that for :math:`B \\to A`. Standard Metropolis acceptance
    therefore preserves detailed balance. For an involution
    :math:`\\sigma^{-1} = \\sigma` the two direction branches coincide.

    Parameters
    ----------
    operations
        Non-empty sequence of operations. Each operation is a sparse
        mapping in which entry ``i: j`` means site ``i`` takes the
        occupation currently on site ``j``, given either as a
        ``Mapping[int, int]`` or as a sequence of ``(i, j)`` pairs. Sites
        not named are fixed points. Site indices are opaque labels; no
        sublattice or geometry validation is performed.
    name
        Identifier used for per-move acceptance tracking.

    Raises
    ------
    ValueError
        If ``operations`` is empty, contains an empty operation,
        contains an explicit ``i == j`` entry (a fixed point), names the
        same site as a key more than once, maps two sites to the same
        source (not a bijection), or has open support (the set of keys
        differs from the set of values).

    Notes
    -----
    Indices are opaque. If a mapping sends a site to one on a different
    sublattice with a different allowed-species set, the swap can produce
    an invalid configuration. For a genuine lattice symmetry operation
    this never arises (images lie on the same sublattice); like
    :class:`CyclicReflection`, this is the caller's responsibility rather
    than something the move validates.

    Configurations invariant under the chosen operation return ``None``
    (identity-skip) so the per-move acceptance rate stays meaningful and
    the null is recorded in ``MoveStats.null_rate``. Identity is
    reversible at zero cost, so dropping these as nulls does not bias
    detailed balance.
    """

    def __init__(
        self,
        operations: Sequence[OperationInput],
        name: str = "site_permutation",
    ) -> None:
        if len(operations) == 0:
            raise ValueError("`operations` must contain at least one operation.")
        materialised: list[_Operation] = []
        for idx, op in enumerate(operations):
            raw_items = op.items() if isinstance(op, Mapping) else op
            items = [tuple(r) for r in raw_items]
            if len(items) == 0:
                raise ValueError(
                    f"Operation {idx} is empty; an empty operation is the "
                    "identity and not a useful move."
                )
            forward: dict[int, int] = {}
            for raw in items:
                if len(raw) != 2:
                    raise ValueError(
                        f"Operation {idx} entry {raw!r} is not a (site, source) "
                        "pair."
                    )
                site, read_from = int(raw[0]), int(raw[1])
                if site == read_from:
                    raise ValueError(
                        f"Operation {idx} maps site {site} to itself; omit "
                        "fixed points rather than listing them explicitly."
                    )
                if site in forward:
                    raise ValueError(
                        f"Operation {idx} names site {site} as a key more "
                        "than once."
                    )
                forward[site] = read_from
            sources = list(forward.values())
            if len(set(sources)) != len(sources):
                raise ValueError(
                    f"Operation {idx} is not a bijection: two sites read from "
                    "the same source."
                )
            if set(forward.keys()) != set(sources):
                raise ValueError(
                    f"Operation {idx} has open support: the set of keys "
                    f"{sorted(forward.keys())} differs from the set of values "
                    f"{sorted(sources)}, so it is not a permutation. Every moved "
                    "site must both give up and receive an occupation."
                )
            inverse = {read_from: site for site, read_from in forward.items()}
            support = tuple(forward.keys())
            materialised.append(_Operation(support, forward, inverse))
        super().__init__(name)
        self._operations = materialised

    @property
    def operations(self) -> list[dict[int, int]]:
        """Copy of the configured operations as ``{i: j}`` maps.

        Each entry ``i: j`` means site ``i`` takes the occupation of
        site ``j`` (``new[i] = old[j]``). Mutating the returned list or
        its dicts does not affect the move's internal state.
        """
        return [dict(op.forward) for op in self._operations]

    @property
    def n_operations(self) -> int:
        """Number of operations configured."""
        return len(self._operations)

    def propose(
        self,
        configuration: ConfigurationManager,
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose a permutation of one operation's occupations.

        Picks one operation uniformly, then applies it (forward) or its
        inverse, each with probability one half. Returns the operation's
        support sites together with the species they would carry after
        the chosen permutation, or ``None`` when the configuration is
        invariant under the chosen operation (identity-skip).
        """
        # Uniform [0, K) integer; the floating-point bias is negligible
        # for K far below 2^52, which holds for any realistic operation
        # count.
        k = int(next_random_number() * len(self._operations))
        operation = self._operations[k]
        # The direction draw is unconditional: skipping it for operations
        # that happen to be involutions would silently break detailed
        # balance for any non-involution operation.
        sigma = operation.forward if next_random_number() < 0.5 else operation.inverse

        occupations = configuration.occupations
        new_species = [int(occupations[sigma[i]]) for i in operation.support]
        current_species = [int(occupations[i]) for i in operation.support]
        if new_species == current_species:
            return None
        return list(operation.support), new_species

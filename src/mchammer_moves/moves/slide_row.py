"""Row-translation trial move.

A *row* is a list of site indices in geometric order along a chain. The
:class:`SlideRow` move shifts the species pattern along a single row by
:math:`\\pm 1` site, with periodic boundary conditions within the row.
For period-:math:`p` ordered chains, the move connects equivalent
period-:math:`p` configurations without ever passing through a
disordered intermediate — useful for sampling between inter-chain phase
relationships when standard single-site swaps are kinetically blocked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class SlideRow(Move):
    """Translate the species pattern along a chain by :math:`\\pm 1`.

    At each proposal, picks one row uniformly at random from the
    configured list, picks a direction uniformly from ``{+1, -1}``, and
    proposes assigning each row site the species of its neighbour in
    that direction along the row (with periodic boundaries within the
    row).

    Detailed balance: the proposal probability for "row :math:`r`,
    direction :math:`d`" is :math:`1/(2 N_\\text{rows})`, depending only
    on the row count, not on the configuration. The reverse of a
    :math:`+1` slide along row :math:`r` is a :math:`-1` slide along
    the same row, with the same selection probability. Standard
    Metropolis acceptance therefore preserves detailed balance.

    Parameters
    ----------
    rows
        List of rows, where each row is a list of site indices in
        geometric order along the chain. Rows may be of different
        lengths. Site indices are not validated against any particular
        sublattice — that is the caller's responsibility.
    name
        Identifier used for per-move acceptance tracking.

    Raises
    ------
    ValueError
        If ``rows`` is empty, contains an empty row, or contains a row
        of length 1 (for which a slide is the identity and contains no
        information).
    """

    def __init__(self, rows: list[list[int]], name: str = "slide_row") -> None:
        if not rows:
            raise ValueError("`rows` must contain at least one row.")
        for r, row in enumerate(rows):
            if len(row) == 0:
                raise ValueError(f"Row {r} is empty.")
            if len(row) == 1:
                raise ValueError(
                    f"Row {r} has length 1; a slide on a single-site row is "
                    "the identity and not a useful move."
                )
        self.name = name
        self._rows: list[tuple[int, ...]] = [tuple(row) for row in rows]

    @property
    def rows(self) -> list[tuple[int, ...]]:
        """Read-only view of the configured rows."""
        return list(self._rows)

    @property
    def n_rows(self) -> int:
        """Number of rows configured."""
        return len(self._rows)

    def propose(
        self,
        configuration: "ConfigurationManager",
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose a single-step slide along one row.

        Returns all sites in the chosen row together with the species
        they would carry after the slide. Sites whose species are
        unchanged by the slide (e.g., where a neighbour happens to
        share the same species) are still included — including them
        keeps the proposal characterised purely by ``(row, direction)``,
        which is what the detailed-balance argument requires.
        """
        # Uniform [0, n) integer; the floating-point bias is negligible
        # for n far below 2^52, which holds for any realistic row count.
        row_index = int(next_random_number() * len(self._rows))
        direction = 1 if next_random_number() < 0.5 else -1
        row = self._rows[row_index]
        occupations = configuration.occupations
        L = len(row)

        # Direction +1: each site i in the row receives the species of
        # row[(i - 1) % L]. Direction -1: row[(i + 1) % L].
        offset = -1 if direction == 1 else 1
        new_species = [int(occupations[row[(i + offset) % L]]) for i in range(L)]
        return list(row), new_species

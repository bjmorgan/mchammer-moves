"""Abstract base class for custom Monte Carlo trial moves."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


class Move(ABC):
    """Abstract base class for a Monte Carlo trial move.

    Subclasses implement :meth:`propose`, which returns the (sites,
    species) pair describing the proposed change to the current
    configuration, or ``None`` for a null proposal. The proposal is then
    evaluated by the ensemble using its standard energy and acceptance
    machinery.

    Detailed balance under standard Metropolis acceptance requires that
    the probability of proposing any specific (sites, species) change
    depends only on lattice geometry and the move definition — not on
    the current configuration. Subclasses must guarantee this by
    construction. For example, the canonical pair swap satisfies this
    because, for fixed composition, the number of distinct-species pairs
    on a sublattice is composition-invariant.

    Attributes
    ----------
    name
        Human-readable identifier used for per-move acceptance tracking.
    """

    name: str

    @abstractmethod
    def propose(
        self, configuration: "ConfigurationManager"
    ) -> tuple[list[int], list[int]] | None:
        """Propose a trial move from the current configuration.

        Parameters
        ----------
        configuration
            The current ensemble configuration. Subclasses should treat
            this as read-only — committing the move is the ensemble's
            responsibility.

        Returns
        -------
        tuple[list[int], list[int]] | None
            A pair ``(sites, species)`` giving the site indices to be
            updated and the new atomic numbers for those sites, or
            ``None`` to indicate a null proposal that the ensemble
            should reject without an energy evaluation.
        """
        ...

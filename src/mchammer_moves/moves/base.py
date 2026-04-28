"""Abstract base class for custom Monte Carlo trial moves."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mchammer.configuration_manager import (  # type: ignore[import-untyped]
        ConfigurationManager,
    )


class Move(ABC):
    """Abstract base class for a Monte Carlo trial move.

    Subclasses implement :meth:`propose`, which returns the (sites,
    species) pair describing the proposed change to the current
    configuration, or ``None`` for a null proposal. The proposal is then
    evaluated by the ensemble using its standard energy and acceptance
    machinery.

    Detailed balance under standard Metropolis acceptance requires that
    the forward and reverse proposal probabilities are equal for every
    pair of states the move connects. The simplest way to guarantee
    this is to make proposal probabilities depend only on lattice
    geometry, not on the current configuration — for example, the
    canonical pair swap (where, for fixed composition, the number of
    distinct-species pairs on a sublattice is composition-invariant).
    Symmetric state-dependent proposal probabilities are also fine;
    only asymmetric ones break the contract and require a
    proposal-ratio correction in the acceptance criterion.

    Attributes
    ----------
    name
        Human-readable identifier used for per-move acceptance tracking.
    """

    name: str

    @abstractmethod
    def propose(
        self,
        configuration: ConfigurationManager,
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        """Propose a trial move from the current configuration.

        Parameters
        ----------
        configuration
            The current ensemble configuration. Subclasses should treat
            this as read-only — committing the move is the ensemble's
            responsibility.
        next_random_number
            Zero-argument callable returning a uniform float in
            ``[0, 1)``. Subclasses requiring randomness should draw
            from this callable rather than Python's global ``random``
            module so that the move's randomness is tied to the
            ensemble's seeded RNG stream and remains reproducible
            under per-replica RNG isolation.

        Returns
        -------
        tuple[list[int], list[int]] | None
            A pair ``(sites, species)`` giving the site indices to be
            updated and the new atomic numbers for those sites, or
            ``None`` to indicate a null proposal that the ensemble
            should reject without an energy evaluation.
        """
        ...

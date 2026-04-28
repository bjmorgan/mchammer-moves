"""Canonical ensemble that draws trial moves from a user-supplied list."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ase import Atoms
from ase.units import kB
from mchammer.calculators.base_calculator import BaseCalculator
from mchammer.ensembles import CanonicalEnsemble

if TYPE_CHECKING:
    from mchammer_moves.moves.base import Move


class CustomCanonicalEnsemble(CanonicalEnsemble):
    """Canonical ensemble parameterised by an arbitrary list of moves.

    Drop-in replacement for :class:`mchammer.ensembles.CanonicalEnsemble`
    that overrides :meth:`_do_trial_step` to draw a move from a
    user-supplied weighted list. Each draw delegates to the move's
    :meth:`Move.propose` to obtain a candidate change, then reuses the
    standard ensemble machinery — :meth:`_get_property_change`,
    :meth:`_acceptance_condition`, and :meth:`update_occupations` — so
    that detailed balance with Metropolis acceptance is preserved
    provided the move's proposal probabilities are state-independent.

    The public interface inherited from ``CanonicalEnsemble`` is left
    intact, so external orchestrators (notably ``mchammer_pt``) can use
    this ensemble without modification.

    Parameters
    ----------
    structure
        Atomic configuration; defines the initial occupation vector.
    calculator
        Energy calculator (typically a
        :class:`ClusterExpansionCalculator`).
    temperature
        Temperature :math:`T` in units consistent with
        ``boltzmann_constant``.
    moves
        List of ``(move, weight)`` tuples. Weights need not sum to one;
        they are normalised internally. At each trial step, one move is
        selected with probability proportional to its weight.
    boltzmann_constant
        Boltzmann constant; default ``ase.units.kB``.
    user_tag, random_seed, dc_filename, data_container,
    data_container_write_period, ensemble_data_write_interval,
    trajectory_write_interval
        Forwarded to :class:`CanonicalEnsemble`.

    Notes
    -----
    The ``sublattice_probabilities`` argument of ``CanonicalEnsemble`` is
    not exposed. Sublattice selection is the responsibility of each
    :class:`Move` (e.g., :class:`PairSwap` takes a ``sublattice_index``).
    """

    def __init__(
        self,
        structure: Atoms,
        calculator: BaseCalculator,
        temperature: float,
        moves: list[tuple["Move", float]],
        user_tag: str | None = None,
        boltzmann_constant: float = kB,
        random_seed: int | None = None,
        dc_filename: str | None = None,
        data_container: str | None = None,
        data_container_write_period: float = 600,
        ensemble_data_write_interval: int | None = None,
        trajectory_write_interval: int | None = None,
    ) -> None:
        if not moves:
            raise ValueError("`moves` must contain at least one (move, weight) entry.")
        for entry in moves:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[1], (int, float))
            ):
                raise TypeError(
                    "`moves` must be a list of (Move, weight) tuples; "
                    f"got entry {entry!r}."
                )
            if entry[1] <= 0:
                raise ValueError(
                    f"Move weights must be strictly positive; got {entry[1]} "
                    f"for move {entry[0].name!r}."
                )

        super().__init__(
            structure=structure,
            calculator=calculator,
            temperature=temperature,
            user_tag=user_tag,
            boltzmann_constant=boltzmann_constant,
            random_seed=random_seed,
            dc_filename=dc_filename,
            data_container=data_container,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval,
        )

        self._moves: list[Move] = [m for m, _ in moves]
        self._move_weights: list[float] = [float(w) for _, w in moves]
        self._total_weight: float = sum(self._move_weights)
        self._move_accept_counts: Counter = Counter()
        self._move_reject_counts: Counter = Counter()

        names = [m.name for m in self._moves]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Move names must be unique for per-move tracking; got {names}."
            )

    @property
    def moves(self) -> list["Move"]:
        """The moves registered with the ensemble (read-only view)."""
        return list(self._moves)

    @property
    def move_weights(self) -> list[float]:
        """The (un-normalised) weights for each registered move."""
        return list(self._move_weights)

    def _do_trial_step(self) -> int:
        """Carry out one Monte Carlo trial step.

        Picks a move by weight, asks it for a proposal, evaluates the
        energy change with :meth:`_get_property_change`, and applies
        Metropolis acceptance. Per-move accept/reject counts are updated;
        the integer return value (0 or 1) feeds the inherited global
        counters in :class:`BaseEnsemble`.

        Move selection and the move's own randomness both draw from
        :meth:`_next_random_number`, so the entire trial-step sequence
        is reproducible from the ensemble's seeded RNG stream and
        survives `mchammer_pt`'s per-replica RNG isolation.
        """
        move = self._weighted_move_choice()
        proposal = move.propose(self.configuration, self._next_random_number)
        if proposal is None:
            self._move_reject_counts[move.name] += 1
            return 0
        sites, species = proposal
        potential_diff = self._get_property_change(sites, species)
        if self._acceptance_condition(potential_diff):
            self.update_occupations(sites, species)
            self._move_accept_counts[move.name] += 1
            return 1
        self._move_reject_counts[move.name] += 1
        return 0

    def _weighted_move_choice(self) -> "Move":
        """Pick a move by weight, drawing from the seeded RNG stream."""
        threshold = self._next_random_number() * self._total_weight
        acc = 0.0
        # Linear scan; the move count is small (typically 2-5) and the
        # extra clarity beats bisect-on-cumulative-weights at this size.
        for move, weight in zip(self._moves, self._move_weights):
            acc += weight
            if threshold < acc:
                return move
        # Floating-point edge case: threshold slipped past the final
        # cumulative sum. Fall through to the last move.
        return self._moves[-1]

    def acceptance_rates(self) -> dict[str, dict[str, float | int]]:
        """Return per-move acceptance statistics.

        Returns
        -------
        dict[str, dict]
            Mapping from move name to a dictionary with keys
            ``accepted``, ``rejected``, ``proposed`` (sum of the two),
            and ``acceptance_rate`` (``accepted / proposed`` or 0 if no
            proposals have been made).
        """
        stats: dict[str, dict[str, float | int]] = {}
        for move in self._moves:
            accepted = self._move_accept_counts[move.name]
            rejected = self._move_reject_counts[move.name]
            proposed = accepted + rejected
            stats[move.name] = {
                "accepted": accepted,
                "rejected": rejected,
                "proposed": proposed,
                "acceptance_rate": accepted / proposed if proposed > 0 else 0.0,
            }
        return stats

    def reset_acceptance_counts(self) -> None:
        """Zero the per-move accept/reject counters.

        Useful for separating equilibration and production statistics.
        Does not affect the inherited global counters.
        """
        self._move_accept_counts.clear()
        self._move_reject_counts.clear()

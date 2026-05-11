"""Canonical ensemble that draws trial moves from a user-supplied list."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ase import Atoms
from ase.units import kB
from mchammer.calculators.base_calculator import BaseCalculator
from mchammer.ensembles import CanonicalEnsemble

if TYPE_CHECKING:
    from mchammer_moves.moves.base import Move


@dataclass(frozen=True)
class MoveStats:
    """Per-move acceptance statistics returned by `acceptance_rates`.

    Args:
        accepted: Number of proposals accepted by the Metropolis
            criterion.
        rejected: Number of proposals evaluated for energy and
            rejected by the Metropolis criterion.
        null_proposed: Number of trials where the move returned
            ``None`` (no candidate proposed, no energy evaluation).
            Examples: a `PairSwap` on a single-species sublattice;
            a `MultiPairSwap` on a sublattice with fewer than ``k``
            of the minority species; an `IndexSetSwap` whose drawn
            pair has mismatched composition or already-identical
            occupations. A move with ``accepted == 0`` and
            ``null_rate == 1`` is structurally infeasible on the
            current configuration and will never advance the chain
            until either the configuration or the move's
            constraints change.

    The ``proposed``, ``acceptance_rate``, and ``null_rate``
    properties are computed from these counters.
    """

    accepted: int
    rejected: int
    null_proposed: int = 0

    @property
    def proposed(self) -> int:
        """Total number of trials (``accepted + rejected + null_proposed``)."""
        return self.accepted + self.rejected + self.null_proposed

    @property
    def acceptance_rate(self) -> float:
        """Fraction of trials accepted by Metropolis; ``0.0`` if no trials."""
        return self.accepted / self.proposed if self.proposed > 0 else 0.0

    @property
    def null_rate(self) -> float:
        """Fraction of trials where the move returned ``None``; ``0.0`` if no trials.

        Diagnostic for moves that are structurally unable to propose
        a candidate on the current configuration. A persistently
        high ``null_rate`` (especially ``1.0``) at non-trivial trial
        counts indicates the move is misconfigured for the
        sublattice composition or filter constraints in use.
        """
        return self.null_proposed / self.proposed if self.proposed > 0 else 0.0


class CustomCanonicalEnsemble(CanonicalEnsemble):  # type: ignore[misc]
    """Canonical ensemble parameterised by an arbitrary list of moves.

    Drop-in replacement for :class:`mchammer.ensembles.CanonicalEnsemble`
    that overrides :meth:`_do_trial_step` to draw a move from a
    user-supplied weighted list. Each draw delegates to the move's
    :meth:`Move.propose` to obtain a candidate change, then reuses the
    standard ensemble machinery — :meth:`_get_property_change`,
    :meth:`_acceptance_condition`, and :meth:`update_occupations` — so
    that detailed balance with Metropolis acceptance is preserved
    provided each move's forward and reverse proposal probabilities are
    equal. (State-independent proposal probabilities are sufficient for
    this; symmetric state-dependent probabilities also satisfy it.)

    The public interface inherited from ``CanonicalEnsemble`` is left
    intact, so external orchestrators that accept a `CanonicalEnsemble`
    extension point can use this ensemble without modification.

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

    For multiprocess parallel tempering via
    ``mchammer_pt.CanonicalParallelTempering.process_pool``, this class
    and every registered :class:`Move` subclass must be importable by
    fully qualified name in the spawn workers — i.e. defined in a
    ``.py`` module file, not in a Jupyter cell or a function body.
    ``mchammer_pt.ProcessPool`` rejects interactive-``__main__`` and
    function-local classes up-front.
    """

    def __init__(
        self,
        structure: Atoms,
        calculator: BaseCalculator,
        temperature: float,
        moves: list[tuple[Move, float]],
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
        self._move_accept_counts: Counter[str] = Counter()
        self._move_reject_counts: Counter[str] = Counter()
        self._move_null_counts: Counter[str] = Counter()
        # Snapshots taken at each `_get_ensemble_data` call so the
        # data-container column reports per-interval (not cumulative)
        # acceptance, matching mchammer's `acceptance_ratio` convention.
        self._last_recorded_accept_counts: Counter[str] = Counter()
        self._last_recorded_reject_counts: Counter[str] = Counter()
        self._last_recorded_null_counts: Counter[str] = Counter()

        names = [m.name for m in self._moves]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Move names must be unique for per-move tracking; got {names}."
            )

    @property
    def moves(self) -> list[Move]:
        """Copy of the moves registered with the ensemble."""
        return list(self._moves)

    @property
    def move_weights(self) -> list[float]:
        """The (un-normalised) weights for each registered move."""
        return list(self._move_weights)

    def _do_trial_step(self) -> int:
        """Carry out one Monte Carlo trial step.

        Picks a move by weight, asks it for a proposal, evaluates the
        energy change with :meth:`_get_property_change`, and applies
        Metropolis acceptance. Per-move accept / reject / null counts
        are updated; the integer return value (0 or 1) feeds the
        inherited global counters in :class:`BaseEnsemble`.

        If the move returns ``None`` (a null proposal — e.g., a
        :class:`PairSwap` on a sublattice with no distinct-species
        sites), no energy evaluation is performed and the step is
        counted on the move's null counter (separate from
        Metropolis-rejected proposals so that ``MoveStats.null_rate``
        can diagnose structurally infeasible moves).

        Move selection and the move's own randomness both draw from
        :meth:`_next_random_number`, so the entire trial-step sequence
        is reproducible from the ensemble's seeded RNG stream and
        survives `mchammer_pt`'s per-replica RNG isolation.
        """
        move = self._weighted_move_choice()
        proposal = move.propose(self.configuration, self._next_random_number)
        if proposal is None:
            self._move_null_counts[move.name] += 1
            return 0
        sites, species = proposal
        potential_diff = self._get_property_change(sites, species)
        if self._acceptance_condition(potential_diff):
            self.update_occupations(sites, species)
            self._move_accept_counts[move.name] += 1
            return 1
        self._move_reject_counts[move.name] += 1
        return 0

    def _weighted_move_choice(self) -> Move:
        """Pick a move by weight, drawing from the seeded RNG stream."""
        threshold = self._next_random_number() * self._total_weight
        acc = 0.0
        # Linear scan; the move count is small (typically 2-5) and the
        # extra clarity beats bisect-on-cumulative-weights at this size.
        for move, weight in zip(self._moves, self._move_weights, strict=True):
            acc += weight
            if threshold < acc:
                return move
        # Floating-point edge case: threshold slipped past the final
        # cumulative sum. Fall through to the last move.
        return self._moves[-1]

    def acceptance_rates(self) -> dict[str, MoveStats]:
        """Return per-move acceptance statistics, keyed by move name."""
        return {
            move.name: MoveStats(
                accepted=self._move_accept_counts[move.name],
                rejected=self._move_reject_counts[move.name],
                null_proposed=self._move_null_counts[move.name],
            )
            for move in self._moves
        }

    def _get_ensemble_data(self) -> dict[str, float]:
        """Extend the standard ensemble-data dict with per-move stats.

        Adds two keys per registered move:
        ``<move_name>_acceptance_rate`` (fraction of trials accepted
        by Metropolis) and ``<move_name>_null_rate`` (fraction of
        trials where the move returned ``None``). Both are recorded
        as *per-interval* rates over the trials since the previous
        call to this method, matching `BaseEnsemble`'s native
        ``acceptance_ratio`` convention so the per-move and global
        columns of the resulting `BaseDataContainer` are directly
        comparable. The fields round-trip through the HDF5 bundle
        written by `mchammer_pt`, so per-move statistics are
        recoverable from a `ProcessPool` run without observer
        forwarding.

        Cumulative statistics remain available via `acceptance_rates`,
        which reads the lifetime counters directly; only the
        data-container fields are per-interval.
        """
        data = super()._get_ensemble_data()
        for move in self._moves:
            cum_accepted = self._move_accept_counts[move.name]
            cum_rejected = self._move_reject_counts[move.name]
            cum_null = self._move_null_counts[move.name]
            interval_accepted = (
                cum_accepted - self._last_recorded_accept_counts[move.name]
            )
            interval_rejected = (
                cum_rejected - self._last_recorded_reject_counts[move.name]
            )
            interval_null = (
                cum_null - self._last_recorded_null_counts[move.name]
            )
            if (
                interval_accepted < 0
                or interval_rejected < 0
                or interval_null < 0
            ):
                # Should be unreachable: the cumulative counters only
                # ever increase, and snapshots are taken from them.
                # A negative interval delta means a snapshot was not
                # cleared in step with its cumulative counter — most
                # commonly a refactor of `reset_acceptance_counts`
                # that forgot one of the `_last_recorded_*` Counters.
                raise RuntimeError(
                    f"Per-interval counter went negative for move "
                    f"{move.name!r}: accepted={interval_accepted}, "
                    f"rejected={interval_rejected}, null={interval_null}. "
                    "This indicates a snapshot counter was not cleared "
                    "alongside its cumulative counter."
                )
            interval_proposed = interval_accepted + interval_rejected + interval_null
            if interval_proposed > 0:
                data[f"{move.name}_acceptance_rate"] = (
                    interval_accepted / interval_proposed
                )
                data[f"{move.name}_null_rate"] = interval_null / interval_proposed
            else:
                data[f"{move.name}_acceptance_rate"] = 0.0
                data[f"{move.name}_null_rate"] = 0.0
            self._last_recorded_accept_counts[move.name] = cum_accepted
            self._last_recorded_reject_counts[move.name] = cum_rejected
            self._last_recorded_null_counts[move.name] = cum_null
        return data

    def reset_acceptance_counts(self) -> None:
        """Zero the per-move accept / reject / null counters.

        Resets both the lifetime counters (read by `acceptance_rates`)
        and the per-interval snapshot counters used by
        `_get_ensemble_data`, so the next data-container write reports
        an interval starting from zero.

        Useful for separating equilibration and production statistics.
        Does not affect the inherited global counters.
        """
        self._move_accept_counts.clear()
        self._move_reject_counts.clear()
        self._move_null_counts.clear()
        self._last_recorded_accept_counts.clear()
        self._last_recorded_reject_counts.clear()
        self._last_recorded_null_counts.clear()

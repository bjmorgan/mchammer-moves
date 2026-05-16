"""Canonical ensemble that draws trial moves from a user-supplied list."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ase import Atoms
from ase.units import kB
from mchammer.calculators.base_calculator import BaseCalculator
from mchammer.ensembles import CanonicalEnsemble, WangLandauEnsemble

from mchammer_moves.moves.base import Move

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class MoveStats:
    """Per-move acceptance statistics returned by `acceptance_rates`.

    Parameters
    ----------
    accepted
        Number of proposals accepted by the Metropolis criterion.
    rejected
        Number of proposals evaluated for energy and rejected by the
        Metropolis criterion.
    null_proposed
        Number of trials where the move returned ``None`` (no candidate
        proposed, no energy evaluation). Examples: a `PairSwap` on a
        single-species sublattice; a `MultiPairSwap` on a sublattice
        with fewer than ``k`` of the minority species; an
        `IndexSetSwap` whose drawn pair already holds identical
        occupations (or, with ``require_matching_composition=True``,
        has mismatched composition). A move with ``accepted == 0`` and
        ``null_rate == 1`` is structurally infeasible on the current
        configuration and will never advance the chain until either the
        configuration or the move's constraints change.

    Notes
    -----
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


class MoveDispatcher:
    """Weighted move selection and per-move bookkeeping.

    Owns the move list, weights, per-move accept/reject/null counters,
    and per-interval rate computation for data-container integration.
    Ensemble adapters hold a ``MoveDispatcher`` instance and delegate
    bookkeeping to it; the dispatcher has no knowledge of ensembles,
    configurations, or energy evaluation.

    Parameters
    ----------
    moves
        List of ``(move, weight)`` tuples. Weights need not sum to one;
        they are used as relative weights. Move names must be unique
        (per-move tracking keys on name).

    Raises
    ------
    ValueError
        If ``moves`` is empty, contains non-positive weights, or
        contains duplicate move names.
    TypeError
        If any entry is not a ``(Move, numeric)`` tuple.
    """

    def __init__(self, moves: list[tuple[Move, float]]) -> None:
        if not moves:
            raise ValueError(
                "`moves` must contain at least one (move, weight) entry."
            )
        for entry in moves:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], Move)
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

        self._moves: list[Move] = [m for m, _ in moves]
        self._move_weights: list[float] = [float(w) for _, w in moves]
        self._total_weight: float = sum(self._move_weights)

        names = [m.name for m in self._moves]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Move names must be unique for per-move tracking; got {names}."
            )

        self._accept_counts: Counter[str] = Counter()
        self._reject_counts: Counter[str] = Counter()
        self._null_counts: Counter[str] = Counter()
        self._last_recorded_accept: Counter[str] = Counter()
        self._last_recorded_reject: Counter[str] = Counter()
        self._last_recorded_null: Counter[str] = Counter()

    @property
    def moves(self) -> list[Move]:
        """Copy of the registered moves."""
        return list(self._moves)

    @property
    def move_weights(self) -> list[float]:
        """The (un-normalised) weights for each registered move."""
        return list(self._move_weights)

    def choose(self, next_random_number: Callable[[], float]) -> Move:
        """Select and return one registered move, sampled by weight.

        Parameters
        ----------
        next_random_number
            Zero-argument callable returning a uniform float in
            ``[0, 1)``. Draws one value to perform the weighted
            selection.

        Returns
        -------
        Move
            The selected move.

        Notes
        -----
        Linear scan; the move count is small (typically 2-5) and the
        extra clarity beats bisect-on-cumulative-weights at this size.
        """
        threshold = next_random_number() * self._total_weight
        acc = 0.0
        for move, weight in zip(
            self._moves, self._move_weights, strict=True
        ):
            acc += weight
            if threshold < acc:
                return move
        # Floating-point edge case: threshold slipped past the final
        # cumulative sum. Fall through to the last move.
        return self._moves[-1]

    def record_accept(self, name: str) -> None:
        """Increment the accept counter for *name*."""
        self._accept_counts[name] += 1

    def record_reject(self, name: str) -> None:
        """Increment the reject counter for *name*."""
        self._reject_counts[name] += 1

    def record_null(self, name: str) -> None:
        """Increment the null-proposal counter for *name*."""
        self._null_counts[name] += 1

    def acceptance_rates(self) -> dict[str, MoveStats]:
        """Return per-move acceptance statistics, keyed by move name."""
        return {
            move.name: MoveStats(
                accepted=self._accept_counts[move.name],
                rejected=self._reject_counts[move.name],
                null_proposed=self._null_counts[move.name],
            )
            for move in self._moves
        }

    def get_interval_data(self) -> dict[str, float]:
        """Return per-interval acceptance and null rates for all moves.

        Computes rates over the trials since the previous call (or
        since construction / ``reset``), then updates the snapshot
        baseline. Callers (``_get_ensemble_data`` in ensemble adapters)
        do not manage snapshots.

        Raises
        ------
        RuntimeError
            If any per-interval delta is negative — indicates a
            snapshot counter was not cleared alongside its cumulative
            counter (e.g. a ``reset`` that forgot one counter).
        """
        data: dict[str, float] = {}
        for move in self._moves:
            name = move.name
            cum_acc = self._accept_counts[name]
            cum_rej = self._reject_counts[name]
            cum_null = self._null_counts[name]
            interval_acc = cum_acc - self._last_recorded_accept[name]
            interval_rej = cum_rej - self._last_recorded_reject[name]
            interval_null = cum_null - self._last_recorded_null[name]
            if interval_acc < 0 or interval_rej < 0 or interval_null < 0:
                raise RuntimeError(
                    f"Per-interval counter went negative for move "
                    f"{name!r}: accepted={interval_acc}, "
                    f"rejected={interval_rej}, null={interval_null}. "
                    "This indicates a snapshot counter was not cleared "
                    "alongside its cumulative counter."
                )
            interval_proposed = interval_acc + interval_rej + interval_null
            if interval_proposed > 0:
                data[f"{name}_acceptance_rate"] = (
                    interval_acc / interval_proposed
                )
                data[f"{name}_null_rate"] = interval_null / interval_proposed
            else:
                data[f"{name}_acceptance_rate"] = 0.0
                data[f"{name}_null_rate"] = 0.0
            self._last_recorded_accept[name] = cum_acc
            self._last_recorded_reject[name] = cum_rej
            self._last_recorded_null[name] = cum_null
        return data

    def reset(self) -> None:
        """Zero all lifetime counters and snapshot baselines."""
        self._accept_counts.clear()
        self._reject_counts.clear()
        self._null_counts.clear()
        self._last_recorded_accept.clear()
        self._last_recorded_reject.clear()
        self._last_recorded_null.clear()


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
        self._dispatcher = MoveDispatcher(moves)

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

    @property
    def moves(self) -> list[Move]:
        """Copy of the moves registered with the ensemble."""
        return self._dispatcher.moves

    @property
    def move_weights(self) -> list[float]:
        """The (un-normalised) weights for each registered move."""
        return self._dispatcher.move_weights

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
        move = self._dispatcher.choose(self._next_random_number)
        proposal = move.propose(self.configuration, self._next_random_number)
        if proposal is None:
            self._dispatcher.record_null(move.name)
            return 0
        sites, species = proposal
        potential_diff = self._get_property_change(sites, species)
        if self._acceptance_condition(potential_diff):
            self.update_occupations(sites, species)
            self._dispatcher.record_accept(move.name)
            return 1
        self._dispatcher.record_reject(move.name)
        return 0

    def acceptance_rates(self) -> dict[str, MoveStats]:
        """Return per-move acceptance statistics, keyed by move name."""
        return self._dispatcher.acceptance_rates()

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
        data.update(self._dispatcher.get_interval_data())
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
        self._dispatcher.reset()


class CustomWangLandauEnsemble(WangLandauEnsemble):  # type: ignore[misc]
    """Wang-Landau ensemble parameterised by an arbitrary list of moves.

    Drop-in replacement for :class:`mchammer.ensembles.WangLandauEnsemble`
    that overrides :meth:`_do_trial_step` to draw a move from a
    user-supplied weighted list. Same composition pattern as
    :class:`CustomCanonicalEnsemble`: holds a :class:`MoveDispatcher`
    instance and delegates move selection and per-move bookkeeping to it.

    In addition to the accept/reject/null counters shared with the
    canonical adapter, this class classifies rejections as
    *window-rejected* (the proposed energy fell outside the energy
    window) or *WL-rejected* (in-window but rejected by the WL
    entropy criterion). Classification is only performed once the
    walker has reached the energy window; pre-window search-phase
    rejections are counted in the aggregate reject counter but not
    broken down further.

    Parameters
    ----------
    structure
        Atomic configuration; defines the initial occupation vector.
    calculator
        Energy calculator (typically a
        :class:`ClusterExpansionCalculator`).
    energy_spacing
        Bin size of the energy grid for the microcanonical entropy.
    moves
        List of ``(move, weight)`` tuples. Weights need not sum to one;
        they are used as relative weights.
    energy_limit_left
        Lower bound of the energy window (``None`` for unbounded).
    energy_limit_right
        Upper bound of the energy window (``None`` for unbounded).
    fill_factor_limit
        Terminate when the fill factor falls below this value.
    flatness_check_interval
        MC steps between flatness checks. Default: 1000 sweeps.
    flatness_limit
        Histogram flatness threshold.
    window_search_penalty
        Penalty strength for the pre-window distance heuristic.
    user_tag
        Human-readable tag.
    dc_filename
        Path for the data container file.
    data_container
        Existing data container to resume from.
    random_seed
        Seed for the Monte Carlo RNG.
    data_container_write_period
        Seconds between data container writes to disk.
    ensemble_data_write_interval
        MC steps between data container row writes.
    trajectory_write_interval
        MC steps between trajectory snapshots.
    schedule
        Fill-factor update schedule: ``'halving'`` or ``'1_over_t'``.

    Notes
    -----
    The ``trial_move`` and ``sublattice_probabilities`` arguments of
    ``WangLandauEnsemble`` are not exposed. Sublattice selection is the
    responsibility of each :class:`Move`, and ``self.do_move`` is never
    called because ``_do_trial_step`` is overridden.
    """

    def __init__(
        self,
        structure: Atoms,
        calculator: BaseCalculator,
        energy_spacing: float,
        moves: list[tuple[Move, float]],
        energy_limit_left: float | None = None,
        energy_limit_right: float | None = None,
        fill_factor_limit: float = 1e-6,
        flatness_check_interval: int | None = None,
        flatness_limit: float = 0.8,
        window_search_penalty: float = 2.0,
        user_tag: str | None = None,
        dc_filename: str | None = None,
        data_container: str | None = None,
        random_seed: int | None = None,
        data_container_write_period: float = 600,
        ensemble_data_write_interval: int | None = None,
        trajectory_write_interval: int | None = None,
        schedule: str = "halving",
    ) -> None:
        self._dispatcher = MoveDispatcher(moves)
        self._window_reject_counts: Counter[str] = Counter()
        self._wl_reject_counts: Counter[str] = Counter()
        self._last_recorded_window_reject: Counter[str] = Counter()
        self._last_recorded_wl_reject: Counter[str] = Counter()
        self._last_window_allowed: bool | None = None

        super().__init__(
            structure=structure,
            calculator=calculator,
            energy_spacing=energy_spacing,
            trial_move="swap",  # unused; our _do_trial_step overrides
            energy_limit_left=energy_limit_left,
            energy_limit_right=energy_limit_right,
            fill_factor_limit=fill_factor_limit,
            flatness_check_interval=flatness_check_interval,
            flatness_limit=flatness_limit,
            window_search_penalty=window_search_penalty,
            user_tag=user_tag,
            dc_filename=dc_filename,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval,
            schedule=schedule,
        )

    @property
    def moves(self) -> list[Move]:
        """Copy of the moves registered with the ensemble."""
        return self._dispatcher.moves

    @property
    def move_weights(self) -> list[float]:
        """The (un-normalised) weights for each registered move."""
        return self._dispatcher.move_weights

    def _allow_move(self, bin_cur: int | None, bin_new: int) -> bool:
        """Capture the window decision for rejection classification.

        ``WangLandauEnsemble._acceptance_condition`` calls this to gate
        histogram and entropy accumulation — it returns ``False`` when a
        proposed move would take the walker outside the defined energy
        window, blocking that move. The override records the return
        value so that ``_do_trial_step`` can classify a rejection as
        window-blocked or WL-rejected after
        ``_acceptance_condition`` returns.

        The classification relies on ``_acceptance_condition`` calling
        ``_allow_move`` exactly once per invocation. This holds for the
        current ``WangLandauEnsemble`` implementation, where
        ``_allow_move`` is the documented subclass hook for
        window-membership decisions.
        """
        result = super()._allow_move(bin_cur, bin_new)
        self._last_window_allowed = result
        return result

    def _do_trial_step(self) -> int:
        """Carry out one Wang-Landau trial step with custom moves.

        Same dispatch skeleton as :class:`CustomCanonicalEnsemble`:
        picks a move by weight, asks it for a proposal, evaluates the
        energy change, and applies the WL acceptance condition.

        Rejection classification (window vs WL) is gated on
        ``self._reached_energy_window``, read after
        ``_acceptance_condition`` returns so it reflects any transition
        that occurred inside it. ``_last_window_allowed`` is set by the
        ``_allow_move`` override on every call, so it always holds the
        correct window decision for the rejected proposal when the gate
        fires.
        """
        move = self._dispatcher.choose(self._next_random_number)
        proposal = move.propose(self.configuration, self._next_random_number)
        if proposal is None:
            self._dispatcher.record_null(move.name)
            return 0
        sites, species = proposal
        potential_diff = self._get_property_change(sites, species)
        self._last_window_allowed = None
        if self._acceptance_condition(potential_diff):
            self.update_occupations(sites, species)
            self._dispatcher.record_accept(move.name)
            return 1
        self._dispatcher.record_reject(move.name)
        if self._reached_energy_window:
            if self._last_window_allowed is False:
                self._window_reject_counts[move.name] += 1
            elif self._last_window_allowed is True:
                self._wl_reject_counts[move.name] += 1
        return 0

    def acceptance_rates(self) -> dict[str, MoveStats]:
        """Return per-move acceptance statistics, keyed by move name."""
        return self._dispatcher.acceptance_rates()

    def rejection_breakdown(self) -> dict[str, tuple[int, int]]:
        """Return per-move (window_rejected, wl_rejected) counts.

        A rejected trial is counted in exactly one of the two
        categories. The two counts together cover only in-window
        rejections; their sum may be less than ``MoveStats.rejected``
        because pre-window search-phase rejections are not classified.
        """
        return {
            move.name: (
                self._window_reject_counts[move.name],
                self._wl_reject_counts[move.name],
            )
            for move in self._dispatcher.moves
        }

    def _get_ensemble_data(self) -> dict[str, float]:
        """Extend the standard ensemble-data dict with per-move stats.

        Adds four keys per registered move:

        ``<move>_acceptance_rate``, ``<move>_null_rate``
            Per-interval rates from the dispatcher; denominator is
            total proposals (accepted + rejected + null) in the
            interval.
        ``<move>_window_rejection_rate``, ``<move>_wl_rejection_rate``
            Per-interval rates among *classified* rejections only
            (window-blocked + WL-rejected in the interval). A
            rejection is classified only once the walker has reached
            the energy window, so these rates do not sum with
            ``<move>_acceptance_rate`` and ``<move>_null_rate`` to
            any fixed value.
        """
        data = super()._get_ensemble_data()
        data.update(self._dispatcher.get_interval_data())
        for move in self._dispatcher.moves:
            name = move.name
            cum_window = self._window_reject_counts[name]
            cum_wl = self._wl_reject_counts[name]
            interval_window = (
                cum_window - self._last_recorded_window_reject[name]
            )
            interval_wl = cum_wl - self._last_recorded_wl_reject[name]
            interval_classified = interval_window + interval_wl
            if interval_classified > 0:
                data[f"{name}_window_rejection_rate"] = (
                    interval_window / interval_classified
                )
                data[f"{name}_wl_rejection_rate"] = (
                    interval_wl / interval_classified
                )
            else:
                data[f"{name}_window_rejection_rate"] = 0.0
                data[f"{name}_wl_rejection_rate"] = 0.0
            self._last_recorded_window_reject[name] = cum_window
            self._last_recorded_wl_reject[name] = cum_wl
        return data

    def reset_acceptance_counts(self) -> None:
        """Zero all per-move counters including WL-specific ones."""
        self._dispatcher.reset()
        self._window_reject_counts.clear()
        self._wl_reject_counts.clear()
        self._last_recorded_window_reject.clear()
        self._last_recorded_wl_reject.clear()

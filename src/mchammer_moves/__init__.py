"""Custom Monte Carlo moves for icet/mchammer.

Provides a small framework for plugging user-defined trial moves into
mchammer sampling. The :class:`Move` base class defines a
sampler-agnostic proposal contract; ensemble adapters consume moves
and handle acceptance, bookkeeping, and data-container integration:

* :class:`Move` — abstract base class for trial moves.
* :class:`PairSwap` — standard two-site canonical swap on a sublattice.
* :class:`MultiPairSwap` — ``k`` site-disjoint pair swaps applied as
  one atomic proposal, for larger jumps in configuration space than a
  single pair swap can provide.
* :class:`CyclicShift` — single-step cyclic shift of species along
  one of a user-supplied set of index cycles. Useful for row /
  ring translations on lattice sublattices with chain-like or
  ring-like topology when standard single-site swaps are
  kinetically blocked.
* :class:`CyclicReflection` — long-range reflection of the species
  pattern along an index cycle around a randomly-chosen pivot;
  complement to ``CyclicShift``'s nearest-neighbour shifts.
* :class:`IndexSetSwap` — generic group-permutation primitive that
  swaps occupations between two equal-length index sets drawn
  uniformly from a user-supplied list.
* :class:`SitePermutation` — applies a caller-supplied permutation of
  site occupations (reflections, point inversion, rotations) drawn
  uniformly from a list, applying each operation or its inverse with
  equal probability so detailed balance holds for any permutation.
* :class:`MoveDispatcher` — weighted move selection and per-move
  bookkeeping, used internally by ensemble adapters.
* :class:`CustomCanonicalEnsemble` — drop-in replacement for
  :class:`mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.
* :class:`CustomWangLandauEnsemble` — drop-in replacement for
  :class:`mchammer.ensembles.WangLandauEnsemble` with the same
  weighted-move dispatch, plus window-vs-WL rejection classification.
"""

from mchammer_moves.ensemble import (
    CustomCanonicalEnsemble,
    CustomWangLandauEnsemble,
    MoveDispatcher,
    MoveStats,
)
from mchammer_moves.moves.base import Move
from mchammer_moves.moves.cyclic_reflection import CyclicReflection
from mchammer_moves.moves.cyclic_shift import CyclicShift
from mchammer_moves.moves.index_set_swap import IndexSetSwap
from mchammer_moves.moves.multi_pair_swap import MultiPairSwap
from mchammer_moves.moves.pair_swap import PairSwap
from mchammer_moves.moves.site_permutation import SitePermutation

__all__ = [
    "CustomCanonicalEnsemble",
    "CustomWangLandauEnsemble",
    "CyclicReflection",
    "CyclicShift",
    "IndexSetSwap",
    "Move",
    "MoveDispatcher",
    "MoveStats",
    "MultiPairSwap",
    "PairSwap",
    "SitePermutation",
]

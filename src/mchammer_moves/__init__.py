"""Custom Monte Carlo moves for icet/mchammer.

Provides a small framework for plugging user-defined trial moves into
mchammer canonical sampling. The package exposes:

* :class:`Move` — abstract base class for trial moves.
* :class:`PairSwap` — standard two-site canonical swap on a sublattice.
* :class:`CyclicShift` — single-step cyclic shift of species along
  one of a user-supplied set of index cycles. Useful for row /
  ring translations on lattice sublattices with chain-like or
  ring-like topology when standard single-site swaps are
  kinetically blocked.
* :class:`CustomCanonicalEnsemble` — drop-in replacement for
  :class:`mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.
"""

from mchammer_moves.ensemble import CustomCanonicalEnsemble, MoveStats
from mchammer_moves.moves.base import Move
from mchammer_moves.moves.cyclic_shift import CyclicShift
from mchammer_moves.moves.multi_pair_swap import MultiPairSwap
from mchammer_moves.moves.pair_swap import PairSwap

__all__ = [
    "CustomCanonicalEnsemble",
    "CyclicShift",
    "Move",
    "MoveStats",
    "MultiPairSwap",
    "PairSwap",
]

"""Custom Monte Carlo moves for icet/mchammer.

Provides a small framework for plugging user-defined trial moves into
mchammer canonical sampling. The package exposes:

* :class:`Move` — abstract base class for trial moves.
* :class:`PairSwap` — standard two-site canonical swap on a sublattice.
* :class:`SlideRow` — translation of a species pattern along a chain
  with periodic boundaries within the chain.
* :class:`CustomCanonicalEnsemble` — drop-in replacement for
  :class:`mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.
"""

from mchammer_moves.ensemble import CustomCanonicalEnsemble
from mchammer_moves.moves.base import Move
from mchammer_moves.moves.pair_swap import PairSwap
from mchammer_moves.moves.slide_row import SlideRow

__all__ = [
    "Move",
    "PairSwap",
    "SlideRow",
    "CustomCanonicalEnsemble",
]

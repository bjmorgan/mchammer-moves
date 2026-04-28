"""Trial-move classes for use with :class:`CustomCanonicalEnsemble`."""

from mchammer_moves.moves.base import Move
from mchammer_moves.moves.cyclic_shift import CyclicShift
from mchammer_moves.moves.pair_swap import PairSwap

__all__ = ["CyclicShift", "Move", "PairSwap"]

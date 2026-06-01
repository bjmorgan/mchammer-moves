# mchammer-moves

Custom Monte Carlo trial moves for [icet/mchammer](https://icet.materialsmodeling.org/).
The `Move` base class defines a sampler-agnostic proposal contract; ensemble
adapters consume moves and handle acceptance, bookkeeping, and data-container
integration for canonical and Wang-Landau sampling, without modification of the
mchammer source or downstream wrappers such as `mchammer-pt`.

The package provides:

- a `Move` abstract base class for user-defined trial moves;
- five built-in moves:
  - `PairSwap` — the standard canonical two-site swap;
  - `MultiPairSwap` — `k` site-disjoint pair swaps applied as one
    atomic proposal; useful when single-pair swaps are kinetically
    blocked between adjacent minima in deep basins;
  - `CyclicShift` — single-step shift of the species pattern along a
    user-supplied index cycle, with periodic boundaries within the
    cycle; useful for row or ring translations on chain-like or
    ring-like sublattices;
  - `CyclicReflection` — long-range reflection of the species pattern
    along an index cycle around a randomly-chosen pivot; complements
    `CyclicShift`'s nearest-neighbour shifts by enabling species to
    hop across a chain in a single accepted move;
  - `IndexSetSwap` — swaps occupations between two equal-length index
    sets drawn uniformly from a user-supplied list of groups; a
    generic primitive for chain-, motif-, or layer-swap moves;
  - `SitePermutation` — applies a caller-supplied permutation of site
    occupations, drawn uniformly from a list of operations, with an
    unconditional forward/inverse direction draw; covers reflections
    (e.g. across a `<100>` plane), point inversion, and proper or
    improper rotations of any order;
- `CustomCanonicalEnsemble`, a drop-in replacement for
  `mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance;
- `CustomWangLandauEnsemble`, a drop-in replacement for
  `mchammer.ensembles.WangLandauEnsemble` with the same weighted-move
  dispatch, plus per-move window-vs-WL rejection classification;
- `MoveDispatcher`, the shared weighted-selection and per-move
  bookkeeping engine used by both ensemble adapters.

Installation (editable):

```bash
pip install -e .
```

To use the parallel-tempering integration, install the `pt` extra
(pulls `mchammer-pt` v0.2.0 from GitHub until it is on PyPI):

```bash
pip install -e ".[pt]"
```

## Quick start

`structure`, `ce`, and `cycles` below are placeholders for your atoms
object, cluster expansion, and chain definitions respectively; the
package contains no system-specific geometry, so you supply them
yourself.

```python
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap

calc = ClusterExpansionCalculator(structure, ce)

ensemble = CustomCanonicalEnsemble(
    structure=structure,
    calculator=calc,
    temperature=600.0,
    moves=[
        (PairSwap(sublattice_index=0), 1.0),
        (CyclicShift(cycles=cycles), 0.05),
    ],
)
ensemble.run(10_000)

print(ensemble.acceptance_rates())
```

## Use with `mchammer-pt`

`mchammer-pt` (v0.2+) accepts a custom ensemble class via its native
`ensemble_cls=` parameter, with constructor arguments forwarded via
`ensemble_kwargs=`:

```python
from mchammer_pt import CanonicalParallelTempering
from mchammer_moves import CustomCanonicalEnsemble, CyclicShift, PairSwap

with CanonicalParallelTempering.process_pool(
    cluster_expansion=ce,
    atoms=initial_structure,
    temperatures=temperatures,
    block_size=block_size,
    random_seed=42,
    ensemble_cls=CustomCanonicalEnsemble,
    ensemble_kwargs={
        "moves": [
            (PairSwap(sublattice_index=anion_sl), 1.0),
            (CyclicShift(cycles=cycles), 0.05),
        ],
    },
) as pt:
    history = pt.run(n_cycles=N_CYCLES)
```

Per-move acceptance and null-proposal rates are recorded into each
replica's `mchammer.BaseDataContainer` at every
`ensemble_data_write_interval` as `<move>_acceptance_rate` and
`<move>_null_rate` columns, so they survive the `ProcessPool`
boundary and are recoverable from the HDF5 bundle written by
`mchammer-pt` without observer forwarding. The two are tracked
separately: a move that returns `None` (e.g. a `PairSwap` on a
single-species sublattice, a `MultiPairSwap` on a sublattice with
fewer than `k` of one species, an `IndexSetSwap` whose drawn pair
already holds identical occupations) increments the null counter rather
than the rejection counter, so `null_rate` distinguishes a
structurally-infeasible move (`null_rate ≈ 1`) from a
low-temperature trapped chain (`acceptance_rate ≈ 0`,
`null_rate ≈ 0`).

For multiprocess runs, `CustomCanonicalEnsemble` and every `Move`
subclass must be importable by fully qualified name in spawn workers
(i.e. defined in `.py` module files, not in `__main__` or notebook
cells). `mchammer-pt`'s `ProcessPool` rejects interactive-`__main__`
and function-local classes up-front.

## Use with Wang-Landau

`CustomWangLandauEnsemble` accepts the same `moves` list as
`CustomCanonicalEnsemble` and forwards all other parameters to
`WangLandauEnsemble`:

```python
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import CustomWangLandauEnsemble, PairSwap

calc = ClusterExpansionCalculator(structure, ce)

mc = CustomWangLandauEnsemble(
    structure=structure,
    calculator=calc,
    energy_spacing=0.1,
    moves=[
        (PairSwap(sublattice_index=0), 1.0),
    ],
    energy_limit_left=-100.0,
    energy_limit_right=-90.0,
    schedule="1_over_t",
)
mc.run(1_000_000)

print(mc.acceptance_rates())
print(mc.rejection_breakdown())
```

Per-move acceptance, null, window-rejection, and WL-rejection rates are
recorded into the `WangLandauDataContainer` at every
`ensemble_data_write_interval` as `<move>_acceptance_rate`,
`<move>_null_rate`, `<move>_window_rejection_rate`, and
`<move>_wl_rejection_rate` columns. The `rejection_breakdown()` method
provides cumulative window-vs-WL rejection counts for interactive use.

`<move>_acceptance_rate` and `<move>_null_rate` use total proposals
(accepted + rejected + null) as the denominator. `<move>_window_rejection_rate`
and `<move>_wl_rejection_rate` use classified in-window rejections as the
denominator — they do not share a denominator with the first two columns and
do not sum with them to any fixed value.

Rejection classification is only performed once the walker has reached
the energy window. Pre-window search-phase rejections are counted in the
aggregate `MoveStats.rejected` counter but not broken down further.

## Constructing cycles for `CyclicShift`

`CyclicShift` expects a list of *cycles*, where each cycle is a list of
site indices in the order along which species are to be shifted. Cycles
may have any length and may differ in length from one another; the move
treats each cycle as periodic in itself (the last site wraps to the
first). The supplied indices are opaque labels — there is no requirement
that they correspond to physically collinear sites.

The package contains no system-specific geometry. Cycle construction is
the caller's responsibility. The recipe for a typical anion-ordered
ReO3-type supercell, where each cycle corresponds to a one-dimensional
chain of anion sites, is:

1. Identify a single-axis chain of anion sites — for example, all sites of
   the form `(i, 0, 0), (i, 0, 1), …, (i, 0, N-1)` along the *z* axis at
   `(x=i, y=0)` — and list their flat site indices in geometric order.
2. Repeat for each starting `(x, y)` to obtain the full set of *z*-cycles.
3. Repeat the procedure for *x*-cycles and *y*-cycles if your problem has
   chain ordering along multiple axes.
4. Pass the combined list to `CyclicShift(cycles=...)`.

For NbO2F at 6×6×6, the relevant cycles are anion chains along each cubic
axis (108 cycles per axis, 324 cycles total). See the integration script
in the `data_NbO2F` project for a concrete construction.

## Detailed balance

All built-in moves have proposal probabilities that depend only on lattice
geometry and composition, not on the current configuration:

- `PairSwap`: at fixed canonical composition, the number of distinct-species
  pairs on a sublattice is composition-invariant, so the probability of
  selecting any specific pair is symmetric in the forward and reverse
  directions.
- `MultiPairSwap`: each pair is drawn by picking site 1 uniformly from
  the non-used sublattice sites and site 2 uniformly from the non-used
  sites of differing species. Summed over the `k!` orderings of the
  same site-disjoint pair-set, the forward and reverse proposal
  probabilities are equal: composition is invariant under any valid
  swap, and the dependence on already-used sites cancels by symmetry
  between the two directions.
- `CyclicShift`: a cycle and direction are chosen uniformly at random.
  The reverse of a `+1` shift along cycle *c* is a `-1` shift along the
  same cycle, with the same selection probability.
- `CyclicReflection`: a cycle and integer pivot are chosen uniformly
  at random. Cyclic reflection is an involution, so the reverse of a
  reflection along `(c, p)` is the same reflection along `(c, p)`,
  with the same selection probability.
- `IndexSetSwap`: an unordered pair of index sets is drawn uniformly
  from `C(N, 2)` distinct pairs. Selection probability depends only
  on the fixed list of index sets, not on the configuration, so
  `P(A → B) = P(B → A)` directly. The optional
  `require_matching_composition` filter (off by default) does not
  break this: swapping any pair only exchanges the two groups'
  contents, so the multiset of compositions held across the groups
  is invariant under the move, and a pair filtered out in one
  direction is also filtered out in the other.
- `SitePermutation`: an operation is drawn uniformly from the fixed
  list, then applied forward or inverted, each with probability one
  half. The applied-permutation multiset is closed under inversion with
  equal weights, so `P(A → B) = P(B → A)` for any permutation.

Standard Metropolis acceptance therefore satisfies detailed balance for any
weighted combination of these moves. A symmetry test that empirically
verifies this property is provided in the test suite for each move and
should be the first thing you run when adding a new move.

For Wang-Landau sampling, `CustomWangLandauEnsemble` replaces the
Metropolis criterion with the WL entropy-based acceptance condition
inherited from `WangLandauEnsemble`. The symmetric-proposal property
of each move still holds, so the WL algorithm's convergence guarantees
are preserved for any weighted combination of the built-in moves.

## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

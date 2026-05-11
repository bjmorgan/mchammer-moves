# mchammer-moves

Custom Monte Carlo trial moves for [icet/mchammer](https://icet.materialsmodeling.org/),
designed to slot into existing canonical sampling without modification of the
mchammer source or downstream wrappers such as `mchammer-pt`.

The package provides:

- a `Move` abstract base class for user-defined trial moves;
- four built-in moves:
  - `PairSwap` — the standard canonical two-site swap;
  - `MultiPairSwap` — `k` site-disjoint pair swaps applied as one
    atomic proposal; useful when single-pair swaps are kinetically
    blocked between adjacent minima in deep basins;
  - `CyclicShift` — single-step shift of the species pattern along a
    user-supplied index cycle, with periodic boundaries within the
    cycle; useful for row or ring translations on chain-like or
    ring-like sublattices;
  - `IndexSetSwap` — swaps occupations between two equal-length index
    sets drawn uniformly from a user-supplied list of groups; a
    generic primitive for chain-, motif-, or layer-swap moves;
- `CustomCanonicalEnsemble`, a drop-in replacement for
  `mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.

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

Per-move acceptance is recorded into each replica's
`mchammer.BaseDataContainer` at every `ensemble_data_write_interval`,
so it survives the `ProcessPool` boundary and is recoverable from the
HDF5 bundle written by `mchammer-pt` without observer forwarding.

For multiprocess runs, `CustomCanonicalEnsemble` and every `Move`
subclass must be importable by fully qualified name in spawn workers
(i.e. defined in `.py` module files, not in `__main__` or notebook
cells). `mchammer-pt`'s `ProcessPool` rejects interactive-`__main__`
and function-local classes up-front.

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
- `IndexSetSwap`: an unordered pair of index sets is drawn uniformly
  from `C(N, 2)` distinct pairs. Swapping the two sets exchanges their
  entire contents, so each set's composition is preserved; any pair
  valid in the forward direction is therefore also valid in the
  reverse direction with the same selection probability.

Standard Metropolis acceptance therefore satisfies detailed balance for any
weighted combination of these moves. A symmetry test that empirically
verifies this property is provided in the test suite for each move and
should be the first thing you run when adding a new move.

## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

# mchammer_moves

Custom Monte Carlo trial moves for [icet/mchammer](https://icet.materialsmodeling.org/),
designed to slot into existing canonical sampling without modification of the
mchammer source or downstream wrappers such as `mchammer_pt`.

The package provides:

- a `Move` abstract base class for user-defined trial moves;
- two built-in moves: `PairSwap` (the standard canonical two-site swap) and
  `SlideRow` (translation of a species pattern along a chain with periodic
  boundaries within the chain);
- `CustomCanonicalEnsemble`, a drop-in replacement for
  `mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.

Installation (editable):

```bash
pip install -e .
```

## Quick start

```python
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import CustomCanonicalEnsemble, PairSwap, SlideRow

calc = ClusterExpansionCalculator(structure, ce)

ensemble = CustomCanonicalEnsemble(
    structure=structure,
    calculator=calc,
    temperature=600.0,
    moves=[
        (PairSwap(sublattice_index=0), 1.0),
        (SlideRow(rows=rows), 0.05),
    ],
)
ensemble.run(10_000)

print(ensemble.acceptance_rates())
```

## Use with `mchammer_pt`

`mchammer_pt.replica.Replica` constructs an `mchammer.CanonicalEnsemble`
internally, so `CustomCanonicalEnsemble` cannot be passed directly to
`CanonicalParallelTempering` via its `atoms`/`temperatures` route.
However, `CanonicalParallelTempering` accepts an arbitrary `pool=`
argument satisfying the `ReplicaPool` protocol, which is the supported
extension point for custom replica state.

The `mchammer_moves.pt_adapter` module supplies the glue:

```python
from mchammer_pt import CanonicalParallelTempering
from mchammer_moves import PairSwap, SlideRow
from mchammer_moves.pt_adapter import make_serial_pool

pool = make_serial_pool(
    cluster_expansion=ce,
    atoms=initial_structure,
    temperatures=temperatures,
    moves=[
        (PairSwap(sublattice_index=anion_sl), 1.0),
        (SlideRow(rows=rows), 0.05),
    ],
    random_seed=42,
)
pt = CanonicalParallelTempering(
    cluster_expansion=ce,
    atoms=initial_structure,
    temperatures=temperatures,
    block_size=block_size,
    random_seed=42,
    pool=pool,
)
history = pt.run(n_cycles=N_CYCLES)

# Per-move acceptance per replica:
for r, replica in enumerate(pool._replicas):
    print(replica.temperature, replica.ensemble.acceptance_rates())
```

`make_serial_pool` mirrors the per-replica seeding used by
`CanonicalParallelTempering` itself, so the resulting pool behaves
identically to the wrapper's default `SerialPool` apart from the choice
of ensemble class.

## Constructing rows for `SlideRow`

`SlideRow` expects a list of *rows*, where each row is a list of site indices
in order along a one-dimensional chain. The row may have any length; the
move treats the chain as periodic in itself (the last site wraps to the
first).

The package contains no system-specific geometry. Row construction is the
caller's responsibility. The recipe for a typical anion-ordered ReO3-type
supercell is:

1. Identify a single-axis chain of anion sites — for example, all sites of
   the form `(i, 0, 0), (i, 0, 1), …, (i, 0, N-1)` along the *z* axis at
   `(x=i, y=0)` — and list their flat site indices in geometric order.
2. Repeat for each starting `(x, y)` to obtain the full set of *z*-rows.
3. Repeat the procedure for *x*-rows and *y*-rows if your problem has chain
   ordering along multiple axes.
4. Pass the combined list to `SlideRow(rows=...)`.

For NbO2F at 6×6×6, the relevant rows are anion chains along each cubic axis
(108 rows per axis, 324 rows total). See the integration script in the
`data_NbO2F` project for a concrete construction.

## Detailed balance

Both built-in moves have proposal probabilities that depend only on lattice
geometry, not on the current configuration:

- `PairSwap`: at fixed canonical composition, the number of distinct-species
  pairs on a sublattice is composition-invariant, so the probability of
  selecting any specific pair is symmetric in the forward and reverse
  directions.
- `SlideRow`: a row and direction are chosen uniformly at random. The reverse
  of a `+1` slide along row *r* is a `-1` slide along the same row, with the
  same selection probability.

Standard Metropolis acceptance therefore satisfies detailed balance for any
weighted combination of these moves. A symmetry test that empirically
verifies this property is provided in the test suite and should be the first
thing you run when adding a new move.

## Running tests

```bash
pip install -e ".[test]"
pytest -v
```

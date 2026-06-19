# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- `schedule` keyword argument from `CustomWangLandauEnsemble`. It mirrored the
  patched icet fork's Wang-Landau constructor (the same fork-only family as the
  `switch_mode` argument removed in 0.4.0) rather than adding behaviour of its
  own: the class has no 1/t engine and inherits `_update_entropy` from the base.
  `CustomWangLandauEnsemble` is now a drop-in replacement for the stock
  `WangLandauEnsemble` (halving Wang-Landau with custom moves). The 1/t schedule
  with custom moves remains available through
  `mchammer_pt.contrib.CoordinatedCustomWangLandauEnsemble`.

### Changed

- Relaxed the `icet` dependency from the patched bjmorgan fork to a plain
  `icet>=3.2`, now that the Wang-Landau adapter uses only the stock Wang-Landau
  API.

## [0.4.1] - 2026-06-01

### Changed

- Version bump only; no functional changes since v0.4.0. The `0.4.0`
  version string was set before `SitePermutation` was added and was
  installed from an untagged `main` in some environments, so a package
  reporting `0.4.0` may predate `SitePermutation`. Releasing the same
  code as `0.4.1` lets `pip` detect the upgrade.

## [0.4.0] - 2026-06-01

### Added

- `SitePermutation` move: applies a caller-supplied permutation of site
  occupations (each entry `i: j` means site `i` takes the occupation of
  site `j`), drawn uniformly from a configured list of operations. It
  covers reflections (for example across a `<100>` plane), point
  inversion, and proper or improper rotations of any order. An
  unconditional forward/inverse direction draw preserves detailed
  balance for non-involutions as well as involutions.
- `MoveDispatcher`: weighted move selection and per-move bookkeeping,
  shared by the ensemble adapters.
- `CustomWangLandauEnsemble`: a drop-in replacement for
  `mchammer.ensembles.WangLandauEnsemble` using the same weighted-move
  dispatch, with window-versus-Wang-Landau rejection classification.

### Changed

- Dropped the obsolete `switch_mode` keyword arguments from the
  Wang-Landau integration to track the upstream `mchammer` API.

## [0.3.0] - 2026-05-13

### Added

- `CyclicReflection` move: long-range reflection of the species pattern
  along an index cycle around a randomly-chosen pivot, complementing
  `CyclicShift`'s nearest-neighbour shifts.

## [0.2.0] - 2026-05-12

### Added

- Core framework: the `Move` abstract base class and
  `CustomCanonicalEnsemble`, a drop-in replacement for
  `mchammer.ensembles.CanonicalEnsemble` that draws moves from a
  user-supplied weighted list and tracks per-move acceptance.
- Built-in moves `PairSwap`, `MultiPairSwap`, `CyclicShift`, and
  `IndexSetSwap`, with per-move null-proposal tracking.
- Optional composition matching for `IndexSetSwap`, off by default.

### Changed

- Migrated to the `mchammer-pt` v0.2 `ensemble_cls=` API.

[0.4.1]: https://github.com/bjmorgan/mchammer-moves/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bjmorgan/mchammer-moves/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bjmorgan/mchammer-moves/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bjmorgan/mchammer-moves/releases/tag/v0.2.0

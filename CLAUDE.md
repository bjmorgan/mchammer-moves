# Claude Code Guidelines — mchammer-moves

Personal conventions for working on this repository. Local file; not committed.

## Development Process

- Pair programming approach: small incremental changes verified by targeted unit tests.
- Do not provide code unless specifically asked for it.
- Only provide the parts of code that are needed rather than entire updated files.
- After any refactoring, review the unit tests. Would you design them differently from scratch? Can they be modified to better match the refactored code?
- Design specs and plans are local dev notes. The `docs/superpowers/` directory is gitignored; do not commit its contents.
- Always work in a new branch. Ask the user what branch to base off if you are unsure.
- **Mise-en-place:** Fix issues when you find them. Do not defer fixes to future PRs or issues — this accumulates technical debt. If you encounter a bug, code smell, or missing validation during development, address it now.
- **Prefer the better solution over the quicker one.** When addressing review feedback or fixing a problem, consider whether the fix is at the right level. A `try/finally` might fix the immediate symptom, but a context manager might be the right abstraction. A validation check might catch bad input, but a type that makes bad input unrepresentable is better. Always ask: does this fix the right problem at the right level, or would solving it differently produce cleaner, more idiomatic, more maintainable code?
- **Decide, do not defer.** For every reviewer finding, code smell, or TODO you notice, make an explicit call: fix it now (at the right level, not the quick-fix level), or decline it with a stated reason. "Approved with nits" is not a terminal state — it is a list of pending decisions. Postponing is the failure mode; declining for a real reason is fine.
- **Red-flag rationalisations.** The phrases below are examples of deferral patterns dressed as reasoning. Treat their *shape* as a signal, not the exact wording — anything that smells like this family is a flag to stop and reverse the default:
  - *Anything that invokes the cost of undoing* ("committed, leave it", "already done, move on", "too late to change now") — reverting on a feature branch is cheap, and commit history is ephemeral until the PR merges. The work required to undo something is not a reason to keep it wrong.
  - *Anything that treats a review verdict as permission to skip work* ("approved with nits, move on", "only nits left") — an approval with open findings is not a terminal state; it is a list of pending decisions. Fix each or decline each with a stated reason.
  - *Anything that uses "out of scope" to dismiss a finding without checking* ("pre-existing, not our problem", "unrelated to this PR") — before invoking scope, ask whether the fix is small and in code you are already touching. If yes, it is in scope.
  - *Anything that uses "harmless" as a reason to keep something* — harmless is not a reason. It is an observation that the cost of removing it is low, which is an argument for removing it, not keeping it. Decide on merit.
  - *Anything that substitutes a promise for an action* ("I'll remember next time", "I'll be more careful going forward") — future intentions do not fix present problems. State the operational rule now, or make the fix now.

## Testing

- Tests should be as simple as possible while testing the desired behaviour.
- Tests should be well isolated where possible.
- Tests should test the intended desired behaviour, not legacy behaviour that is targeted for removal.
- Detailed-balance / Boltzmann-sampling correctness should be pinned via the public utility `mchammer_pt.testing.assert_boltzmann_sampling` so the framework's analytic anchor and downstream consumers stay in sync.

## Reviewing

Reviewer findings (from review agents, human reviewers, or self-review passes) are *proposals*, not prescriptions. Every finding still gets a motivated disposition per "decide, don't defer" — but the motivation must be grounded in the library's actual users and contract, not in the shape of the finding.

- **Every disposition names a specific user or failure mode.** "Accept — this catches a missing-`tag` kwarg in NbO2F's slide-row run" is a motivated accept. "Decline — `bytearray` rows don't arise in any realistic calling pattern; a `TypeError` is already adequate" is a motivated decline. "Accept — small fix" and "Decline — minor nit" are not dispositions; they are deferrals wearing the uniform of the rule.

- **"Minor", "small", "cheap to fix", "belt and braces", "consistent with X" are not reasons.** They describe the *cost* of acting, not the *benefit*. A reason answers "what would break, and for whom, if we don't act?" or "what noise does acting add, and at whose expense?"

- **Decline whole classes of findings at once when the reason is shared.** If a batch of findings all assume adversarial or hostile callers, pin implementation choices rather than user-visible contracts, or ask for symmetric guards across functions whose inputs come from a single trusted producer — decline the class with one stated reason, rather than dispositioning each item independently as though they had separate merit.

- **Review rounds compound.** Each iteration asks "what might still be wrong?" and agents look harder each pass. When a round's findings are mostly defensive additions against implausible inputs, or tests that pin implementation choices rather than behaviour, the correct response is to stop reviewing, not to find motivated declines for a longer list. The signal to stop is: no new finding maps to a failure mode a real user of this library would hit.

- **Who uses this library, explicitly.** `mchammer_moves` is used by project members and students adding custom Monte Carlo moves to mchammer canonical sampling — primarily through `mchammer_pt`'s parallel-tempering driver. Currently consumed by the NbO2F project's slide-row run; designed to be reusable across future projects with custom-move requirements (cluster moves, constrained swaps, hybrids). It is not a public API with SLAs, not a framework with adversarial plugin authors, not a multiprocessing primitive with thread-safety requirements. Findings that presuppose any of those audiences are noise unless they also describe a failure mode the actual audience would hit.

## Git and PR Conventions

- Do not acknowledge Claude in commits, pull requests, etc.
- Do not include "Test plan" sections in pull request descriptions.
- Use British spellings in commit messages and pull request descriptions.
- Do not use Unicode characters (e.g., subscripts like H2O) in commits, PRs, or issues — use plain text.
- **PR descriptions describe what is landing, not the process that produced it.** Skip sections that restate CI signal or enumerate declined suggestions — test counts, mypy/ruff status, and declined reviewer items are all visible elsewhere (CI checks, review-comment threads, commit log) and clutter the description. If a declined item is important enough to record, it belongs in a commit message at the point of declining, or as a review-thread response, not in the PR body. The PR body should read as an as-built summary of the change, not as a fix-log or status report.

## Documentation

- **Docs describe the thing, not the conversation that produced them.** A reader coming cold to a docstring, README, or tutorial prose does not have the review conversation in which wording was negotiated. Text that reads as answering unasked questions leaks that conversation — phrases like "note that...", "it's worth noting...", "equivalently...", "this also..." — all signal that the prose is responding to something the cold reader never raised. Before committing doc prose, read it as a first-time reader: if a sentence reads as an answer, ask what the question is. If the question is one that only emerged in review, the sentence doesn't belong — either delete it, or edit the earlier prose so the question never arises.

## Coding Standards

- **Python 3.11+**
- **Google-style docstrings** for all public classes, methods, and functions.
- **Type hints** throughout, using modern syntax (`list[str]` not `List[str]`, `X | None` not `Optional[X]`).
- **British English** in docstrings and comments (e.g. "colour" not "color", "normalise" not "normalize", "centre" not "center").
- **Dataclasses** for simple data containers. Use `frozen=True` where immutability is appropriate. When a class needs validated properties, custom setters, or non-trivial `__init__` logic, use a plain class rather than fighting the dataclass machinery.
- **numpy** for coordinate / occupation maths where vectorised operations are possible.

## mchammer-moves-specific notes

### Purpose and users

`mchammer-moves` provides a small framework for plugging
user-defined trial moves into mchammer canonical sampling. The
package exposes:

- a `Move` abstract base class for user-defined trial moves;
- built-in concrete moves (`PairSwap`, `SlideRow`);
- `CustomCanonicalEnsemble`, a `CanonicalEnsemble` subclass that
  draws moves from a user-supplied weighted list and tracks per-move
  acceptance.

Primary audiences, in order: (1) project members and students adding
custom moves to `mchammer_pt` parallel-tempering runs (NbO2F
slide-row is the in-flight use case); (2) future projects with
custom-move requirements on different systems (oxynitrides,
multi-sublattice fluorides, etc.); (3) future-me on the same
problems.

### Design priorities

In this order, non-negotiable:

1. Clarity, maintainability, readability for project use.
2. Bounded, focused modules (under ~250 lines each, one
   responsibility per file). One move per file.
3. Testability of each unit in isolation.
4. Each `Move` subclass must carry an explicit detailed-balance
   argument in its docstring. Subclasses whose proposal probabilities
   depend on the current configuration are unsafe under standard
   Metropolis acceptance and must say so.

### mchammer conformity policy

- **Do** subclass `mchammer.ensembles.CanonicalEnsemble` for
  `CustomCanonicalEnsemble` so the public interface inherited from
  upstream is preserved exactly. External orchestrators (notably
  `mchammer_pt`) must be able to use this ensemble through any
  `CanonicalEnsemble` extension point without adaptation.
- **Do** reuse mchammer's machinery — `ConfigurationManager`,
  calculator, `_get_property_change`, `_acceptance_condition`,
  `update_occupations`. Do not duplicate logic that already lives
  upstream.
- **Do not** override mchammer methods that are not extension
  points (anything not on the documented subclass surface).

### mchammer-pt integration

- This package is consumed via `mchammer_pt`'s native
  `ensemble_cls=CustomCanonicalEnsemble, ensemble_kwargs={"moves": [...]}`
  surface (mchammer-pt v0.2+). No custom `Replica` wrapper is
  needed; the previous `pt_adapter` module is retired.
- For multiprocess parallel tempering: `ensemble_cls` and every
  `Move` subclass must be importable by fully qualified name in
  spawn workers (i.e. defined in module files, not in `__main__` or
  notebook cells). `mchammer_pt.ProcessPool` rejects
  interactive-`__main__` and function-local classes up-front.

### Scope boundary

Canonical-ensemble moves only. Semi-grand-canonical and
variance-constrained SGC moves are explicit future work and would
likely live in a sibling package (or a sublattice of this one) once
needed. The `Move` ABC is canonical-ensemble-shaped: `propose`
returns `(sites, species)` for a fixed-composition update.

### Module boundaries

One move per file under `src/mchammer_moves/moves/`. The ensemble
that hosts moves lives in `src/mchammer_moves/ensemble.py`. No
multiprocessing concerns in this package — `mchammer_pt` handles
those. No system-specific geometry — chain construction, sublattice
identification, etc. live in the consumer project (e.g.
`data_NbO2F`).

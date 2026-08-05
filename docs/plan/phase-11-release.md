# Phase 11: Release

**Objective.** Make the repository readable by someone who has thirty seconds, five
minutes or an afternoon, then open the final pull request.

**Depends on.** Phase 10.
**Branch.** `phase-11-release`

## Why this phase exists

The repository is the artefact a reader judges. Work that cannot be understood in
the first minute is work that does not count.

## Build

1. **README.** In order: what this is in two sentences, the headline results as a
   small table, one image, how to install and run it in four commands, the layout
   of the code, and the limitations. Limitations are not optional. State the grid
   size, the number of bodies, the dataset size, the CPU only constraint and the
   horizons beyond which each surrogate stops being physical.
2. **Results document.** `docs/results.md`. Per system: the accuracy against
   horizon curve, invariant behaviour, the held out regime gap, the speedup at
   matched accuracy, the break even rollout count, and the calibration result. Then
   the diagnosis agent scores. Every number traceable to a run id.
3. **What did not work.** A short honest section covering the metrics that were cut
   because nothing could make them fail, the architectures that did not beat their
   baseline, and the failure modes that remain. This section is worth more to a
   reader than the successes, so do not compress it away.
4. **Architecture document.** Update `docs/architecture.md` to describe what was
   built rather than what was planned, including any protocol changes forced by the
   second system and why.
5. **Reproduction.** A single documented path from a clean checkout to the headline
   numbers, with the expected runtime on 8 CPU cores stated at each step.
6. **Style sweep.** Read every committed comment, docstring and document. Enforce UK
   English, no em-dashes, succinct plain English. Remove any comment that restates
   the code. Remove dead code, unused configuration keys and stale to-do notes.
7. **Final checks.** Full test suite including slow tests, coverage measured and
   stated, `mypy --strict` clean, `ruff` clean, CI green.

## Definition of done

- A reader who knows physics but not this repository can state what it does and how
  well it works from the README alone.
- Every number in the README and results document maps to a run id that can be
  regenerated.
- No file, function or configuration key exists that nothing uses.
- The final pull request is open, with a description that follows the writing
  rules: UK English, no em-dashes, succinct, plain English. It states what was
  built, what the results were, what the limitations are, and what was left out.

## Out of scope

No blog post, no packaging to PyPI, no documentation site.

## Notes for the session

Write the limitations section first, while the failures are still fresh. It is the
part most likely to be softened if left until last, and softening it is the one
thing that would cost the repository its credibility.

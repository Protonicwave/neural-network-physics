# Implementation plan

Eleven phases. Each phase is one session in a fresh chat.

## How to run a phase

Open a new chat in this repo and say:

> Read docs/plan/README.md, docs/plan/00-architecture.md and
> docs/plan/phase-NN-name.md, then implement that phase.

`CLAUDE.md` loads automatically and carries the writing style and engineering
standards. The architecture document carries the shape of the system. The phase
brief carries the work. Nothing else needs to be in context.

Work on a branch named after the phase, for example `phase-02-nbody`. Open a pull
request at the end of each phase. Do not merge phases out of order.

## Phases

| Phase | Name | Depends on | Delivers |
|---|---|---|---|
| 01 | Foundations | none | Tooling, CI, core protocols, config, CLI skeleton |
| 02 | N-body system | 01 | Validated symplectic solver and its invariants |
| 03 | Fluid system | 01 | Validated pseudo-spectral solver and its invariants |
| 04 | Data pipeline | 02, 03 | Trajectory generation, storage, splits, normalisation |
| 05 | Evaluation harness | 04 | System agnostic metrics, proven against broken baselines |
| 06 | Reporting | 05 | Run records, reports, run to run comparison |
| 07 | Models and training | 06 | Predictor interface, training loop, N-body model |
| 08 | Fluid model | 07 | Neural operator for the fluid system, training code unchanged |
| 09 | Speed and uncertainty | 08 | Speedup at matched accuracy, calibrated uncertainty |
| 10 | Diagnosis agent | 09 | Agent that explains regressions, scored on injected faults |
| 11 | Release | 10 | Documentation, results writeup, final pull request |

## Two rules that hold across the whole plan

**The harness is built before the good model.** Phase 05 lands before phase 07.
An evaluation suite that has only ever seen a working model is untested. Every
metric earns its place by catching a predictor that is broken on purpose.

**The harness never knows which system it is looking at.** Metrics read invariants
and state descriptions that the system declares. If a metric needs to branch on
whether it is N-body or fluid, the abstraction is wrong and the fix goes in the
abstraction, not the metric.

## Scope control

Two physical systems and two model families. That is the ceiling. Additional
systems, architectures or datasets are out of scope. If a phase finishes early,
spend the time on tests and documentation, not on scope.

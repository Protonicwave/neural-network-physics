# Phase 09: Speed and uncertainty

**Objective.** Measure the speedup a surrogate actually delivers at matched
accuracy, and give it calibrated uncertainty.

**Depends on.** Phase 08.
**Branch.** `phase-09-speed-uncertainty`

## Why this phase exists

A surrogate that is not faster than the solver it replaces has no purpose. This is
the question the whole project exists to answer, and it is the one most commonly
skipped.

## Build

1. **Benchmark harness.** Fair timing of the reference solver and each surrogate.
   Warmup runs, repeated trials, median and interquartile range rather than the
   best time. Thread count fixed and recorded. Batch size recorded, since a
   surrogate that only wins when batched must say so.
2. **Matched accuracy comparison.** The honest comparison is not surrogate against
   solver at default settings. It is surrogate against solver run at the coarsest
   settings that reach the same accuracy. Produce a curve of accuracy against wall
   clock for both, and report the speedup at the crossing points. If the solver
   wins in some regime, say so.
3. **Cost accounting.** Report the one off training cost and the data generation
   cost alongside the per rollout saving, and compute the break even number of
   rollouts. A surrogate that saves an hour but costs a day to train pays for
   itself only past a certain point, and that point is the useful number.
4. **Uncertainty.** A deep ensemble of a small number of models trained from
   different seeds, since it is simple, strong and parallel over cores. Predictive
   spread is the uncertainty estimate.
5. **Calibration metrics.** Reliability diagram, expected calibration error, and
   the correlation between predicted spread and actual error over the rollout.
   Added to the phase 05 harness as ordinary metrics.
6. **Uncertainty as a trust signal.** Report whether the ensemble spread grows
   before the rollout error becomes unacceptable. If it does, the surrogate can say
   when to fall back to the solver, which is the practically useful result.

## Definition of done

- Benchmarks are stable, with reported variance across repeats below a stated
  threshold, and the machine specification recorded in the run record.
- An accuracy against wall clock curve exists for both systems, with the speedup at
  matched accuracy stated as a single headline number each.
- The break even rollout count is computed and stated.
- Calibration metrics are integrated into the harness and pass their own sentinel
  tests, using an overconfident predictor with artificially small spread.
- The correlation between spread and error is reported honestly, including if it is
  weak.

## Out of scope

Bayesian neural networks, conformal prediction, GPU benchmarking.

## Notes for the session

Timing on a shared laptop is noisy. Fix thread counts, close other work, repeat
trials, and report the spread. A speedup claim with no variance stated is not a
measurement.

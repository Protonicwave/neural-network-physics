# Phase 05: Evaluation harness

**Objective.** A system agnostic suite of metrics, proven to work by catching
predictors that are broken on purpose.

**Depends on.** Phase 04.
**Branch.** `phase-05-evals`

## Why this phase exists

This is the centre of the project. It is built before any real model so that every
metric is tested against known badness rather than tuned until a favoured model
looks good.

## Build

1. **Rollout driver.** Takes a Predictor, an initial state and a horizon, and
   produces a trajectory. Handles batching. Records wall clock time. Detects
   divergence and stops early with a recorded reason rather than producing NaN.
2. **Reference predictors.** The system's own solver, wrapped as a Predictor, is
   the ground truth. Then a set of deliberately broken ones, which are permanent
   test fixtures and not throwaway code:
   - persistence, which returns the input state unchanged
   - linear extrapolation from the previous two states
   - the true solver with Gaussian noise added at each step
   - the true solver with a small constant energy injection each step, which is
     accurate over one step but unphysical over many
   - the true solver with a deliberately broken symmetry, for example a small fixed
     rotation applied each step
3. **Metrics.** Each registered by name, each returning named scalars and optional
   plot data. All read the system's declared invariants and symmetries rather than
   knowing anything about physics themselves.
   - **One step error.** Normalised error for a single step. Included mainly to
     show later how misleading it is on its own.
   - **Rollout error growth.** Error against ground truth as a function of horizon,
     reported as a curve plus summary numbers: the horizon at which normalised
     error first exceeds given thresholds.
   - **Invariant drift.** For each declared invariant, relative drift over the
     rollout, respecting whether it should be conserved or should decay. A model
     that gains energy in a viscous fluid must be flagged even if its error is low.
   - **Symmetry violation.** Apply a declared symmetry to the initial condition,
     roll out, invert the transformation, and compare with the untransformed
     rollout. Reports equivariance error.
   - **Distribution drift.** Compares summary statistics of predicted states
     against ground truth over long rollouts, for example the energy spectrum for
     the fluid and the radial distribution for N-body. Catches a model that stays
     stable but stops looking like the right system.
   - **Regime gap.** The same metrics computed on the held out regimes, reported
     against in distribution values as an explicit gap.
4. **Suites.** A named collection of metrics and settings, defined in configuration,
   so an evaluation is reproducible and comparable across runs.
5. **CLI.** `nnp eval run` takes a predictor specification and a suite and writes a
   structured result file.

## Definition of done

**Sentinel tests are the core deliverable.** For every metric, a test asserting that
it flags the broken predictors it should flag, and does not flag the true solver.
In particular:

- Invariant drift catches the energy injecting predictor, which one step error does
  not. Assert both facts in the same test, since the contrast is the point.
- Symmetry violation catches the rotating predictor.
- Rollout error growth separates persistence from linear extrapolation in the
  expected order.
- Distribution drift catches the noise injecting predictor at long horizon.
- Every metric returns benign values for the true solver evaluated against itself.

Further requirements:

- No metric imports anything from `systems` or `models`. Enforce this with an
  import test, not a convention.
- The full suite runs on both systems from the same configuration with only the
  system name changed.

## Out of scope

No trained models. No reporting or plots beyond returning plot data. Do not tune
thresholds to make anything look good.

## Notes for the session

If a metric cannot be made to fail by any broken predictor, it is not measuring
anything. Delete it or fix it. Record any metric that was cut for this reason in the
pull request description, because that reasoning is worth showing.

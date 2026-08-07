# Architecture

Read this before any phase. It defines the shape every phase must build towards.

## The idea in one paragraph

A physical system is expensive to simulate accurately. A neural surrogate learns
to imitate the solver and runs faster. The question that matters is not whether
the surrogate has low test error, it is where the surrogate can be trusted. This
repository answers that question with an evaluation harness that measures error
growth over long rollouts, drift in conserved quantities, behaviour outside the
training regime, respect for symmetries, and real speedup at matched accuracy.

## Layers

Dependencies point inwards only. An outer layer may import an inner layer. An
inner layer may never import an outer one.

```
cli            command line entry points
  reporting    run records, reports, comparison
  agent        report reading diagnosis, fault injection scoring
    evals      metrics, rollout drivers, reference predictors
    training   training loop, checkpointing, schedules
      models   predictors: baselines, graph network, neural operator
      data     generation, storage, datasets, splits, normalisation
        systems  nbody, fluid: dynamics, solvers, invariants, regimes
          core   protocols, types, registry, config, seeding, errors
```

`core` imports nothing internal. `systems` imports only `core`. `evals` imports
`core` and `data`, never `systems` and never `models`.

## Target tree

```
src/nnphysics/
  core/        protocols, state and trajectory types, registry, config, seeding, errors, units
  systems/     base declarations, nbody/, fluid/
  data/        generation, store, dataset, splits, normalisation
  models/      base, baselines/, graph/, operator/
  training/    loop, schedules, checkpoint, metrics logging
  evals/       runner, metrics/, predictors/ (reference and broken), suites
  reporting/   run record, plots, render, compare
  agent/       client, diagnose, faults, scoring
  cli/         app and subcommands
tests/         mirrors src, plus tests/integration
configs/       yaml experiment configs
docs/          architecture, results, plan/
```

## Core abstractions

Six protocols carry the whole design. Each is defined in `core` and implemented
outwards.

**System.** Declares what a physical system is: the shape and meaning of its
state, the parameters that define a regime, the invariants it conserves, the
symmetries it respects, and a reference solver that produces ground truth. Both
N-body and fluid implement this identically from the outside.

**Invariant.** A named scalar computed from a state, with a tolerance and a
statement of whether it should be conserved exactly or approximately. Energy,
momentum, enstrophy. The evaluation harness reads these from the system rather
than knowing them itself.

**Symmetry.** A named transformation that can be applied to a state and to a
prediction. Translation, rotation, reflection. Used to test equivariance without
the metric knowing what the system is.

**Refinement.** A named change of the resolution a state is represented at, with its
inverse. Added in phase 08, because a neural operator claims to be a map between
function spaces rather than between arrays, and the harness has to be able to check
that without knowing what a resolution is. A system whose state is not a discretisation
of a continuous field declares none, and is not asked.

**Predictor.** Anything that maps a state to the next state. The reference solver
is one. A trained network is one. A deliberately broken baseline is one. The
evaluation harness only ever sees this interface, which is what lets the same
suite score a solver, a neural operator and a persistence baseline.

**Metric.** Consumes a rollout and returns named scalars plus optional plot data.
Metrics are registered by name and selected by configuration.

## Locked decisions

Do not reopen these. They are settled so that phases execute rather than debate.

| Area | Decision |
|---|---|
| Language | Python 3.12 |
| Packaging | `uv`, `src/` layout, `pyproject.toml` only |
| Arrays | NumPy for solvers, PyTorch for models |
| Deep learning | PyTorch, CPU first, no JAX |
| Config | Pydantic v2 models loaded from YAML, validated at the boundary |
| CLI | Typer, one `nnp` entry point with subcommands |
| Storage | HDF5 via h5py, one file per shard, JSON manifest alongside |
| Plots | Matplotlib, no seaborn |
| Reports | Markdown and a self contained HTML file, no external assets |
| Lint and format | Ruff |
| Types | mypy strict |
| Tests | pytest, with `slow` and `integration` markers |
| CI | GitHub Actions, lint, types and fast tests on push |
| Agent model | Claude via the Anthropic SDK, model id read from config |

## Determinism

Every artefact must be reproducible from a seed and a config. Run identifiers are
derived from a hash of the resolved config plus the seed, so the same inputs give
the same run id. Generation, splitting, initialisation, shuffling and dropout all
take an explicit generator. No module calls a global random function.

## Regimes and honesty about generalisation

Each system defines named regimes through its parameters, for example a virial
ratio for N-body or a Reynolds number for the fluid. Training data comes from a
declared set of regimes. At least one regime is held out entirely and is never
seen during training or model selection. Reported results always separate in
distribution performance from held out regime performance. Normalisation
statistics are computed from the training split alone, and a test enforces this.

## What good looks like at the end

One command generates data, one trains a model, one evaluates it, one renders a
report comparing it to any previous run. A reader can see, per system, how far a
surrogate can be rolled out before it stops being physical, and how much time it
saves for that accuracy.

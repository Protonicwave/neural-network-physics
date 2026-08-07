# Architecture

What was built. `docs/plan/00-architecture.md` is what was planned, and it is kept as the
brief the phases executed against. This document records the shape the code actually took
and where it departed from that brief.

## The idea in one paragraph

A physical system is expensive to simulate accurately. A neural surrogate learns to imitate
the solver and is meant to run faster. The question that matters is not whether the
surrogate has low test error, it is where the surrogate can be trusted. This repository
answers that with an evaluation harness measuring error growth over long rollouts, drift in
conserved quantities, behaviour outside the training regime, respect for symmetries,
consistency under a change of resolution, calibration of the surrogate's own uncertainty,
and speedup at matched accuracy.

## Layers

Dependencies point inwards only.

```
cli            command line entry points
  agent        report reading diagnosis, fault injection scoring
    reporting  run records, reports, comparison
      evals    metrics, rollout drivers, reference predictors
      training training loop, checkpointing, schedules
        models predictors: baselines, graph network, neural operator
        data   generation, storage, datasets, splits, normalisation
          systems  nbody, fluid: dynamics, solvers, invariants, regimes
            core   protocols, types, registry, config, seeding, errors, units
```

Two rules are enforced by tests that parse the source rather than by convention, because a
convention is what an import added in a hurry breaks.

`tests/evals/test_layering.py` asserts that nothing under `src/nnphysics/evals` imports
`nnphysics.systems`, `nnphysics.models` or `nnphysics.training`, checking every import
including ones inside a function or a `TYPE_CHECKING` block. It also asserts the converse,
that the layer does import `core` and `data`, so the rule cannot be satisfied by a layer
that imports nothing. A third test constructs a violating module in a temporary directory
and asserts the check would catch it.

`tests/agent/test_layering.py` asserts that nothing under `agent` or `reporting` imports
`cli`, and that `agent` imports `reporting` and never the reverse.

### Where this departs from the plan

**The plan left the direction between `agent` and `reporting` unsaid.** Its diagram put them
at the same indentation. An agent whose job is to read reports has to sit outside the layer
that writes them, so the resolution is `agent` depends on `reporting`, and the test pins it.

**`training` was added to the list of layers `evals` may not import.** The plan forbade only
`systems` and `models`. An evaluation harness that imported the training loop could not
score anything the loop had not produced, which is the same failure the original rule was
written to prevent.

## Core abstractions

Seven protocols, not the six the plan named. Each is defined in `core` and implemented
outwards.

**System.** The shape and meaning of a state, the parameters defining a regime, the
invariants conserved, the symmetries respected, the refinements available, an initial state
from a generator, and a reference solver. Both systems implement this identically from the
outside.

**Invariant.** A named scalar computed from a state, with a physical dimension, a relative
tolerance and a statement of whether it is conserved exactly, approximately or is decaying.
The harness reads these from the system rather than knowing them.

**Symmetry.** A named transformation with its inverse, applied to a state to test
equivariance without the metric knowing what the system is.

**Refinement.** A named change of the resolution a state is represented at, with its
inverse and its factor. Added in phase 08 because a neural operator claims to be a map
between function spaces rather than between arrays, and the harness has to check that
without knowing what a resolution is. A system whose state is not a discretisation of a
continuous field declares none and is not asked.

**Predictor.** Anything mapping a state to the next state at a fixed interval. The
reference solver is one, a trained network is one, a baseline broken on purpose is one. The
harness only ever sees this interface, which is what lets one suite score all three.

**UncertainPredictor.** A predictor that also states a spread with its prediction. Added in
phase 09, because calibration is a measurement of a claim and there was no way to make the
claim through `Predictor`. It extends `Predictor` rather than replacing it, so the harness
runs a predictor that states nothing without special casing, and `calibration` reports zero
steps for it rather than a perfect score.

**Metric.** Consumes a rollout and returns named scalars plus optional curves for plotting.
Registered by name and selected by configuration.

### What the second system forced

`Refinement` and `UncertainPredictor` are both additions the plan did not have, and only
the first was forced by the second system. The rest of the protocol surface survived two
systems unchanged, which is the result the plan was hoping for: N-body and fluid differ in
state shape, in what they conserve, in which symmetries they have and in whether a finer
grid exists at all, and every one of those differences is declared rather than branched on.

The one place the abstraction had to be extended rather than reused is model applicability.
The plan implied models would be named per system. What was built declares applicability
from the shape of the state: `operator` and `convolution` say `gridded` rather than `fluid`
and accept any system whose fields are two dimensional arrays, refusing at construction
with the reason when handed a field of 32 masses. That is the same rule the harness follows
one layer up.

## The tree as built

```
src/nnphysics/
  core/       protocols, state and trajectory types, registry, config, params,
              seeding, errors, units
  systems/    base declarations, nbody/, fluid/
  data/       spec, generation, store, layout, manifest, dataset, splits,
              normalisation, fields, build, verify
  models/     base, layers, baselines, ensemble, graph/, operator/
  training/   loop, curriculum, losses, schedules, checkpoint, history
  evals/      runner, rollout, result, snapshots, speed, benchmark,
              metrics/, predictors/
  reporting/  record, document, explain, layout, plots, render, style,
              theme, compare, environment, index
  agent/      client, context, causes, diagnose, faults, scoring
  cli/        app, commands, data, evals, train, ensemble, benchmark,
              report, diagnose, faultrun, pipeline, predictors
tests/        mirrors src, plus tests/integration
configs/      nbody.yaml, fluid.yaml, faults.yaml, agent.yaml, example.yaml
docs/         architecture, results, reproduction, results/, plan/
```

Three differences from the target tree in the plan are worth naming.

**There is no `evals/suites.py`.** A suite turned out to be configuration rather than code:
the metrics and predictors a run scores are a list in the YAML, resolved through the
registry. Adding a module to hold named suites would have been a second way to say the same
thing.

**`reporting` grew a document model.** `document.py` builds one structured document and
`render.py` turns it into Markdown and into a single self contained HTML file, so both come
from one build and rendering the same record twice is byte identical. `explain.py` holds
the one line explanation and the improvement direction for every scalar, and a metric
producing a number nobody has explained stops the report rather than printing it bare.

**`cli` has more modules than commands.** `pipeline.py` and `predictors.py` are shared
construction used by several commands, kept out of the command modules so that `train`,
`eval run` and `benchmark` build a predictor the same way.

## The evaluation harness

The harness is handed a factory and builds every predictor through one interface, so it
never learns whether it is looking at a solver, a trained network or a fixture broken on
purpose.

### The predictors

One correct predictor and seven wrong on purpose. They are permanent fixtures rather than
test scaffolding, because a metric only means something next to the numbers they produce.

| Predictor | What it does |
|---|---|
| `reference` | The system's own solver, folded to one stored interval per step |
| `persistence` | Returns its input, one interval later |
| `linear_extrapolation` | Continues in a straight line through the last two states |
| `noise` | The solver with Gaussian noise added each step |
| `energy_injection` | The solver with every declared rate amplified by a fixed factor |
| `symmetry_break` | The solver with a declared symmetry applied after each step |
| `calibrated` | Noise, stating the deviation it has actually accumulated |
| `overconfident` | The same noise, stating a hundredth of it |

The last two differ only in what they claim. At equal confidence they produce bit identical
states, so every metric that reads states alone rates them the same and only `calibration`
separates them. That is what makes it a measurement of the claim rather than of the noise,
and `tests/evals/test_sentinels.py` asserts it.

### The metrics

`one_step_error`, `rollout_error`, `invariant_drift`, `symmetry_violation`,
`distribution_drift`, `resolution_generalisation` and `calibration`. Every one has a
sentinel test asserting both what it catches and what it does not, because a metric that
flags everything measures nothing.

Three results from those sentinels are worth stating, because each is a case where the
obvious metric gives the wrong answer.

- The energy injecting predictor is a thousand times more accurate than persistence over
  one step, and `invariant_drift` still rates it a billion times outside the declared
  tolerance. One step error on its own is close to meaningless.
- `persistence` conserves every N-body invariant perfectly, so `invariant_drift` rates it
  flawless. In the viscous fluid it does not, because energy there is declared decaying and
  holding it still is as unphysical as adding it.
- The predictor that rotates the world each step conserves every invariant and is perfectly
  equivariant under rotation. Only the symmetries it does not commute with reveal it, which
  is why every declared symmetry is tested and the worst reported.

Two metrics decline to score rather than flatter a predictor. `calibration` reports zero
steps for a predictor that states no uncertainty, which is six of the eight above, because
flattering six predictors that never made a claim would leave the metric meaningless.
`resolution_generalisation` reports zero steps for a system that declares no refinement:
there is no finer version of 32 point masses, and a number there would be an invention.
`tests/evals/metrics/test_resolution.py` brackets that metric with a predictor written in
wavenumbers, which it must not flag, and one written in grid cells, which it must catch,
four orders of magnitude apart.

## The models

| Model | System | What it is |
|---|---|---|
| `constant` | any | Persistence with one learned offset per field. The floor. |
| `mlp` | any | A network on the flattened state, predicting the normalised update |
| `graph` | N-body | Message passing for the acceleration, velocity Verlet for the time stepping |
| `operator` | gridded | Learned multipliers on a truncated band of Fourier modes |
| `convolution` | gridded | Periodic padded stencils, at the operator's parameter count |

The last two say `gridded` rather than `fluid` because neither names a field. What makes
them applicable is that the states are two dimensional arrays, which they read from the
dataset; handed a field of masses they refuse at construction and say why.

The graph network carries the one real architectural choice: it predicts the derivative and
lets the symplectic integrator do the time stepping. A model that maps a state to the next
one has to learn the force law, which is hard, and the time stepping, which is already known
exactly, and the approximation to the known half is what a long rollout accumulates.
Velocity Verlet also keeps the energy error bounded whatever acceleration it is handed, so
measured drift belongs to the model rather than to the integrator.

Translation, rotation and Galilean invariance are built in rather than learned: only pair
distances and masses reach any layer, and the output is a sum of learned scalars times unit
vectors. That is the form Newtonian gravity already takes, so it costs nothing to impose,
and the equivariance numbers in `docs/results.md` should be read knowing it. Velocity never
enters the acceleration network.

## The training loop

One loop, system agnostic, driven by configuration: deterministic seeding, gradient
clipping, a cosine schedule with warmup, early stopping on a validation rollout rather than
on the training loss, and best and last checkpoints carrying optimiser state. The rollout
curriculum trains on one step, then four, then eight, with gradients truncated by the
window.

Three decisions in it earned their place by being wrong first.

**Shuffling is a function of the run seed and the epoch number, not of a generator carried
forward.** A resume then needs no random state and cannot forget it.

**The training objective counts a large residual linearly and the validation metric does
not.** Measured here, a curriculum stage lengthening from one step to four under a plain
squared error raised the one step error by a factor of three hundred inside one epoch and
the validation rollout never recovered. A rollout that has begun to diverge produces
residuals of tens of standard deviations, and squaring those is a gradient thousands of
times anything else in the batch. Past one standard deviation the loss is linear. The metric
stays a plain squared error, because nothing is optimised against it and a number that
stopped growing once a rollout went wrong is the wrong number to select a model on.

**Early stopping waits for the last curriculum stage, and each stage gets its own patience.**
Stopping early claims that more training of this kind will not help, and a scheduled longer
window is not more of the same kind. The fluid operator is where this was found and it cost
a whole stage: a patience of ten against a stage change thirteen epochs later ended the run
at epoch 27, so the eight step stage never ran. With the rule in place the same settings
reached it. The N-body runs were exposed to the same defect and escaped it only because
their numbers happened to keep improving.

What the curriculum does is not what one system alone suggested, and the two disagree enough
to be worth stating. On N-body the validation rollout improved at every lengthening: 0.1388
on one step, 0.1015 on four, 0.0785 on eight. On the fluid operator the improvement comes at
the lengthening and is then given back: the four step stage opened at 0.5762, the best number
that run ever produced, and degraded to 4.625 over the next twelve epochs while its training
loss fell the whole way. The eight step stage reset it to 0.6622. The convolutional network
on the same schedule did not do this, improving through both longer stages to 0.3103.

So a longer window helps at the moment it is applied and does not keep helping, and how long
that lasts is a property of the model rather than of the schedule. Selecting on the
validation rollout is what makes that survivable: it picks the epoch the lengthening bought
and ignores the twelve that followed.

## Determinism

Every artefact is reproducible from a seed and a configuration. The run identifier is a
hash of the resolved configuration plus the seed, so the same inputs give the same
identifier, and a dataset identifier is derived the same way. Generation, splitting,
initialisation, shuffling and dropout all take an explicit generator, and no module calls a
global random function.

Shuffling is a function of the run seed and the epoch number rather than of a generator
carried forward, so epoch seventeen sees the same order whether it was reached in one run
or three and a resume needs no random state. `tests/training/test_loop.py` asserts that
training four epochs, and training two then resuming two, give bit identical weights.

Two determinism limits are real and documented rather than fixed. Generating the same
dataset with a different worker count gives the same hash, but evaluating the same weights
at a different thread count does not give the same error, because the order of accumulation
changes and a chaotic system amplifies it. And a timing is not deterministic at all, which
is why the benchmark reports a median with an interquartile range and marks a point
unstable when the range exceeds a tenth of the median.

## Configuration

Pydantic v2 models loaded from YAML and validated at the boundary. Environment variables
named `NNP_SECTION__KEY` override the file, which is what makes a control run a one line
change: `NNP_MODEL__NAME=convolution` trains the fluid control against the same file that
trains the operator, and the comparison stays a comparison.

## Records and schema versioning

`eval run` writes a run record beside its result file: the resolved configuration, the run
identifier, the commit, the machine, the library versions, the timings, every metric output
and the paths to the rest. Both the record and the suite result carry a schema version, and
an old one is upgraded on read rather than refused, because a run whose numbers can no
longer be read is a run that never happened.

One consequence of that principle was found by the fault suite. A diverged run's loss is
not a finite number, Pydantic writes it as `null`, and it then fails to read back as a
float, so the run that most needed explaining was the one whose record could not be opened.
A loss that went to infinity is a measurement rather than a value to clean up, so it is
serialised, and `tests/reporting/test_record.py` pins the round trip.

## Regimes and honesty about generalisation

Each system defines named regimes through its parameters. Training data comes from a
declared set, at least one regime is held out entirely and is never seen during training or
model selection, and results always separate in distribution performance from held out
performance. Normalisation statistics are fitted to the training split alone and a test
enforces it.

The held out regimes were chosen to be a different class rather than a harder setting of
the same one: 4 bodies in two tight binaries against 32 in a cluster, and a shear layer at
Reynolds 200 against a Taylor-Green vortex at 100 and decaying turbulence at 500. That is why both
surrogates fail there, and reporting the failure is the reason the regimes exist.

## Locked decisions, as built

| Area | Decision | As built |
|---|---|---|
| Language | Python 3.12 | unchanged |
| Packaging | `uv`, `src/` layout, `pyproject.toml` only | unchanged |
| Arrays | NumPy for solvers, PyTorch for models | unchanged |
| Deep learning | PyTorch, CPU first, no JAX | unchanged, CPU wheel pinned through a `uv` index |
| Config | Pydantic v2 from YAML, validated at the boundary | unchanged, plus `NNP_` environment overrides |
| CLI | Typer, one `nnp` entry point | unchanged, seven top level commands |
| Storage | HDF5 via h5py, one file per shard, JSON manifest | unchanged |
| Plots | Matplotlib, no seaborn | unchanged |
| Reports | Markdown and a self contained HTML file | unchanged, plots embedded as data URIs |
| Lint and format | Ruff | unchanged |
| Types | mypy strict | unchanged, two overrides in `pyproject.toml`, each with a reason |
| Tests | pytest, `slow` and `integration` markers | unchanged |
| CI | GitHub Actions, lint, types and fast tests | unchanged |
| Agent model | Claude via the Anthropic SDK, model id from config | built, but scored without a credential; see `docs/results.md` |

# neural-network-physics

Neural surrogates for two physical systems, gravitational N-body and 2D incompressible
fluid flow, with an evaluation harness that measures where a surrogate can be trusted:
error growth over long rollouts, drift in conserved quantities, behaviour outside the
training regime, respect for symmetries, and speedup at matched accuracy. The harness is
system agnostic. See `docs/plan/` for the implementation plan.

## Getting started

```sh
uv sync
uv run nnp --help
```

## Data

Generate a dataset, check it and describe it:

```sh
uv run nnp data generate --config configs/nbody.yaml
uv run nnp data verify   --config configs/nbody.yaml
uv run nnp data stats    --config configs/nbody.yaml
```

`generate` writes one HDF5 shard per group of trajectories and a JSON manifest recording
the system, its parameters, the seed, both time intervals, the splits and a hash of every
shard. A dataset without a manifest is not a dataset. `verify` re-hashes every shard and
re-derives a sample of trajectories from their recorded seeds, so it catches both a file
that has changed and data that never matched the configuration. `stats` fits
normalisation statistics to the training split alone and writes them alongside.

Generating twice from the same configuration and seed gives the same dataset hash,
whatever the number of workers.

### Default datasets

Both fit the target machine, eight CPU cores and no CUDA. Timings are wall clock on that
machine with the default worker count of cores minus one.

| Config | Trajectories | States each | Stored step | Solver step | On disk | Generation |
|---|---|---|---|---|---|---|
| `configs/nbody.yaml` | 192, 64 per regime | 256 | 0.01 | 0.001 | 30 MB | 54 s |
| `configs/fluid.yaml` | 96, 32 per regime | 64 | 0.05 | 0.005 | 122 MB | 62 s |

The solver step is smaller than the stored one because stability and accuracy demand it,
and the surrogate is not obliged to inherit that: the ratio is where its speedup comes
from, so both intervals are recorded.

Each system trains on two regimes and holds a third out entirely. N-body trains on
`cold_collapse` and `virialised_cluster` and holds out `hierarchical_pair`. The fluid
trains at Reynolds 100 and 500 and holds out the shear layer at 200. Held out data reaches
no training split, and normalisation statistics are fitted to the training split alone.

## Evaluation

Score a set of predictors against the suite a configuration names:

```sh
uv run nnp eval list
uv run nnp eval run --config configs/nbody.yaml
uv run nnp eval run --config configs/nbody.yaml --predictor noise:scale=0.02 --split test
```

`run` rolls each predictor forward from initial conditions taken from the dataset, scores
it against the stored ground truth and writes one JSON result file holding every number,
every curve and the settings that produced them. Both default suites run on the test split
and on the held out regime, and report the difference between them as an explicit gap.

| Config | Predictors | Steps | Initial conditions | Evaluation |
|---|---|---|---|---|
| `configs/nbody.yaml` | 6 | 255 | 4 per split | 8 s |
| `configs/fluid.yaml` | 6 | 63 | 4 per split | 32 s |

The harness is system agnostic and a test enforces it: nothing under `src/nnphysics/evals`
may import a system or a model. Metrics read the invariants and symmetries a system
declares and know no physics of their own.

### The predictors

The suite ships one correct predictor and five that are wrong on purpose. The broken ones
are permanent fixtures rather than test scaffolding, because a metric only means something
next to the numbers they produce.

| Predictor | What it does |
|---|---|
| `reference` | The system's own solver, folded to one stored interval per step |
| `persistence` | Returns its input, one interval later |
| `linear_extrapolation` | Continues in a straight line through the last two states |
| `noise` | The solver with Gaussian noise added each step |
| `energy_injection` | The solver with every declared rate amplified by a fixed factor |
| `symmetry_break` | The solver with a declared symmetry applied after each step |

### The metrics

`one_step_error`, `rollout_error`, `invariant_drift`, `symmetry_violation`,
`distribution_drift` and `resolution_generalisation`. Every one of them has a sentinel test
asserting both what it catches and what it does not, because a metric that flags everything
measures nothing. Three results worth stating, all asserted in
`tests/evals/test_sentinels.py`:

- The energy injecting predictor is a thousand times more accurate than persistence over
  one step, and `invariant_drift` still rates it a billion times outside the tolerance the
  system declared. One step error on its own is close to meaningless.
- `persistence` conserves every N-body invariant perfectly, so `invariant_drift` rates it
  flawless. In the viscous fluid it does not, because energy there is declared decaying and
  holding it still is as unphysical as adding it.
- The predictor that rotates the world each step conserves every invariant and is perfectly
  equivariant under rotation. Only the symmetries it does not commute with reveal it, which
  is why every declared symmetry is tested and the worst is reported.

`resolution_generalisation` is the one metric that does not apply to every system, and it
says so rather than scoring. It refines the initial condition to a resolution the system
declares, rolls the predictor forward there, coarsens the result back, and reports both how
far the predictor drifted from its own answer and how much accuracy that cost. A system
whose state is not a discretisation of a continuous field declares no refinement and reports
zero steps tested: there is no finer version of thirty two point masses, and a number there
would be an invention. `tests/evals/metrics/test_resolution.py` brackets it with a predictor
written in wavenumbers, which it must not flag, and one written in grid cells, which it must
catch; the two are separated by four orders of magnitude.

## Models and training

Train a surrogate and score it against the same suite:

```sh
uv run nnp train --config configs/nbody.yaml
uv run nnp train --config configs/nbody.yaml --resume
uv run nnp train --config configs/fluid.yaml
NNP_MODEL__NAME=mlp NNP_MODEL__HYPERPARAMETERS='{}' uv run nnp train --config configs/nbody.yaml
```

`train` trains the configured model, loads the best checkpoint back and evaluates it beside
the reference solver and the five broken baselines, on the test split and on the held out
regime, then writes one run record carrying both halves. A set of weights whose numbers
came from a different invocation against a suite nobody wrote down is not a result.

The evaluation harness never learns that it is looking at a model. It is handed a factory
and builds the predictor through the same interface it builds a solver through, so nothing
under `src/nnphysics/evals` imports the models layer and the test that enforces that still
passes.

### The models

| Model | System | What it is |
|---|---|---|
| `constant` | any | Persistence with one learned offset per field. The floor. |
| `mlp` | any | A network on the flattened state, predicting the normalised update |
| `graph` | N-body | Message passing for the acceleration, velocity Verlet for the time stepping |
| `operator` | gridded | Learned multipliers on a truncated band of Fourier modes |
| `convolution` | gridded | Periodic padded stencils, at the operator's parameter count |

The last two say `gridded` rather than `fluid` because neither names a field. What makes
them applicable is that the states are two dimensional arrays, which they read from the
dataset; handed a field of masses, one number per body, they refuse at construction and say
why. That is the same rule the harness follows one layer up.

The graph network is where the phase's one real design choice lives: **it predicts the
derivative and lets the symplectic integrator from phase 02 do the time stepping.** A model
that maps a state to the next one has to learn the force law, which is hard, and time
stepping, which is already known exactly; the approximation to the known half is what a
long rollout accumulates. Velocity Verlet also keeps the energy error bounded whatever
acceleration it is handed, so drift that is measured belongs to the model rather than to
the integrator.

Two properties are built into the network rather than learned, and the equivariance numbers
below should be read knowing it. Only pair distances and masses reach any layer, and the
output is a sum of learned scalars times unit vectors, so translation, rotation and
Galilean invariance hold at floating point round off by construction. That is the form
Newtonian gravity already takes, so it costs nothing to impose.

Velocity never enters the acceleration network. The `mlp` sees the whole flattened state,
which ties it to the exact number of bodies it was trained on: it does not do worse on the
four body held out regime, it cannot run there at all, and it says so rather than
producing a number.

### The loop

One loop, system agnostic, driven by configuration. Deterministic seeding, gradient
clipping, a cosine schedule with warmup, early stopping on a validation rollout rather
than on the training loss, and best and last checkpoints carrying optimiser state.

Three decisions in it earned their place by being wrong first.

**Shuffling is a function of the run seed and the epoch number, not of a generator carried
forward.** Epoch seventeen sees the same order whether it was reached in one run or three,
so a resume needs no random state and cannot forget it. `tests/training/test_loop.py`
asserts that training four epochs and training two then resuming two give bit identical
weights, which is what catches a forgotten optimiser state.

**The training objective counts a large residual linearly and the validation metric does
not.** Measured here, a curriculum stage lengthening from one step to four under a plain
squared error raised the one step error by a factor of three hundred inside one epoch and
the validation rollout never recovered: a rollout that has begun to diverge produces
residuals of tens of standard deviations, and squaring those is a gradient thousands of
times anything else in the batch. Past one standard deviation the loss is linear. The
metric stays a plain squared error, because nothing is optimised against it and a number
that stopped growing once a rollout went wrong is the wrong number to select a model on.

**Early stopping waits for the last curriculum stage, and each stage gets its own
patience.** Stopping early claims that more training of this kind will not help, and a
scheduled longer window is not more of the same kind. The fluid operator is where this was
found and it cost a whole stage: patience of ten against a stage change thirteen epochs
later ended the run at epoch 27, so the eight step stage never ran. With the rule in place
the same settings reached it and every one of its epochs beat the best of the four step
stage. The N-body runs were exposed to the same defect and escaped it only because their
numbers happened to keep improving.

The rollout curriculum trains on one step, then four, then eight, with gradients truncated
by the window. It is what makes a surrogate stable over a long horizon, and its absence is
the usual reason surrogates diverge.

What it does is not what the phase 07 numbers alone suggested, and the two systems disagree
enough to be worth stating. On N-body the validation rollout improved at every lengthening,
0.1388 on one step, 0.1015 on four, 0.0785 on eight. On the fluid operator the improvement
comes at the lengthening and is then given back: the four step stage opened at 0.5762, the
best number the run ever produced, and degraded monotonically to 4.625 over the next twelve
epochs while its training loss fell the whole way. The eight step stage reset it to 0.66.
The convolutional network on the same schedule did not do this, improving through both
longer stages to 0.3103.

So a longer window helps at the moment it is applied and does not keep helping, and how
long that lasts is a property of the model rather than of the schedule. Selecting on the
validation rollout is what makes the difference survivable: it picks the epoch the
lengthening bought and ignores the twelve that followed.

### What the default N-body run produces

`uv run nnp train --config configs/nbody.yaml` on eight CPU cores: 1328 s of training over
40 epochs and 12 s of evaluation, for a graph network of 62,849 parameters.

Both splits, 255 steps from four initial conditions. A horizon is in simulated time, so a
horizon of 0.0375 is 3.75 steps of the stored interval.

| Predictor | one step error | horizon at 10% error | error at 255 steps | worst invariant drift |
|---|---|---|---|---|
| `reference` | 0 | never exceeded | 0 | 0 |
| `persistence` | 0.342 | 0.0125 | 1.04 | 0.1 |
| `mlp` | 0.350 | 0.0125 | 2.13 | 3.3e13 |
| `graph` | **0.053** | **0.0375** | 2.08 | 4.0e13 |

On the held out regime, two tight binaries the model has never seen:

| Predictor | one step error | horizon at 10% error | error at 255 steps |
|---|---|---|---|
| `persistence` | 0.042 | 0.03 | 0.82 |
| `graph` | 1.11 | 0.01 | 635 |

Four things in that, and only the first is good.

**The graph network clears persistence in distribution.** Six times lower one step error,
and three times longer before the error reaches a tenth of the signal.

**Its error at 255 steps is worse than persistence, and that is not a contradiction.**
Persistence saturates: once the states are uncorrelated it cannot get worse. A surrogate
that has gone unstable can, and does. The horizon is the fair comparison and the final
error is the honest warning, which is why the suite reports both.

**It does not generalise to the held out regime at all.** One step error of 1.11 against
persistence's 0.042, an error of 635 by the end, and one rollout abandoned as divergent.
Two separated timescales is a configuration class no training regime contained, and
nothing in the training numbers predicted this. Reporting it is the whole reason a regime
is held out.

**The multilayer perceptron does not beat persistence on one step error**, which the phase
brief expected it to. It scores 0.350 against persistence's 0.342, and no learning rate
between 0.0002 and 0.002, and no window stride between one and eight, changed that. What
happens is visible in the training log: its own training loss falls at every epoch while
its validation rollout worsens from the second, so selecting on a rollout selects an almost
untrained model. Training it further makes the reported one step error worse rather than
better, 0.343 at the selected epoch against 0.481 nine epochs later.

The two numbers disagree because they normalise differently. The training loss divides by
the spread of the whole training split, so the collapsing phase of a trajectory, where
velocities are largest, dominates it. The suite divides by the size of the state at each
step, so every phase counts equally. The perceptron buys accuracy in the violent phase and
pays for it everywhere else, and only one of those two measures notices. That is the same
point the phase 05 fixtures make about one step error, arriving this time from a model that
was trying its best rather than from one broken on purpose.

The graph network beats the perceptron on rollout error, which is the comparison the plan
asks for: a horizon three times longer, and a lower error at the end of it.

### Equivariance, and what its number means here

The suite reports 0.0355 for the graph network's worst symmetry violation on the test split
and 2.5e-7 on the held out one. Per step the model is equivariant to floating point round
off under all three declared symmetries, by construction rather than by training, and
`tests/models/graph/test_model.py` asserts exactly that. The suite measures over a 32 step
rollout instead. The reference solver, in double precision, reads 1.5e-12 over the same
rollout on the same split; this model works in single precision, where round off is nine
orders of magnitude larger. Round off amplifying in a chaotic 32 body cluster accounts for
the gap; a broken symmetry would not explain why the four body held out regime, which is
far more regular, stays at 2.5e-7.

### What the default fluid run produces

`uv run nnp train --config configs/fluid.yaml` trains the neural operator; the same file
with `NNP_MODEL__NAME=convolution` trains the control. Both are 2D fields on a 64 by 64
grid, and their parameter counts are matched deliberately: 149,001 against 148,889, which
is eight parts in ten thousand. Without that the comparison would be about capacity.

Both splits, 63 steps from four initial conditions. The operator took 1,941 s to train on
eight cores. The convolutional network took about 5,800 s, roughly three times as much for
the same number of parameters, because a stencil costs nine multiplies per grid point where
a one by one convolution costs one. That figure is the sum of its epochs with two excluded:
the run was interrupted and resumed, and the machine was suspended during those two, so
their wall clock says nothing about the model.

| Predictor | one step error | horizon at 10% error | error at 63 steps | worst invariant drift |
|---|---|---|---|---|
| `reference` | 0 | never exceeded | 0 | 0 |
| `persistence` | 0.174 | 0.625 | 1.80 | 6.5e7 |
| `operator` | 0.171 | 0.625 | 937 | 2.3e14 |
| `convolution` | **0.087** | 0.575 | **0.876** | **1.1e8** |

On the held out shear layer at Reynolds 200:

| Predictor | one step error | error at 63 steps | rollouts completed |
|---|---|---|---|
| `persistence` | 0.005 | 0.305 | 4 of 4 |
| `operator` | 0.147 | 656 | 0 of 4, all diverged by step 28 |
| `convolution` | **0.013** | 0.942 | 4 of 4 |

**The neural operator loses to the convolutional baseline, and the phase expected it to
win.** It is beaten on every accuracy number, on every invariant, on equivariance and on
distributional drift, at a parameter count matched to eight parts in ten thousand and on
identical training settings. The operator does not clear persistence on one step error at
all, 0.171 against 0.174, and its rollout diverges: 937 at the end of the test rollout
where persistence saturates at 1.80, and every held out rollout abandoned by step 28.

**The convolutional network is the first surrogate in this repository to beat persistence
on both horizon and final error.** Twice as accurate over one step, and a final error of
0.876 where persistence reaches 1.80, so it is still tracking the flow at the end of the
rollout rather than merely failing to get worse.

**Its enstrophy is close to right and its energy is not.** The suite reports an enstrophy
excursion of 28.8 for the convolution against the solver's own 29.0, so the model
reproduces the viscous decay of enstrophy almost exactly. Energy does not fare as well: an
excursion of 0.513 against the solver's 0.234, roughly twice as far as the true dynamics
move. The metric also separates the direction, and it is growth rather than over
dissipation: the predicted energy stands as much as 0.566 above the true value at some
point in the rollout and only 0.006 below it. The model adds energy the physics did not
supply. That is the unphysical growth this phase was told to report rather than hide, and
it is why the invariant violation reads 1.1e8 for a model whose error curve looks healthy.

**Neither model is fast enough to matter yet.** Measured against the reference solver in
the same run, the operator takes 0.0091 s per stored interval against 0.0115 and the
convolution 0.0072 against 0.0094, so both are about a quarter to a third faster. That is
against a solver taking ten substeps per stored interval, which is where a surrogate's
speedup is supposed to come from. A model that saves thirty per cent and is less accurate
has not earned its place, and the honest reading is that a 64 by 64 grid is too small for
the fixed costs of a network to amortise against a spectral step.

### Resolution generalisation, and what its number means here

This is the one place the operator wins, and it wins by the margin the design predicts.

| Predictor | split | inconsistency | degradation |
|---|---|---|---|
| `operator` | test | **0.022** | **-0.0003** |
| `convolution` | test | 0.636 | 0.233 |
| `operator` | held out | **0.00024** | **-8.6e-7** |
| `convolution` | held out | 0.064 | -0.015 |
| `persistence` | either | 4.4e-16 | 1.1e-16 |
| `reference` | either | not tested, 0 steps | |

Trained at 64 by 64 and evaluated at 128 by 128, the operator disagrees with itself by two
per cent over eight steps and running at the finer grid costs it nothing. The convolutional
network disagrees by 64 per cent, twenty eight times worse, and pays 0.233 of accuracy for
the change. Its weights are a stencil, and a stencil means a different thing when the
spacing halves; the operator's weights are indexed by wavenumber and mean the same thing on
both grids. That is the claim the phase set out to test, and it holds.

Three qualifications, because the number is easy to overread.

**The reference solver cannot be tested at all.** It is constructed for one grid and
refuses a state of another shape, so the suite records zero steps for it rather than a
score. That is a fact about the solver worth reading in the table: the resolution
independence a neural operator claims is not something the thing it imitates has.

**A degradation of zero is not the same as a faithful one.** Ground truth is stored at 64
by 64, and the two grids genuinely evolve apart: running the exact solver at 128 by 128
from the refined initial condition and coarsening back differs from the coarse ground truth
by 0.31 after eight steps on the test split, unchanged at four times the substeps, so it is
resolution rather than time stepping. A predictor that reproduced the fine scale physics
would therefore have to score a degradation near that. The operator scores zero because it
is smooth and inaccurate, and its two paths are about equally wrong. Consistency is the
number that means what it says; degradation is the number a user feels.

**The held out regime is easier on both.** The shear layer is more regular than decaying
turbulence, so the coarse and fine grids agree to 3e-5 there rather than 0.31, and both
models' inconsistencies fall by an order of magnitude. A resolution claim measured only on
the smooth regime would look far better than it is.

The honest summary of the phase is that the spectral structure bought exactly one thing,
and it is not accuracy. Without the convolutional control at matched parameter count there
would have been no way to know that, and without the resolution metric there would have
been no way to see what was bought instead.

## Reporting

Render a run, list the history and compare runs:

```sh
uv run nnp report render  --config configs/nbody.yaml
uv run nnp report list    --root runs --predictor reference
uv run nnp report compare <baseline-run-id> <run-id> --threshold 0.05
```

`eval run` writes a run record beside its result file: the resolved configuration, the run
identifier, the commit, the machine, the library versions, the timings, every metric output
and the paths to the rest. `report render` turns that record into a Markdown report and a
single HTML file with the plots embedded as data URIs, so one file can be sent to someone
with nothing beside it. Both come from one document built once, and rendering the same
record twice gives byte identical output.

| Config | Evaluation | Rendering | Plots | HTML report |
|---|---|---|---|---|
| `configs/nbody.yaml` | 8 s | 5 s | 18 | 1.2 MB |
| `configs/fluid.yaml` | 32 s | 5 s | 20 | 6.1 MB |

Each run gets a directory under `runs/` holding `record.json`, the result file, the states
kept for the qualitative plot, `plots/`, `report.md` and `report.html`. None of it is
committed. The record carries a schema version and an old one is upgraded rather than
refused, because a run whose numbers can no longer be read is a run that never happened.

Four figures per split. Error against horizon, drift of every declared invariant against
horizon with the declared tolerance shaded, the spread of the final error across
trajectories so a mean cannot hide one bad initial condition, and the predicted state
beside the true one at four horizons. The last of these knows no more physics than the
metrics do: it draws a field of two component vectors in the plane and a two dimensional
field as an image, choosing from the shape of the array rather than from the name of the
system.

Every number in a report is named, given units and explained in one line, and the direction
that counts as an improvement is stated. A metric that produces a scalar nobody has
explained stops the report rather than printing it bare. Comparison uses those directions:
it reports an improvement, a regression or no change per scalar, declines to judge the ones
whose ideal is neither extreme, and flags regressions above a threshold.

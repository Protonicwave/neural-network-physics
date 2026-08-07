# Results

Every number here comes from a run record under `runs/`, and the run identifier is given
beside it. Run records are not committed, because they are generated: the identifier is a
hash of the resolved configuration and the seed, so the same configuration and seed
reproduce the same identifier. `docs/reproduction.md` is the path from a clean checkout to
these numbers.

Read the limitations before the results. They are first because they are what the results
mean.

## Limitations

**One machine, eight CPU cores, no CUDA.** Every timing in this document was measured on
one Windows laptop with eight logical cores. Absolute timings do not transfer, and the
speed section documents a factor of two of drift on that machine alone.

**The systems are small.** N-body is 32 point masses in the two training regimes and 4 in
the held out one. The fluid is a 64 by 64 grid on a periodic square, evaluated at 128 by
128 only for the resolution metric. Neither is a size at which a surrogate would be
reached for in practice.

**The datasets are small.** 192 N-body trajectories of 256 states and 96 fluid
trajectories of 64 states, 31 MB and 128 MB on disk. Held out regimes are a third of each
but reach no training split.

**The horizons are short, and the models stop being physical well inside them.** Measured
at a tenth of the state size, the N-body graph network is worth using for 0.0375 of
simulated time, which is 3.75 stored steps out of 255. The fluid convolutional network
reaches 0.575, which is 11.5 stored steps out of 63. Past those the error is larger than
the signal, and both fluid operators diverge outright rather than degrading.

**No surrogate here is established as faster than the solver at matched accuracy.** That
is the headline result and it is negative. One case, the fluid operator, is undecided
rather than lost.

**The third significant figure of a chaotic error is a property of the machine.** The
N-body graph network's worst error over 255 steps reads 2.143 on eight threads
(`f68ffffd42abb2f4`) and 1.523 on one (`c8630755200dd028`), same weights and same initial
conditions. Threading changes the order the sums inside the network accumulate in, and the
cluster amplifies that round off. Evaluation tables here are eight threads and speed
tables are one; they are internally consistent and must not be read across.

**The eight repeat benchmark medians are not reproducible from a retained artefact.** Each
benchmark invocation overwrites `benchmark.json` in its run directory, so of the eight
repeats behind the speed tables only the last survives. The speed section gives both the
phase 09 median and the retained repeat, and says which is which.

**The diagnosis agent scores were not produced by the shipped command.** No API credential
was available, so each context was answered by a separate Claude instance and scored by
the same code. The ranking is honest; the cost per diagnosis was not measured at all.
`docs/results/diagnosis.md` states this at the top.

## What did not work

This section is worth more than the successes, so it is not compressed.

**The neural operator lost to its own control.** Phase 08 expected a spectral architecture
to beat a convolutional one on a fluid. At a parameter count matched to eight parts in ten
thousand, 149,001 against 148,889, and on identical training settings, the convolutional
network beat it on one step error, on rollout error, on every invariant, on equivariance
and on distributional drift. The operator did not clear persistence on one step error at
all. Its only win was resolution consistency, and that win is real and large.

**The multilayer perceptron never beat persistence on one step error.** 0.349 against
0.342 (`9cbf17e8fc998e3a` and `f68ffffd42abb2f4`). No learning rate between 0.0002 and
0.002 and no window stride between one and eight changed it. Its training loss fell every
epoch while its validation rollout worsened from the second, so selecting on a rollout
selects an almost untrained model. The two measures disagree because the training loss
normalises by the spread of the whole training split, where the collapsing phase
dominates, and the suite normalises by the size of the state at each step, where every
phase counts equally.

**Neither surrogate generalises to its held out regime.** The N-body graph network scores
1.11 on one step error against persistence's 0.042, and all four rollouts diverge between
steps 153 and 159. The fluid operator diverges on all four held out rollouts by step 28.
Only the fluid convolutional network survives its held out regime, and it is still beaten
by persistence there on one step error, 0.013 against 0.005.

**The ensemble did not rescue a model that had failed.** On the N-body held out regime the
single network diverges on 4 of 4 rollouts and so does the ensemble, a few steps later.
Averaging four paths that have left the physical manifold gives a path that is not on it
either.

**Calibration can be bought with vagueness, and on the held out regimes it was.** The
fluid ensemble's mean stated spread is 109 times the size of the true state on the test
split and 159 times on the held out one; the N-body ensemble reads 168 times on its held
out regime. Coverage of 0.96 next to a sharpness of 159 is a predictor saying the answer
might be anywhere. Sharpness is reported beside every coverage number for that reason.

**The uncertainty warns in time on one system and not the other.** The fluid ensemble's
spread crosses the trust threshold 0.4 of simulated time before its error does. The N-body
ensemble crosses 0.088 after. A surrogate that notices it has gone wrong only once it has
gone wrong cannot be used to decide when to fall back to the solver.

**Benchmark stability was the phase 09 criterion and it was mostly not met.** A timing
whose interquartile range exceeds a tenth of its median is marked unstable. In the
retained N-body repeat, 4 of 6 points are unstable; in the retained fluid repeat, all 4
are. The conclusions drawn are only those that survive the spread.

**Two numbers in the previous README were wrong, and the records say so.** The N-body
graph network was reported as completing 3 of 4 held out rollouts; the record
`f68ffffd42abb2f4` shows 0 of 4, all diverged. The ensemble was reported as diverging
sooner than the single network; the records show it diverging later, at steps 154 to 163
against 153 to 159. Both are corrected here.

**No metric was cut for being unable to fail.** All seven shipped metrics have sentinel
tests asserting both what they catch and what they do not, in `tests/evals/test_sentinels.py`
and the per metric tests beside it. Two metrics decline to score rather than flatter a
predictor: `calibration` reports zero steps for a predictor that states no uncertainty, and
`resolution_generalisation` reports zero steps for a system that declares no refinement and
for the reference solver, which is built for one grid and refuses another shape.

## The N-body system

Gravitational N-body under a softened Newtonian potential, integrated with velocity
Verlet. Trained on `cold_collapse` and `virialised_cluster`, both 32 bodies; `hierarchical_pair`,
4 bodies in two tight binaries, is held out entirely.

Dataset `54c793471686905f`: 192 trajectories, 256 states each, stored step 0.01, solver
step 0.001, 31 MB over 12 shards.

### Accuracy against horizon

Run `f68ffffd42abb2f4`, the graph network of 62,849 parameters, 40 epochs, 1,328 s of
training and 12 s of evaluation on eight threads. The perceptron column is run
`9cbf17e8fc998e3a`, 139,904 parameters. Both score 255 steps from 4 initial conditions. A
horizon is in simulated time, so 0.0375 is 3.75 stored steps.

| Predictor | one step error | horizon at 10% error | error at 255 steps | worst invariant violation | run |
|---|---|---|---|---|---|
| `reference` | 0 | never exceeded | 0 | 0 | `f68ffffd42abb2f4` |
| `persistence` | 0.342 | 0.0125 | 1.038 | 0.1 | `f68ffffd42abb2f4` |
| `mlp` | 0.349 | 0.0125 | 2.127 | 3.3e13 | `9cbf17e8fc998e3a` |
| `graph` | **0.053** | **0.0375** | 2.079 | 4.0e13 | `f68ffffd42abb2f4` |

The graph network clears persistence in distribution: six times lower one step error and
three times longer before the error reaches a tenth of the signal. Its error at 255 steps
is worse than persistence, and that is not a contradiction. Persistence saturates, because
once the states are uncorrelated it cannot get worse. A surrogate that has gone unstable
can. The horizon is the fair comparison and the final error is the honest warning, which is
why both are reported.

The graph network predicts the acceleration and lets velocity Verlet do the time stepping,
so the approximation is confined to the half that is not already known exactly. Translation,
rotation and Galilean invariance hold by construction: only pair distances and masses reach
any layer, and the output is a sum of learned scalars times unit vectors.

### Invariant behaviour

Energy is declared approximately conserved at a relative tolerance of 1e-06, linear
momentum exactly at 1e-12 and angular momentum exactly at 1e-10. Violation is the drift in
excess of the reference solver's own, in units of the declared tolerance, so a violation of
1 is at tolerance.

| Predictor | energy violation | linear momentum violation | angular momentum violation |
|---|---|---|---|
| `reference` | 0 | 0 | 0 |
| `persistence` | 0 | 0 | 0 |
| `graph` | 279 | 4.0e13 | 8.0e11 |

Two readings matter here. `persistence` conserves every N-body invariant perfectly and the
metric rates it flawless, which is correct and is why one step error alone means little:
the metric is measuring conservation, not accuracy. The graph network's momentum violations
look enormous because the allowance they are measured against is round off. The allowance
is the larger of the declared tolerance and ten times the reference solver's own drift, and
the solver conserves linear momentum to 5e-16, so the model's drift of 0.21 comes out at
4e13 times what is allowed. The violation is a ratio against the best the solver manages,
not against anything a user would notice.

The `energy_injection` fixture makes the same point from the other side. In run
`f68ffffd42abb2f4` it scores a one step error of 0.00058, roughly a thousandth of
persistence's, and `invariant_drift` still rates it 1.45e9 times outside tolerance.

### Equivariance

Run `f68ffffd42abb2f4` reports a worst symmetry violation of 0.0355 for the graph network
on the test split and 2.5e-07 on the held out one, over a 32 step rollout under three
declared symmetries. Per step the model is equivariant to floating point round off by
construction, and `tests/models/graph/test_model.py` asserts that. The reference solver, in
double precision, reads 1.5e-12 over the same rollout on the same split. This model works in
single precision, where round off is nine orders of magnitude larger, and a chaotic 32 body
cluster amplifies it. The held out regime, which is far more regular, stays at 2.5e-07,
which a broken symmetry would not explain.

### The held out regime gap

| Predictor | one step error | horizon at 10% error | error at end | rollouts completed |
|---|---|---|---|---|
| `persistence` | 0.042 | 0.03 | 0.823 | 4 of 4 |
| `graph` | 1.111 | 0.01 | 634.7 | **0 of 4**, diverged at steps 153, 154, 158, 159 |
| `mlp` | not applicable | | | 0 of 4, refused |

Run `f68ffffd42abb2f4` for the graph network and `9cbf17e8fc998e3a` for the perceptron.

The graph network does not generalise to the held out regime at all. Two separated
timescales is a configuration class no training regime contained, and nothing in the
training numbers predicted it. Reporting it is the whole reason a regime is held out.

The perceptron does not produce a number there because it cannot run: it sees the whole
flattened state, so it is tied to the exact number of bodies it trained on. Its record
carries the reason, `built for field 'mass' of shape (32,) and cannot take (4,)`, rather
than a score. The graph network runs on 4 bodies because message passing does not care how
many there are, which is a property of the architecture rather than of the training.

### Speed at matched accuracy

The solver has one knob, the substeps it folds into a stored interval, and `benchmark`
walks every count that divides the dataset's. A surrogate is one more point on the same
axes and the speedup is read off the crossing. Accuracy here is the worst normalised error
over the horizon rather than the error at the end, since a rollout that went wrong and came
back is not a rollout that stayed right.

One thread, because both solvers are single threaded whatever the setting, the fluid
through `numpy.fft` and the N-body through `numpy.einsum`. Anything higher gives the
network more of the machine than the thing it is compared against.

Retained repeat `c8630755200dd028`, 15 trials after 5 warmups, 255 steps from 4 initial
conditions. The phase 09 column is the median over eight repeats of the whole benchmark,
and those eight were not retained.

| Predictor | worst error | ms per step, retained | spread, retained | ms per step, median of 8 | spread over 8 |
|---|---|---|---|---|---|
| solver, 1 substep | 1.597 | 0.073 | 11% | 0.071 | 14% |
| solver, 2 substeps | 1.259 | 0.138 | 3% | 0.142 | 140% |
| solver, 5 substeps | 1.212 | 0.536 | 21% | 0.685 | 40% |
| solver, 10 substeps | 0 | 1.163 | 26% | 1.420 | 38% |
| `graph` | 1.523 | 2.388 | 27% | 2.877 | 60% |
| `ensemble` of four | **1.243** | 9.025 | 7% | 10.69 | 63% |

The graph network is matched by the solver at 2 substeps and runs at 0.058 of its speed in
the retained repeat, 0.056 as the median of eight with a range of 0.043 to 0.107. The
ensemble is matched at 5 substeps and runs at 0.059, median 0.062. Neither pays for itself
in any repeat, and neither is near enough that the machine's noise could be the reason.

The N-body ladder collapses in the direction that removes the competition. Over 255 steps
of a chaotic cluster the solver at 5 substeps has already reached an error of 1.212 against
the ground truth it generated at 10. There is no cheap accurate setting to compete with,
and no setting inaccurate enough to match a surrogate either.

### Break even

From `c8630755200dd028`. The data cost is the intervals the dataset holds times what the
solver costs for one of them at this thread count, rather than the wall clock generation
happened to take under an unknown load.

| Predictor | training | data | saving per rollout | break even |
|---|---|---|---|---|
| `graph` | 1,328 s | 57 s | -0.574 s | never |
| `ensemble` | 5,297 s | 57 s | -2.165 s | never |

A negative saving never repays anything, whatever the rollout count. For these datasets
the training dominates and the generation is a rounding error: 57 seconds against 1,328.

### Calibration and what the ensemble buys

Four models of the same configuration, differing only in initialisation and in the order
they saw their data. The member index shifts every seed except the dataset's, and a test
asserts the dataset identifier does not move; members trained on different data would be
measuring the data rather than the initialisation. Member zero is the plain run.

Members and their best validation rollout errors: `c8630755200dd028` 0.0785,
`09836a02939895eb` 0.0801, `50ce773bb22f7d76` 0.0726, `5460001c0e5c0ef3` 0.0669. They did
land in different minima.

Every member keeps its own trajectory rather than being reset to the mean each step,
because averaging and feeding back gives a spread that measures one step of disagreement
and never grows. The ensemble's prediction is the mean of the members, so the trajectory
scored is the one the spread describes.

| Split | Predictor | one step error | error at end | rollouts completed | run |
|---|---|---|---|---|---|
| test | `graph` | 0.053 | 2.079 | 4 of 4 | `f68ffffd42abb2f4` |
| test | `ensemble` | **0.044** | **1.304** | 4 of 4 | `c8630755200dd028` |
| held out | `graph` | 1.111 | 634.7 | 0 of 4, at steps 153 to 159 | `f68ffffd42abb2f4` |
| held out | `ensemble` | 1.331 | 659.2 | 0 of 4, at steps 154 to 163 | `c8630755200dd028` |

Averaging helps where the models were working. On the test split the ensemble improves
every accuracy number, and its final error of 1.304 is a quarter above persistence's 1.038
rather than twice it. On the held out regime it changes nothing worth having: four models
that all diverge produce a mean that also diverges, a handful of steps later.

Calibration, run `c8630755200dd028`, test split. Coverage is the fraction of the truth that
fell inside one stated standard deviation, and a correctly sized claim delivers 0.683.
Sharpness is the mean stated spread relative to the size of the true state, and low is
sharp.

| Predictor | calibration error | coverage | spread against error | sharpness | warning lead |
|---|---|---|---|---|---|
| `overconfident` | 0.498 | 0.003 | 0.965 | 0.0011 | no warning |
| `calibrated` | 0.296 | 0.276 | 0.971 | 0.115 | -0.905 |
| `ensemble` | **0.135** | **0.494** | 0.954 | 0.607 | -0.088 |

The ensemble is the best calibrated predictor in the suite and it beats the fixture that is
telling the truth about its own noise. That is not a paradox: the honest fixture states the
deviation its noise accumulates to as a random walk, and neither system is a random walk.
Both amplify a perturbation faster, so a claim honest about the noise still understates the
error. The ensemble's spread is four models disagreeing, and the disagreement grows the way
the error does.

The warning lead is the result that matters and on this system it is negative. The error
crosses the trust threshold at a horizon of 0.055 and the spread only at 0.143, so the
ensemble warns 0.088 of simulated time too late.

The correlation between spread and error, 0.95 to 1.00 across every row, is the weaker
result. Both curves grow monotonically over a rollout, so a correlation this high mostly
says that neither went down.

## The fluid system

2D incompressible Navier-Stokes in vorticity form on a periodic square, solved
pseudo-spectrally on a 64 by 64 grid. Trained on `taylor_green` at Reynolds 100 and
`decaying_turbulence` at Reynolds 500; `shear_layer` at Reynolds 200 is held out.

Dataset `7466dc27083c6a0f`: 96 trajectories, 64 states each, stored step 0.05, solver step
0.005, 128 MB over 12 shards.

### Accuracy against horizon

Run `2ecd7cb6a640ed0e` is the neural operator, 149,001 parameters, and run
`d9241046329768bd` is the convolutional control, 148,889 parameters. The parameter counts
are matched to eight parts in ten thousand deliberately; without that the comparison would
be about capacity. Identical training settings, 63 steps from 4 initial conditions.

| Predictor | one step error | horizon at 10% error | error at 63 steps | worst invariant violation | rollouts completed | run |
|---|---|---|---|---|---|---|
| `reference` | 0 | never exceeded | 0 | 0 | 4 of 4 | `2ecd7cb6a640ed0e` |
| `persistence` | 0.174 | 0.625 | 1.801 | 6.5e7 | 4 of 4 | `2ecd7cb6a640ed0e` |
| `operator` | 0.171 | 0.625 | 936.6 | 2.3e14 | 2 of 4, diverged at step 52 | `2ecd7cb6a640ed0e` |
| `convolution` | **0.087** | 0.575 | **0.876** | **1.1e8** | 4 of 4 | `d9241046329768bd` |

The neural operator loses to the convolutional baseline, and phase 08 expected it to win.
It does not clear persistence on one step error at all, 0.171 against 0.174, and its rollout
diverges where persistence merely saturates.

The convolutional network is the first surrogate in this repository to beat persistence on
both horizon and final error. Twice as accurate over one step, and a final error of 0.876
where persistence reaches 1.801, so it is still tracking the flow at the end of the rollout
rather than failing to get worse.

Training cost, from the epoch timings in each record: the operator 2,010 s over 39 epochs,
the convolutional network 5,286 s over 40, which is 2.6 times as much at the same parameter
count. A stencil costs nine multiplies per grid point where a one by one convolution costs
one. The convolution's record sums to 40,658 s because the run was interrupted and resumed
and the machine was suspended during two epochs, which took 21,636 s and 13,736 s of wall
clock and say nothing about the model; the 5,286 s figure excludes exactly those two.

### Invariant behaviour

Energy is declared decaying and enstrophy decaying, so holding either still is as
unphysical as adding to it. This is why `persistence` violates fluid invariants where it
conserved every N-body one: run `d9241046329768bd` rates it 6.5e7, entirely as excess.

Run `d9241046329768bd`, convolution, test split:

| Invariant | model drift | solver's own drift | most above truth | most below truth | violation |
|---|---|---|---|---|---|
| enstrophy | 28.81 | 29.04 | 0.538 | 7.711 | 4.0e6 |
| energy | 0.513 | 0.234 | 0.566 | 0.006 | 1.1e8 |

The model reproduces the viscous decay of enstrophy almost exactly, 28.81 against the
solver's 29.04. Energy does not fare as well: it moves roughly twice as far as the true
dynamics, and the direction separates it. The predicted energy stands as much as 0.566
above the true value at some point in the rollout and only 0.006 below it. The model adds
energy the physics did not supply. That is why the invariant violation reads 1.1e8 for a
model whose error curve looks healthy, and it is the failure mode an error curve alone
would hide.

### The held out regime gap

The shear layer at Reynolds 200, a structure neither training regime contained.

| Predictor | one step error | error at 63 steps | rollouts completed | run |
|---|---|---|---|---|
| `persistence` | 0.005 | 0.305 | 4 of 4 | `2ecd7cb6a640ed0e` |
| `operator` | 0.147 | 656.0 | 0 of 4, all diverged by step 28 | `2ecd7cb6a640ed0e` |
| `convolution` | **0.013** | 0.942 | 4 of 4 | `d9241046329768bd` |

The convolutional network survives the held out regime and the operator does not. Neither
beats persistence there, because the shear layer evolves slowly enough over one stored
interval that returning the input is a good answer.

### Resolution generalisation

This is the one place the operator wins, and it wins by the margin the design predicts.
The metric refines the initial condition, rolls the predictor forward on the finer grid,
coarsens the result back, and reports both how far the predictor drifted from its own
answer (consistency) and how much accuracy the change cost (degradation). Trained at 64 by
64, evaluated at 128 by 128, over 8 steps.

| Predictor | split | inconsistency | degradation | run |
|---|---|---|---|---|
| `operator` | test | **0.022** | **-0.0003** | `2ecd7cb6a640ed0e` |
| `convolution` | test | 0.635 | 0.233 | `d9241046329768bd` |
| `operator` | held out | **0.00024** | **-8.6e-07** | `2ecd7cb6a640ed0e` |
| `convolution` | held out | 0.064 | -0.015 | `d9241046329768bd` |
| `persistence` | test | 4.4e-16 | 1.1e-16 | either |
| `reference` | either | not tested, 0 steps for the refinement | | either |

The operator disagrees with itself by two per cent over eight steps and running at the
finer grid costs it nothing. The convolutional network disagrees by 64 per cent, twenty
eight times worse, and pays 0.233 of accuracy. Its weights are a stencil, and a stencil
means a different thing when the spacing halves; the operator's weights are indexed by
wavenumber and mean the same thing on both grids.

Three qualifications, because the number is easy to overread.

**The reference solver cannot be tested at all.** It is constructed for one grid and
refuses a state of another shape, so the refinement records zero steps for it rather than a
score. The resolution independence a neural operator claims is not something the thing it
imitates has.

**A degradation of zero is not the same as a faithful one.** Ground truth is stored at 64
by 64, and the two grids genuinely evolve apart: running the exact solver at 128 by 128
from the refined initial condition and coarsening back differs from the coarse ground truth
by 0.31 after eight steps on the test split, unchanged at four times the substeps, so it is
resolution rather than time stepping. A predictor reproducing the fine scale physics would
have to score a degradation near that. The operator scores zero because it is smooth and
inaccurate, and its two paths are about equally wrong. Consistency means what it says;
degradation is what a user feels.

**The held out regime is easier on both.** The shear layer is more regular than decaying
turbulence, so the coarse and fine grids agree to 3e-05 there rather than 0.31, and both
models' inconsistencies fall by an order of magnitude. A resolution claim measured only on
the smooth regime would look far better than it is.

The honest summary is that the spectral structure bought exactly one thing and it is not
accuracy. Without the convolutional control at matched parameter count there would have
been no way to know that, and without the resolution metric no way to see what was bought
instead.

### Speed at matched accuracy

Retained repeat `d3a72cf54859602c`, one thread, 15 trials after 5 warmups, 63 steps from 4
initial conditions.

| Predictor | worst error | ms per step, retained | spread, retained | ms per step, median of 8 | spread over 8 |
|---|---|---|---|---|---|
| solver, 1 and 2 substeps | cannot run | | | | |
| solver, 5 substeps | 7.2e-05 | 9.603 | 18% | 7.457 | 40% |
| solver, 10 substeps | 0 | 17.46 | 14% | 17.52 | 46% |
| `operator` | 936.6 | **7.218** | 23% | **6.162** | 50% |
| `ensemble` of four | 978.6 | 24.81 | 15% | 23.33 | 46% |

**The fluid operator is the one genuinely undecided case, and no speedup was established.**
Its speedup against the solver at 5 substeps is 1.33 in the retained repeat and 1.23 as the
median of eight, with a range of 0.81 to 1.63. It was faster in six repeats and slower in
two. A number whose spread straddles one is not a measurement of a speedup, and quoting the
1.23 without the range would be the exact error this section exists to avoid. The ensemble
is at 0.387 retained and 0.315 as the median, which is not close.

**The solver's own ladder is where the interesting result is.** Halving the substeps from
ten to five costs an accuracy of 7.2e-05 over 63 steps, which is nothing, and saves 45 per
cent of the wall clock in the retained repeat. The dataset's ten substeps are twice what
this horizon needs, so any surrogate competes against a solver that can be run at half the
advertised price before it starts. A benchmark against default settings silently skips
that comparison.

Below five substeps the fluid solver does not run at all: one and two are several times
past the stability limit and cannot take a step. They are named and left out of the ladder,
because a setting that does not run is not a slower setting, and putting an accuracy
against it would be inventing one.

### Break even

From `d3a72cf54859602c`.

| Predictor | training | data | saving per rollout | break even |
|---|---|---|---|---|
| `operator` | 2,010 s | 106 s | 0.150 s | 14,079 rollouts |
| `ensemble` | 7,644 s | 106 s | -0.958 s | never |

The one positive figure carries the caveat above. It comes from an accounting in which the
operator was faster, and the same accounting over the repeats where it was faster at all
gives 9,700 to 26,700 rollouts. Since the speedup is not established, neither is the break
even count. The honest summary is that the fluid operator would need tens of thousands of
rollouts to repay its training even on the assumption that it saves anything.

### Calibration and what the ensemble buys

Members and their best validation rollout errors: `d3a72cf54859602c` 0.5762,
`62bc4f82c144c032` 0.5694, `af9bc06a246bb76a` 0.5483, `e2bd31ead7a06f4c` 0.5130.

| Split | Predictor | one step error | error at end | rollouts completed | run |
|---|---|---|---|---|---|
| test | `operator` | 0.171 | 936.6 | 2 of 4, at step 52 | `2ecd7cb6a640ed0e` |
| test | `ensemble` | 0.167 | 978.6 | 2 of 4, at steps 45 and 50 | `d3a72cf54859602c` |
| held out | `operator` | 0.147 | 656.0 | 0 of 4, at step 28 | `2ecd7cb6a640ed0e` |
| held out | `ensemble` | 0.148 | 740.7 | 0 of 4, at step 33 | `d3a72cf54859602c` |

Averaging four diverging models gives a diverging mean. Nothing here is worth having.

Calibration, run `d3a72cf54859602c`, test split:

| Predictor | calibration error | coverage | spread against error | sharpness | warning lead |
|---|---|---|---|---|---|
| `overconfident` | 0.492 | 0.011 | 0.959 | 0.0005 | no warning |
| `calibrated` | 0.282 | 0.614 | 0.964 | 0.054 | no warning |
| `ensemble` | **0.090** | **0.721** | 0.998 | 109.2 | **+0.4** |

**This is the practically useful result the phase set out to find.** The ensemble's spread
crosses the trust threshold at a horizon of 1.0 and its error crosses at 1.4, so it warns
0.4 of simulated time before the prediction stops being worth using. That is enough to
decide when to fall back to the solver.

It is bought with vagueness. A sharpness of 109 means the stated spread is 109 times the
size of the true state, so the warning is honest about being uncertain rather than precise
about how uncertain. On the held out split the ensemble reads a coverage of 0.955 against a
sharpness of 159, and that row is vagueness rather than a success.

## The diagnosis agent

`nnp diagnose` reads two run records, reduces them to a summary and asks a model what
changed and why. It returns the metrics that regressed, a ranked list of candidate causes
with a confidence for each, and one suggested next check. It modifies nothing and runs
nothing: it reads and explains.

The deliverable is the scored fault injection, because an agent that produces a fluent
explanation of a run nobody broke cannot be told apart from one that produces a fluent
explanation of anything.

Seven faults, one per cause, injected one at a time into copies of the same known good run.
What was broken is written down before anything is asked. `configs/faults.yaml` is the same
pipeline in miniature, dataset `7bc543499b385a4d`, baseline run `9e286fe7985da402`, a graph
network of 9,697 parameters over 12 epochs, and the whole suite of three generated datasets
and eight trained models runs in about seventy seconds.

| Fault | Cause | Visible in the configuration | Run |
|---|---|---|---|
| `wrong_normalisation` | statistics that do not describe the data | no | `9e286fe7985da402` |
| `broken_symmetry` | a declared symmetry applied after every step | no | `9e286fe7985da402` |
| `no_optimiser_state` | resumed from a checkpoint with the moments discarded | no | `9e286fe7985da402` |
| `high_learning_rate` | two hundred times the configured rate | yes | `a3ccf12383392733` |
| `wrong_regime` | trained and held out regimes swapped | yes | `67a5d102a7278afe` |
| `no_curriculum` | the rollout curriculum reduced to a single step | yes | `5b2997ede7faef3a` |
| `unstable_integrator` | one solver substep per stored interval | yes | `e02021ba4d8ef896` |

Three of the faults leave the configuration untouched and change only the run identifier's
inputs that are not configuration, so they share the baseline identifier; their run
directories are distinguished by the fault name.

The split is the interesting part. Four are visible in a configuration diff and could be
named by reading two lines, so they test whether a diagnoser reads the evidence at all. The
other three appear nowhere in the configuration and leave a trace only in the numbers.

### The scores

Both diagnosers choose from the same twelve cause labels, five of which no fault uses;
without those five the list would be the answer key. The rule based diagnoser names
whichever metric regressed most and maps it to the cause that metric is usually about, from
a table written knowing which faults were coming. It is the strongest trivial diagnoser
rather than a straw one.

| Diagnoser | top 1 | top 3 | cause named at all | mean rank when named |
|---|---|---|---|---|
| agent | **86%** | **100%** | 7 of 7 | 1.29 |
| rule based | 14% | 43% | 4 of 7 | 2.50 |

Full per fault tables are in `docs/results/diagnosis.md` and the scored data in
`docs/results/fault-scores.json`.

The agent beats the baseline, and it is worth being precise about how much of that is
skill. Four of the seven are a configuration diff away, and naming them is reading
comprehension rather than diagnosis. The result that is not is `wrong_normalisation`, where
nothing in the configuration differs and the training numbers get an order of magnitude
better while every metric in physical units gets worse. The agent named it first, from the
observation that one step error rose by the same 32 per cent on position and on velocity,
which is a scale factor rather than a learning failure.

The one it got wrong is the most interesting row. On `no_optimiser_state` it ranked the
true cause third, behind `random_seed` and `no_regression`, reasoning that the two loss
curves start at the same value, end within half a per cent of each other and show no
discontinuity where the resume happened. That is correct. Losing the Adam moments for five
epochs of a twelve epoch run of a 9,697 parameter model on this dataset does almost
nothing, and the honest reading is not that the agent failed to spot the fault but that the
fault is barely there. A scored suite in which every answer was findable would be measuring
the suite rather than the diagnoser.

### What the scores do not say

**The vocabulary is closed.** Choosing from twelve labels is an easier job than writing a
paragraph, and it is the choice that makes an exact score possible at all.

**The runs are small.** The baseline model is weak, which cuts both ways: some faults are
harder to see against a weak baseline and none are easier.

**The agent card was not produced by `nnp diagnose score`.** No API credential was
available, so each rendered context was given separately to a Claude instance with no
access to this repository and its answer scored by the same code through `RecordedClient`.
No tokens were counted, so there is no cost per diagnosis, and the strict tool schema was
enforced by this package afterwards rather than by the API. Two of the seven answers
carried extra keys the API would have rejected. The ranking is measured honestly; the cost
is not measured at all.

### One bug found before the agent was asked anything

The excessive learning rate makes training diverge, and a diverged run's loss is not a
finite number. Pydantic writes such a number as `null`, which then fails to read back as a
float, so the run that most needed explaining was the one whose record could not be opened.
A loss that went to infinity is a measurement rather than a value to clean up, so the fix is
to serialise it. `tests/reporting/test_record.py` pins the round trip.

## Run index

| Run | What it holds |
|---|---|
| `f68ffffd42abb2f4` | N-body graph network, 62,849 parameters, 40 epochs, the default run |
| `9cbf17e8fc998e3a` | N-body perceptron, 139,904 parameters, the comparison |
| `4a54adc0fa01835b` | N-body graph network, 24 epochs, an earlier stopping rule |
| `c8630755200dd028` | N-body ensemble member zero, its evaluation and its benchmark |
| `09836a02939895eb`, `50ce773bb22f7d76`, `5460001c0e5c0ef3` | N-body ensemble members one to three |
| `2ecd7cb6a640ed0e` | Fluid neural operator, 149,001 parameters, evaluated with the resolution metric |
| `d9241046329768bd` | Fluid convolutional control, 148,889 parameters |
| `d3a72cf54859602c` | Fluid ensemble member zero, its evaluation and its benchmark |
| `62bc4f82c144c032`, `af9bc06a246bb76a`, `e2bd31ead7a06f4c` | Fluid ensemble members one to three |
| `9e286fe7985da402` | Fault suite baseline, and the three faults invisible in configuration |
| `a3ccf12383392733`, `67a5d102a7278afe`, `5b2997ede7faef3a`, `e02021ba4d8ef896` | The four faults visible in configuration |

Datasets: `54c793471686905f` N-body, `7466dc27083c6a0f` fluid, `7bc543499b385a4d` the fault
suite.

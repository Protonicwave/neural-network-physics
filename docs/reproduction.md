# Reproduction

One path from a clean checkout to the numbers in `docs/results.md`. Every step states its
measured runtime on the target machine, eight logical CPU cores and no CUDA.

Timings on this machine drift. Verifying the N-body dataset took 49 s once and 4 s twice
either side of it, on the same data, because the first run followed a 31 MB write. Treat
every figure below as the right order of magnitude rather than a stopwatch.

The whole path is about five and a half hours, almost all of it training. Stages 1, 2 and 7
take under four minutes together and check most of the machinery, so run those first if you
only want to know that the repository works.

## Stage 0: install

```sh
uv sync
uv run nnp --version
```

Network bound, a few minutes. `uv` pins the CPU build of PyTorch through an explicit index,
so this does not pull the CUDA wheel.

## Stage 1: data, about 95 s

```sh
uv run nnp data generate --config configs/nbody.yaml   # 38 s
uv run nnp data generate --config configs/fluid.yaml   # 49 s
uv run nnp data stats    --config configs/nbody.yaml   # 3 s
uv run nnp data stats    --config configs/fluid.yaml   # 5 s
```

`generate` writes one HDF5 shard per group of trajectories and a JSON manifest recording the
system, its parameters, the seed, both time intervals, the splits and a hash of every shard.
`stats` fits normalisation statistics to the training split alone.

The dataset identifier is printed and must match, because everything downstream is keyed to
it:

| Config | Dataset | Trajectories | On disk |
|---|---|---|---|
| `configs/nbody.yaml` | `54c793471686905f` | 192, 64 per regime | 31 MB |
| `configs/fluid.yaml` | `7466dc27083c6a0f` | 96, 32 per regime | 128 MB |

Generating twice from the same configuration and seed gives the same identifier whatever the
worker count. That was checked while writing this document: both datasets were regenerated
over the top of themselves and both identifiers were unchanged.

## Stage 2: check the data, about 10 s

```sh
uv run nnp data verify --config configs/nbody.yaml   # 4 s
uv run nnp data verify --config configs/fluid.yaml   # 6 s
```

`verify` re-hashes every shard and re-derives a sample of trajectories from their recorded
seeds, so it catches both a file that has changed and data that never matched the
configuration.

## Stage 3: the N-body surrogate, about 23 min

```sh
uv run nnp train --config configs/nbody.yaml
```

1,328 s of training over 40 epochs for a graph network of 62,849 parameters, then 12 s of
evaluation against the reference solver and the broken baselines, on the test split and on
the held out regime. Produces run `f68ffffd42abb2f4` and the first N-body table in
`docs/results.md`.

The comparison model is the same file with the model overridden, about 6 s of training
because it stops early:

```sh
NNP_MODEL__NAME=mlp NNP_MODEL__HYPERPARAMETERS='{}' uv run nnp train --config configs/nbody.yaml
```

That produces run `9cbf17e8fc998e3a`.

## Stage 4: the fluid surrogates, about 2 hours

```sh
uv run nnp train --config configs/fluid.yaml                              # 2,010 s
NNP_MODEL__NAME=convolution uv run nnp train --config configs/fluid.yaml  # 5,286 s
```

The neural operator, 149,001 parameters, and the convolutional control at 148,889. The
parameter counts are matched deliberately; without that the comparison would be about
capacity. Runs `2ecd7cb6a640ed0e` and `d9241046329768bd`.

The convolutional network takes 2.6 times as long for the same parameter count, because a
stencil costs nine multiplies per grid point where a one by one convolution costs one. If
the run is interrupted and resumed, exclude the suspended epochs from any timing claim: the
record for `d9241046329768bd` sums to 40,658 s because two of its epochs span a machine
suspension.

## Stage 5: the ensembles, about 2 hours 40 min

```sh
uv run nnp ensemble train --config configs/nbody.yaml   # 3,969 s for members one to three
uv run nnp ensemble run   --config configs/nbody.yaml   # 30 s
uv run nnp ensemble train --config configs/fluid.yaml   # 5,634 s for members one to three
uv run nnp ensemble run   --config configs/fluid.yaml   # 68 s
```

Four models per system, differing only in initialisation and in the order they saw their
data. Member zero is the plain run from stages 3 and 4, so a configuration already trained
is not trained again, and the times above are the three remaining members. `ensemble run`
scores them together as one predictor and produces the calibration tables.

## Stage 6: speed, about 10 min

```sh
uv run nnp benchmark --config configs/nbody.yaml --threads 1 --trials 15 --warmup 5 --steps 255
uv run nnp benchmark --config configs/fluid.yaml --threads 1 --trials 15 --warmup 5 --steps 63
```

One thread, because both solvers are single threaded whatever the setting is. Solver only,
this is 8 s for N-body; with the surrogates and the ensemble it is a few minutes per system.

Two warnings. **This command overwrites `benchmark.json` in the run directory it writes
to**, so a previous repeat is lost rather than kept beside the new one. That is how the
eight repeats behind the phase 09 medians came to be unrecoverable, and it happened again
while this document was being written. **Timings will not match.** The absolute numbers
drifted by a factor of two across one session on the machine they were measured on, so
reproduce the ordering and the conclusions rather than the milliseconds.

## Stage 7: reports and the fault suite, about 70 s

```sh
uv run nnp report render --run f68ffffd42abb2f4 --root runs   # 6 s, 20 plots, 1.3 MB HTML
uv run nnp report render --run 2ecd7cb6a640ed0e --root runs   # 5 s, 20 plots, 6.1 MB HTML
uv run nnp diagnose score --config configs/faults.yaml --rule-based --rerun   # 57 s
```

`report render` turns a run record into a Markdown report and one self contained HTML file
with the plots embedded as data URIs. Rendering the same record twice gives byte identical
output.

`diagnose score` generates three small datasets, trains eight models, injects each fault and
scores the answers. The rule based column reproduces exactly: top 1 of 14 per cent and top 3
of 43 per cent, which is what it printed when this document was written. Pass `--agent`
instead of `--rule-based` to score the agent, which needs an `ANTHROPIC_API_KEY`. Without
one the shipped `docs/results/diagnosis.md` keeps the agent card that was produced by hand,
and says so at the top.

Writing to `docs/results` is the default and it overwrites the committed table. Pass
`-o <dir>` to score somewhere else.

## What will not reproduce exactly

**Timings.** Stated above, and the reason the speed section quotes medians and spreads.

**The third significant figure of a chaotic error.** Evaluating the same N-body weights on
eight threads and on one gives a worst error over 255 steps of 2.143 and 1.523. Threading
changes the order the sums accumulate in and the cluster amplifies it. Evaluation tables in
`docs/results.md` are eight threads and speed tables are one.

**The agent column of the fault scores.** It was produced without an API credential, by
giving each context to a separate Claude instance and scoring the answer with the same code.
Running `nnp diagnose score --agent` with a credential replaces that card with a measured
one, including the cost per diagnosis that the hand produced card does not have.

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
| `configs/fluid.yaml` | 6 | 63 | 4 per split | 37 s |

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

`one_step_error`, `rollout_error`, `invariant_drift`, `symmetry_violation` and
`distribution_drift`. Every one of them has a sentinel test asserting both what it catches
and what it does not, because a metric that flags everything measures nothing. Three
results worth stating, all asserted in `tests/evals/test_sentinels.py`:

- The energy injecting predictor is a thousand times more accurate than persistence over
  one step, and `invariant_drift` still rates it a billion times outside the tolerance the
  system declared. One step error on its own is close to meaningless.
- `persistence` conserves every N-body invariant perfectly, so `invariant_drift` rates it
  flawless. In the viscous fluid it does not, because energy there is declared decaying and
  holding it still is as unphysical as adding it.
- The predictor that rotates the world each step conserves every invariant and is perfectly
  equivariant under rotation. Only the symmetries it does not commute with reveal it, which
  is why every declared symmetry is tested and the worst is reported.

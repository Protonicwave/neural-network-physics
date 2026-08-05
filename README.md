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
| `configs/nbody.yaml` | 192, 64 per regime | 256 | 0.01 | 0.001 | 30 MB | 56 s |
| `configs/fluid.yaml` | 96, 32 per regime | 64 | 0.05 | 0.005 | 122 MB | 89 s |

The solver step is smaller than the stored one because stability and accuracy demand it,
and the surrogate is not obliged to inherit that: the ratio is where its speedup comes
from, so both intervals are recorded.

Each system trains on two regimes and holds a third out entirely. N-body trains on
`cold_collapse` and `virialised_cluster` and holds out `hierarchical_pair`. The fluid
trains at Reynolds 100 and 500 and holds out the shear layer at 200. Held out data reaches
no training split, and normalisation statistics are fitted to the training split alone.

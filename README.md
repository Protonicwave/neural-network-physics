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

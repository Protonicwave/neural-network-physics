# neural-network-physics

Neural surrogates for physical simulation, with an evaluation harness that shows
where a surrogate can be trusted and where it cannot.

Two systems are supported: gravitational N-body and 2D incompressible fluid flow.
The evaluation harness is system agnostic and must stay that way.

## Working on this repo

Work is organised into phases. Each phase is a single session and has a brief in
`docs/plan/`. Read `docs/plan/README.md` first, then the brief for the phase you
are on. Do not start a phase before the ones it depends on are merged.

## Writing style

Applies to all committed text: code comments, docstrings, README, docs, commit
messages, pull request descriptions.

- UK English. Use artefact, behaviour, normalisation, optimise, analyse, modelling.
- No em-dashes and no en-dashes used as punctuation. Use commas, colons or full stops.
- Succinct. Cut any sentence that does not add information.
- Plain English. No marketing tone, no hype, no exclamation marks.
- Comments explain why, not what. Do not narrate the code.
- No emoji.

## Engineering standards

- Python 3.12. Dependencies and tasks are managed with `uv`.
- `src/` layout. The package is `nnphysics`.
- Type hints everywhere. `mypy --strict` must pass with no ignores outside
  `pyproject.toml`, and any ignore there needs a one line reason.
- `ruff check` and `ruff format` must pass.
- Public functions and classes have docstrings. Private helpers need one only if
  the reason for their existence is not obvious.
- Pure functions where practical. Side effects live at the edges, in the CLI and
  in IO modules.
- Dependency direction is one way: `core` depends on nothing internal, everything
  else depends inwards. Never import a concrete system or model from `core`.
- No global mutable state. Configuration is passed in, not reached for.
- Randomness is explicit. Every stochastic function takes a seed or a generator.
  Nothing calls a global random function.
- Fail fast and loudly. Validate inputs at boundaries, raise specific exceptions,
  never silently coerce or swallow.

## Testing

- `pytest`. Tests mirror the source tree under `tests/`.
- Numerical code is tested against analytic solutions where one exists, and
  against convergence order where one does not.
- Every metric in the evaluation harness has a sentinel test: a deliberately
  broken predictor that the metric must catch. A metric that cannot fail is not
  a metric.
- Tests are deterministic. Seed everything. No test may depend on wall clock time
  or network access.
- Slow tests are marked `slow` and excluded from the default run.

## Performance

- The target machine is CPU only, 8 logical cores, no CUDA. Default configurations
  must run there in reasonable time.
- Vectorise with NumPy or PyTorch. Do not write Python loops over particles, grid
  cells or time steps in any hot path.
- Optimise only after profiling, and record the measurement in the commit message.

## Commits

- Conventional commits, for example `feat(evals): add invariant drift metric`.
- One logical change per commit.
- Never commit data, checkpoints, run artefacts or plots. They are generated.

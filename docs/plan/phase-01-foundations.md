# Phase 01: Foundations

**Objective.** A repository that lints, type checks, tests and runs a CLI, with the
core protocols in place and no physics yet.

**Depends on.** Nothing.
**Branch.** `phase-01-foundations`

## Why this phase exists

Every later phase inherits these standards. Getting them wrong here means fixing
them eleven times later. Nothing in this phase does anything useful, and that is
correct. It is scaffolding.

## Build

1. **Tooling.** Install `uv`. Create `pyproject.toml` with project metadata, the
   `src/` layout, pinned Python 3.12, and dependency groups for runtime and dev.
   Runtime: numpy, scipy, torch, h5py, pydantic, pyyaml, typer, matplotlib.
   Dev: pytest, pytest-cov, mypy, ruff, pre-commit.
2. **Quality gates.** Ruff configuration with a strict but not absurd rule set,
   line length 100. mypy in strict mode. A pre-commit configuration running ruff
   format, ruff check and mypy on changed files.
3. **CI.** One GitHub Actions workflow on push and pull request: install with uv,
   ruff check, ruff format check, mypy, pytest excluding slow and integration.
   Cache the uv environment. Fail on any warning that matters.
4. **Core types.** In `core`, define the state and trajectory containers, the
   dataclasses describing array shapes and units, and the error hierarchy. One
   base exception for the package, with specific subclasses for configuration,
   validation and numerical failure.
5. **Core protocols.** Define System, Invariant, Symmetry, Predictor and Metric as
   described in the architecture document. Protocols only, no implementations.
6. **Registry.** A small generic registry that maps names to implementations, used
   later for systems, models and metrics. It must fail loudly on duplicate
   registration and on lookup of an unknown name, and it must be able to list what
   it holds.
7. **Config.** Pydantic v2 base models for a run configuration, plus a loader that
   reads YAML, applies environment overrides, validates, and returns a frozen
   object. Include the deterministic run id derivation described in the
   architecture document.
8. **Seeding.** A single module owning random state. It produces independent
   generators for NumPy and PyTorch from one root seed, and it has a function that
   makes a session deterministic.
9. **CLI.** A Typer application exposing `nnp` with placeholder subcommands for
   generate, train, evaluate, report and diagnose. Each prints what it will do and
   exits cleanly. Add `--version`.
10. **Repository furniture.** `.gitignore` covering data, checkpoints, runs, plots
    and the usual Python noise. An MIT licence. A README stub stating what the
    project is in five lines.

## Definition of done

- `uv sync` succeeds from a clean checkout.
- `uv run nnp --help` lists every subcommand, and `nnp --version` works.
- `ruff check`, `ruff format --check` and `mypy --strict src tests` all pass clean.
- `pytest` passes, covering: registry duplicate and missing name behaviour, config
  validation rejecting a bad file, run id stability for the same inputs and change
  for different ones, and seeding reproducibility.
- CI is green on the pull request.

## Out of scope

No physics, no data, no models, no metrics. Protocols must not be shaped around a
specific system. If you find yourself writing anything about gravity or vorticity,
stop.

## Notes for the session

Test the protocols by writing a throwaway trivial implementation in the test file
only. This proves the interfaces are implementable without committing a fake
system to the source tree.

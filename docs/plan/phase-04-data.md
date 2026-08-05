# Phase 04: Data pipeline

**Objective.** Reproducible trajectory generation, storage and loading, with splits
that make honest generalisation claims possible.

**Depends on.** Phases 02 and 03.
**Branch.** `phase-04-data`

## Why this phase exists

Most published surrogate results are optimistic because of how the data was split.
Getting this right is the difference between a project that measures something and
one that measures itself.

## Build

1. **Generation.** A generator that takes a system name, a regime, a count, a seed
   and a trajectory specification, and produces trajectories using the system's
   reference solver. Parallel across processes, with the number of workers
   configurable and defaulting to cores minus one. Each trajectory gets its own
   derived seed so results do not depend on worker scheduling.
2. **Sub-sampling.** Solvers need a small internal step for stability. The stored
   trajectory is sampled at a coarser interval, which is the interval the surrogate
   learns. Store both intervals in the metadata. This distinction matters later for
   the speedup measurement.
3. **Storage.** One HDF5 file per shard, holding stacked trajectories with
   compression, plus a JSON manifest recording the system, regime, parameters,
   seeds, solver settings, both time intervals, the code version and a content hash.
   The manifest is the record of provenance. A dataset without one is invalid.
4. **Verification.** A command that re-derives a small sample from the recorded
   seeds and checks it matches the stored data. This catches silent corruption and
   proves reproducibility.
5. **Splits.** Train, validation and test drawn from the declared training regimes,
   split by trajectory and never by time step, so that no time step from a
   trajectory in training appears in test. Separately, one or more regimes held out
   entirely as the out of distribution set. Splits are deterministic from a seed and
   recorded in the manifest.
6. **Normalisation.** Statistics computed from the training split alone, stored as
   an artefact, and applied at load time. Fit and apply are separate operations.
7. **Dataset and loader.** A PyTorch dataset yielding either single step pairs or
   short sequences of a configurable length, since phase 07 needs both. Memory
   mapped reads so a dataset larger than RAM still works. Worker seeding handled
   explicitly.
8. **CLI.** Wire up `nnp data generate`, `nnp data verify` and `nnp data stats`.

## Definition of done

- Generating twice with the same seed produces identical content hashes.
- `nnp data verify` passes on a freshly generated dataset and fails on a file that
  has been altered by one byte. Test both directions.
- **Leakage test.** Asserts that normalisation statistics are unchanged when the
  validation and test data change. This is the test that proves no leakage, so make
  it explicit and name it clearly.
- **Split disjointness test.** No trajectory identifier appears in more than one
  split, and no held out regime appears in train, validation or test.
- A loader test confirming worker count does not change the epoch contents.
- Default configurations for both systems generate their datasets on 8 CPU cores in
  a time recorded in the pull request description.

## Out of scope

No models, no metrics, no training. Do not add data augmentation.

## Notes for the session

Decide dataset sizes by what fits the machine, not by ambition. A few hundred
trajectories per system is enough to demonstrate everything in this project. Record
the sizes in the manifest and the README.

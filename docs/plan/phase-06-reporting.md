# Phase 06: Reporting

**Objective.** Turn evaluation results into run records and readable reports, and
compare any two runs.

**Depends on.** Phase 05.
**Branch.** `phase-06-reporting`

## Why this phase exists

Results that are not comparable across runs cannot support a claim of improvement.
The comparison view is also what the phase 10 agent reads, so its structure matters
more than its appearance.

## Build

1. **Run record.** A single serialised object holding the resolved configuration,
   the run id, code version, environment summary, timings, all metric outputs and
   paths to artefacts. Versioned with a schema version field so old runs remain
   readable.
2. **Artefact layout.** A predictable directory per run under a runs root, holding
   the record, plots and any checkpoints referenced. Never committed.
3. **Plots.** Rollout error against horizon, invariant drift against horizon, error
   distribution across trajectories, and one qualitative side by side of predicted
   against true state at several horizons. For the fluid this is a vorticity field,
   for N-body it is trajectories in the plane. Consistent styling defined once.
4. **Rendering.** A Markdown report and a self contained HTML report with images
   embedded as data URIs, so a single file can be shared. Both are generated from
   the same record by the same code path.
5. **Comparison.** Given two or more run ids, produce a table of metric deltas with
   the direction of improvement stated per metric, and overlay plots. Flag
   regressions above a configurable threshold.
6. **Index.** A command listing all runs with their key metrics, sorted, so the
   history is visible without opening files.
7. **CLI.** `nnp report render`, `nnp report compare`, `nnp report list`.

## Definition of done

- A run record round trips through serialisation without loss, including for a
  record written under a previous schema version. Test the migration path.
- Rendering is deterministic. The same record produces byte identical Markdown.
- The HTML report opens with no network access and no external files. Test that the
  output contains no external references.
- Comparison correctly identifies an improvement, a regression and a no change
  case. Use three synthetic records as fixtures.
- Every metric added in phase 05 appears in the report with units and a one line
  plain English explanation of what it means. A number without an explanation is
  not finished work.

## Out of scope

No web server, no interactive dashboard, no experiment tracking service.

## Notes for the session

Write the plain English explanations carefully. They are what a reader of the
repository sees first, and they are reused in the README in phase 11.

# Phase 10: Diagnosis agent

**Objective.** An agent that reads two run reports and explains what changed and
why, scored against faults whose true cause is known.

**Depends on.** Phase 09.
**Branch.** `phase-10-diagnosis-agent`

## Why this phase exists

Building an agent is common. Measuring whether it is right is not. The scored fault
injection is the deliverable here, not the agent.

## Build

1. **Client.** A thin wrapper over the Anthropic SDK. The model id, temperature and
   token limits come from configuration and are recorded in the run record. Retries
   with backoff, a request timeout, and a clear error if no API key is present. No
   key ever touches the repository or a log line.
2. **Context assembly.** Turn two run records into a compact structured summary:
   configuration differences, metric deltas, and the shape of the diagnostic curves
   reduced to a few numbers. Do not send raw arrays. The quality of this reduction
   determines the quality of the diagnosis, so it deserves the effort.
3. **Diagnosis.** Given the summary, the agent returns a structured result:
   the metrics that regressed, a ranked list of candidate causes with a confidence
   for each, and a suggested next check. Enforce the structure through the SDK's
   tool use so the output is validated rather than parsed out of prose.
4. **Fault injection.** A set of faults that can be applied to a known good
   configuration or model, each with a recorded true cause. Cover at least: a
   normalisation applied with the wrong statistics, a broken symmetry in the model,
   a learning rate far too high, training data drawn from the wrong regime, a
   rollout curriculum disabled, an integrator step size that breaks stability, and
   a checkpoint loaded without optimiser state.
5. **Scoring.** For each injected fault, run the pipeline, produce a report, ask the
   agent to diagnose it, and score whether the true cause appears in its ranked
   list and at what position. Report top one and top three accuracy across the fault
   set, with the per fault results shown.
6. **Baseline.** Score a trivial rule based diagnoser that simply names whichever
   metric regressed most. If the agent does not beat it, that is the finding and it
   goes in the results.
7. **CLI.** `nnp diagnose` for a single comparison, `nnp diagnose score` for the
   full fault suite.

## Definition of done

- Every unit test runs against a recorded fixture response, not the live API. The
  test suite must pass with no network access and no API key.
- One integration test hits the real API, marked `integration` and excluded from CI.
- The fault suite runs end to end and produces a scored table, committed as a
  results artefact.
- Top one and top three accuracy are reported honestly against the rule based
  baseline.
- Cost per diagnosis is measured and recorded.

## Out of scope

The agent does not modify code, run training, or take any action. It reads and
explains. Autonomy is not the point of this phase and adds risk without adding a
result.

## Notes for the session

Load the `claude-api` skill before writing any client code, and take model ids,
pricing and SDK usage from it rather than from memory. Do not hardcode a model id
in the source.

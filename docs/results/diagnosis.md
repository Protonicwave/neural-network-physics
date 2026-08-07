# Diagnosis agent: scored fault injection

System `nbody`, baseline run `9e286fe7985da402`, 7 faults, generated 2026-08-06T15:35:23+00:00.

Every fault is injected into a copy of the same known good run, the resulting record is compared against the baseline, and each diagnoser is asked to rank the causes. The true cause is known before the question is asked. Both diagnosers choose from the same twelve cause labels, five of which no fault uses.

## How this was produced

The seven fault runs, the baseline they are compared against and the `rule_based` card
were all produced by `nnp diagnose score --config configs/faults.yaml`, and reproducing
them needs nothing but this repository.

The `agent` card was not. No API credential was available on the machine this was run on,
so each rendered context was given, on its own and with no access to this repository, to a
separate Claude instance holding only that text and the cause vocabulary. Its answer went
back through the same validation and the same scoring code the agent path uses, via
`RecordedClient`. Three things follow, and they are the reasons this card is labelled
rather than presented as the measured result:

- It is not reproducible from this repository. `nnp diagnose score` with a credential is,
  and that is the command that should replace this card.
- No tokens were counted, so the cost per diagnosis is absent rather than zero. The brief
  asks for that number and this run does not have it.
- The strict tool schema was not enforced by the API, only by this package's own
  validation afterwards. Two of the seven answers carried extra keys the API would have
  rejected outright.

What the card does measure honestly is the ranking, because the true cause of every fault
was written down before any question was asked and the scoring code is the same for both
diagnosers.


## Accuracy

| Diagnoser | model | top 1 | top 3 | named at all | mean rank when named | cost per diagnosis |
|---|---|---|---|---|---|---|
| `agent` | claude-opus-5 via a separate instance, unmetered | 86% | 100% | 7 of 7 | 1.29 | not measured |
| `rule_based` | worst regressed metric | 14% | 43% | 4 of 7 | 2.50 | free |

## Per fault

### `agent`

| Fault | true cause | rank | first named | confidence in the truth |
|---|---|---|---|---|
| `wrong_normalisation` | `normalisation_statistics` | 1 | `normalisation_statistics` | 0.62 |
| `broken_symmetry` | `model_symmetry` | 1 | `model_symmetry` | 0.86 |
| `high_learning_rate` | `learning_rate` | 1 | `learning_rate` | 0.95 |
| `wrong_regime` | `training_regime` | 1 | `training_regime` | 0.88 |
| `no_curriculum` | `rollout_curriculum` | 1 | `rollout_curriculum` | 0.93 |
| `unstable_integrator` | `integrator_step_size` | 1 | `integrator_step_size` | 0.93 |
| `no_optimiser_state` | `optimiser_state` | 3 | `random_seed` | 0.12 |

### `rule_based`

| Fault | true cause | rank | first named | confidence in the truth |
|---|---|---|---|---|
| `wrong_normalisation` | `normalisation_statistics` | not named | `integrator_step_size` | 0.00 |
| `broken_symmetry` | `model_symmetry` | 3 | `integrator_step_size` | 0.00 |
| `high_learning_rate` | `learning_rate` | 4 | `integrator_step_size` | 0.00 |
| `wrong_regime` | `training_regime` | not named | `integrator_step_size` | 0.00 |
| `no_curriculum` | `rollout_curriculum` | 2 | `integrator_step_size` | 0.50 |
| `unstable_integrator` | `integrator_step_size` | 1 | `integrator_step_size` | 1.00 |
| `no_optimiser_state` | `optimiser_state` | not named | `integrator_step_size` | 0.00 |

## Cost

- `agent`: 0 input and 0 output tokens over the whole suite, not measured per diagnosis.
- `rule_based`: 0 input and 0 output tokens over the whole suite, free per diagnosis.


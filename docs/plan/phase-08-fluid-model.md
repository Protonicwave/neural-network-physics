# Phase 08: Fluid model

**Objective.** A neural operator for the fluid system, trained by the phase 07
training loop with no changes to that loop.

**Depends on.** Phase 07.
**Branch.** `phase-08-fluid-model`

## Why this phase exists

This phase is the test of the architecture. If the training loop, the data pipeline
and the evaluation harness all work on a completely different system and model
family without modification, the abstractions were correct. If they do not, the
repair belongs in the shared layer, not in a special case.

## Build

1. **Fourier neural operator.** Spectral convolution layers that multiply by learned
   weights on a truncated set of Fourier modes, with a pointwise channel mixing
   path alongside. Lifting and projection layers at the ends. Mode count and width
   configurable, defaulted to something that trains on CPU.
2. **Convolutional baseline.** A plain periodic padded convolutional network of
   similar parameter count. Without it there is no way to tell whether the spectral
   structure is doing the work or the parameter count is.
3. **Reuse.** Train both with the existing loop. Any change required to the training
   loop, dataset or harness must be a general improvement, not a fluid branch. If a
   branch on system type appears anywhere outside `systems` and `models`, it is a
   defect.
4. **Resolution generalisation.** Because a neural operator is resolution
   independent in principle, evaluate a model trained at 64 by 64 on data at 128 by
   128. This is a strong claim and worth testing honestly. Add it as a metric in the
   existing harness, not as a one off script.

## Definition of done

- The training loop, dataset and evaluation modules have no fluid specific code.
  Demonstrate this with a diff summary in the pull request description.
- The neural operator beats the convolutional baseline on rollout error at equal
  parameter count, or the result is reported plainly if it does not.
- Long rollouts are evaluated for enstrophy behaviour, and any unphysical energy
  growth is reported rather than hidden.
- The resolution generalisation result is reported with its degradation stated.
- The report from phase 06 renders for the fluid system with no changes.

## Out of scope

Three dimensional flow, turbulence closure modelling, adaptive meshes.

## Notes for the session

If the model is stable but the energy spectrum is wrong at high wavenumber, that is
the expected failure mode and it is worth a paragraph in the results rather than an
attempt to hide it.

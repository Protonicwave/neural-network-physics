# Phase 07: Models and training

**Objective.** A model interface, a reusable training loop, and the first learned
surrogate for the N-body system.

**Depends on.** Phase 06.
**Branch.** `phase-07-models-training`

## Why this phase exists

Only now, with a harness that is known to detect badness, is it worth training
something. Every result in this phase is measured by machinery that was written
before the model existed.

## Build

1. **Model interface.** A base class implementing the Predictor protocol, wrapping a
   PyTorch module, handling normalisation, device placement and saving. Models are
   registered by name and constructed from configuration.
2. **Learned baselines.** A model that predicts a constant zero update, and a small
   multilayer perceptron on flattened state. These are weak on purpose. They set the
   floor that any real model must clear, and they exercise the training loop
   cheaply.
3. **Graph network for N-body.** Particles as nodes, interactions as edges over a
   neighbourhood or fully connected for small N. Message passing predicting
   acceleration rather than next position, so the known integrator does the time
   stepping. This is the single most important design choice in the phase: predict
   the derivative, integrate with the symplectic scheme from phase 02. State the
   reason in the docstring.
4. **Training loop.** One loop, system agnostic, driven by configuration. Includes:
   - deterministic seeding of everything
   - gradient clipping and a cosine schedule with warmup
   - early stopping on a validation metric, not on training loss
   - checkpointing of best and last, with optimiser state, so runs resume exactly
   - structured logging of metrics per epoch to the run record from phase 06
   - a rollout curriculum, training first on one step then on progressively longer
     sequences with gradients truncated. This is what makes a surrogate stable over
     long horizons, and its absence is the usual reason surrogates diverge.
5. **Losses.** Normalised one step error, plus an optional multi step term over the
   curriculum window. Any physics informed penalty is optional and off by default,
   because a model that conserves energy only because it was told to is a weaker
   result than one that learns it.
6. **CLI.** `nnp train` taking a config and producing a run record.

## Definition of done

- Training is reproducible: same config and seed give identical final weights.
- Resuming from a checkpoint gives the same result as an uninterrupted run to the
  same epoch. Test this, it catches optimiser state bugs.
- The multilayer perceptron beats persistence on one step error, and the graph
  network beats the multilayer perceptron on rollout error. Record both.
- The graph network is evaluated by the full phase 05 suite, and the report includes
  the held out regime gap. Report the numbers even if they are poor. Poor numbers
  honestly reported are a result.
- Training the default N-body configuration completes on 8 CPU cores in a time
  recorded in the pull request description.

## Out of scope

No fluid model. No hyperparameter search beyond a small documented sweep. No
architecture zoo.

## Notes for the session

Expect the rollout curriculum to matter more than model capacity. If the model is
unstable, increase curriculum length before increasing parameters.

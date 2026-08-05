# Phase 02: N-body system

**Objective.** A gravitational N-body system implementing the System protocol, with
a solver whose correctness is proven against analytic results.

**Depends on.** Phase 01.
**Branch.** `phase-02-nbody`

## Why this phase exists

Everything downstream is built on this ground truth. A solver that quietly leaks
energy will later look like a model failure and cost days. Validate it now.

## Build

1. **State and parameters.** Positions, velocities and masses for N bodies in two
   dimensions. Two dimensions, not three, keeps plots readable and cost low. Regime
   parameters: number of bodies, mass distribution, virial ratio, softening length.
2. **Forces.** Vectorised pairwise gravitational acceleration with Plummer
   softening to avoid singular close encounters. No Python loop over pairs. Use the
   symmetry of the force matrix.
3. **Integrator.** Velocity Verlet, which is symplectic and therefore bounds energy
   error rather than letting it grow without limit. Provide a fixed step size and
   an explicit statement of the stability limit in the docstring. Also implement a
   fourth order Runge-Kutta integrator, not for production use but to demonstrate
   in tests that a non symplectic scheme drifts where the symplectic one does not.
4. **Invariants.** Total energy, linear momentum, angular momentum. Each declared
   through the Invariant protocol with a sensible tolerance.
5. **Symmetries.** Translation, rotation and Galilean boost, each expressed as a
   transformation on a state.
6. **Initial conditions.** Seeded generators for the named regimes, including a
   cold collapse, a virialised cluster and a hierarchical pair. Each takes a
   generator, never a global seed.
7. **Registration.** Register the system under the name `nbody`.

## Definition of done

Tests must include all of the following.

- **Two body circular orbit.** Radius and speed stay constant to tight tolerance
  over many orbits, and the measured period matches the analytic value.
- **Kepler ellipse.** A bound eccentric orbit conserves the Runge-Lenz vector
  direction, so the orbit does not precess beyond tolerance.
- **Convergence order.** Halving the time step reduces global error by the expected
  factor for each integrator. This is the test that catches an integrator that is
  subtly wrong but plausible looking.
- **Energy behaviour.** Over a long run, Verlet energy error stays bounded and
  oscillatory while Runge-Kutta drifts monotonically. Assert both.
- **Momentum.** Linear and angular momentum are conserved to machine precision in
  the absence of external forces.
- **Symmetry.** Rolling out a transformed initial condition gives the transformed
  rollout, for every declared symmetry.
- **Determinism.** The same seed gives bitwise identical trajectories.

## Out of scope

No data generation, no storage, no models. Three dimensions, collisions, mergers
and external potentials are all out of scope.

## Notes for the session

The convergence order test is the single most valuable test in this phase. Write it
first. Mark long integrations as `slow` so the default test run stays fast.

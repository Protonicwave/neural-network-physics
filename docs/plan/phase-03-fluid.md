# Phase 03: Fluid system

**Objective.** A 2D incompressible fluid system implementing the same System
protocol, with a pseudo-spectral solver validated against an analytic solution.

**Depends on.** Phase 01. Independent of phase 02, but merge after it.
**Branch.** `phase-03-fluid`

## Why this phase exists

The second system is what proves the abstraction. If implementing this forces a
change to the System protocol, that is a finding worth recording in the pull
request, and the protocol changes rather than the system bending around it.

## Build

1. **Formulation.** Two dimensional incompressible Navier-Stokes in the vorticity
   and streamfunction form on a doubly periodic square domain. Vorticity is the
   only state variable, which keeps incompressibility exact and the state small.
2. **Spatial discretisation.** Pseudo-spectral. Derivatives in Fourier space, the
   nonlinear term evaluated in physical space, with two thirds rule dealiasing.
   Default grid 64 by 64, which runs on CPU. The grid size is a parameter.
3. **Time stepping.** Explicit fourth order Runge-Kutta for the nonlinear term with
   an integrating factor for the viscous term, so viscosity does not set the step
   size. Include a CFL based step size check that raises rather than silently going
   unstable.
4. **Invariants.** Total energy and enstrophy. In the inviscid limit both are
   conserved. With viscosity both decay, and enstrophy decays faster. Declare them
   with the correct expectation: conserved when viscosity is zero, monotonically
   decreasing otherwise. A metric later reads this distinction, so state it
   honestly.
5. **Symmetries.** Translation on the periodic domain, and rotation by multiples of
   ninety degrees which maps the grid onto itself exactly.
6. **Initial conditions.** Seeded generators for named regimes: Taylor-Green
   vortex, decaying turbulence from a prescribed energy spectrum, and a shear layer.
   Regime parameter is the Reynolds number.
7. **Registration.** Register under the name `fluid`.

## Definition of done

- **Taylor-Green decay.** The analytic solution decays at a known exponential rate
  set by viscosity. The measured rate matches it to tight tolerance. This is the
  anchor test for this phase.
- **Spectral convergence.** Refining the grid on a smooth solution reduces error
  faster than any fixed polynomial order. Assert error falls by a large factor per
  refinement, well beyond second order.
- **Inviscid conservation.** With viscosity set to zero, energy and enstrophy are
  conserved to tolerance over a moderate run.
- **Viscous decay direction.** With viscosity above zero, energy decreases
  monotonically and enstrophy decreases at least as fast.
- **Dealiasing.** A test showing that without dealiasing a high wavenumber initial
  condition contaminates low wavenumbers, and that with it enabled this does not
  happen.
- **Symmetry and determinism.** As in phase 02.

## Out of scope

Three dimensions, forcing, boundaries other than periodic, compressibility. No
adaptive time stepping.

## Notes for the session

Watch for the two common errors in spectral codes: an incorrect Fourier
normalisation convention, which the Taylor-Green test will catch, and forgetting to
zero the mean mode of the streamfunction solve, which leaves an undetermined
constant. Both look fine until they do not.

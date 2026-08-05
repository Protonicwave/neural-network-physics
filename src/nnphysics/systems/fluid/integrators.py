"""The reference solver: fourth order Runge-Kutta with an integrating factor.

The viscous term is linear and diagonal in Fourier space, so it is solved exactly:
substituting `v = exp(nu k^2 t) omega` removes it from the equation, Runge-Kutta
advances what is left, and the substitution is undone. Viscosity therefore places no
limit on the step size at all, which matters because an explicit treatment of it would
demand `dt < dx^2 / nu` and make a fine grid unaffordable.

What does limit the step is advection, so every step checks the CFL condition and raises
rather than letting the solution go unstable. An unstable spectral solution does not
degrade gracefully: it looks plausible for a few steps and then overflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NumericalError, ValidationError

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, State
    from nnphysics.systems.fluid.dynamics import FluidDynamics
    from nnphysics.systems.fluid.grid import ComplexArray

__all__ = ["DEFAULT_CFL_NUMBER", "IntegratingFactorRK4"]

DEFAULT_CFL_NUMBER = 0.5
"""Half a cell per step. Fourth order Runge-Kutta is stable well beyond this, so the
margin is for accuracy rather than for survival."""


@dataclass(frozen=True, slots=True)
class IntegratingFactorRK4:
    """Integrating factor Runge-Kutta, the reference solver for this system.

    Attributes:
        dynamics: The vorticity equation being solved.
        dt: Step size.
        cfl_number: Largest fraction of a cell the fastest flow may cross in one step.
        half_decay: Viscous decay over half a step, cached because it depends only on
            the step size and the viscosity.
        full_decay: Viscous decay over a whole step.
    """

    dynamics: FluidDynamics
    dt: float
    cfl_number: float = DEFAULT_CFL_NUMBER
    half_decay: FloatArray = field(init=False, repr=False, compare=False)
    full_decay: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValidationError(f"step size must be positive, got {self.dt}")
        if self.cfl_number <= 0.0:
            raise ValidationError(f"CFL number must be positive, got {self.cfl_number}")
        exponent = -self.dynamics.viscosity * self.dynamics.operators.squared_wavenumber * self.dt
        object.__setattr__(self, "half_decay", np.exp(0.5 * exponent))
        object.__setattr__(self, "full_decay", np.exp(exponent))

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "fluid-ifrk4"

    def step(self, state: State) -> State:
        """Advance one step.

        Args:
            state: The current state.

        Returns:
            The state one step later.

        Raises:
            NumericalError: If the step exceeds the CFL limit, or the result is not
                finite.
            ValidationError: If the state does not match the solver's grid.
        """
        grid = self.dynamics.grid
        spectrum = grid.forward(grid.unpack(state))
        self._check_step_size(spectrum, state.time)

        # Stages of Runge-Kutta on the transformed variable, written back in terms of
        # the original one: each stage carries the viscous decay of the sub interval it
        # was evaluated over.
        half, full = self.half_decay, self.full_decay
        first = self.dt * self.dynamics.nonlinear(spectrum)
        second = self.dt * self.dynamics.nonlinear(half * (spectrum + 0.5 * first))
        third = self.dt * self.dynamics.nonlinear(half * spectrum + 0.5 * second)
        fourth = self.dt * self.dynamics.nonlinear(full * spectrum + half * third)
        advanced: ComplexArray = (
            full * spectrum + (full * first + 2.0 * half * (second + third) + fourth) / 6.0
        )

        vorticity = grid.inverse(advanced)
        if not np.isfinite(vorticity).all():
            raise NumericalError(
                f"the fluid solver produced a non finite field at t={state.time + self.dt}"
            )
        return grid.make_state(vorticity, time=state.time + self.dt)

    def _check_step_size(self, spectrum: ComplexArray, time: float) -> None:
        """Raise if the fastest flow crosses more than the permitted fraction of a cell.

        Raises:
            NumericalError: If the step size exceeds the advective limit.
        """
        speed = self.dynamics.maximum_speed(spectrum)
        if speed == 0.0:
            return
        limit = self.cfl_number * self.dynamics.grid.spacing / speed
        if self.dt > limit:
            raise NumericalError(
                f"step size {self.dt} exceeds the CFL limit {limit:.6g} at t={time}: "
                f"the flow reaches {speed:.6g} across cells of width "
                f"{self.dynamics.grid.spacing:.6g}"
            )

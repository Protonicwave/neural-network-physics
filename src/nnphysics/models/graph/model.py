"""The N-body surrogate: a learned acceleration inside the solver's own integrator.

This is the single most important design choice in the phase, so it is worth stating why
rather than only what.

**The network predicts a derivative, not a next state.** A model that maps a state to the
state one interval later has to learn two things at once: the force law, which is hard,
and time stepping, which is already known exactly. Learning both means the known half is
approximated, and the approximation is what a long rollout accumulates. Predicting the
acceleration and handing it to velocity Verlet leaves the model only the part nobody has
written down.

**The integrator is the symplectic one from phase 02.** Velocity Verlet has an error in
energy that oscillates within a bound set by the step size rather than accumulating, and
that bound survives the substitution of a learned acceleration for the true one: whatever
the network predicts, the update is still a symplectic map of the state, so the drift the
integrator contributes stays bounded and any drift that is measured belongs to the model.
A model that integrated with an explicit Euler step would drift for a reason that had
nothing to do with what it had learned, and the invariant metrics would be measuring the
integrator.

The step is written out here in single precision torch rather than reusing the solver
from phase 02, because it has to be differentiable and it has to run on a batch. A test
asserts the two agree when handed the same accelerations, which is what keeps this a
restatement of that scheme rather than a second one.

**One evaluation per step.** Velocity Verlet needs the acceleration at the state it just
produced, which is also the acceleration at the start of the next step. The carry the
model interface provides passes it forward, so a rollout costs one network evaluation per
step and not two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nnphysics.core.errors import ValidationError
from nnphysics.core.params import check_parameter_names, float_parameter, int_parameter
from nnphysics.models.base import Carry, ModelContext, ModelHyperparameters, SurrogateModel
from nnphysics.models.graph.network import InteractionNetwork
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GRAPH_NAME", "NBodyGraphModel", "build_graph"]

GRAPH_NAME = "graph"

ACCELERATION_KEY = "acceleration"
"""What the carry holds: the acceleration at the state the last step produced."""

_GRAPH_PARAMETERS = (
    "hidden",
    "rounds",
    "softening",
    "gravitational_constant",
    "physical_softening",
)
_DEFAULT_HIDDEN = 64
_DEFAULT_ROUNDS = 2
_DEFAULT_SOFTENING = 0.02
"""In units of the position scale. Comparable to the physical softening of the default
configuration once that is divided by the spread of a cluster."""

_DIMENSIONS = 2


class NBodyGraphModel(SurrogateModel):
    """Message passing for the acceleration, velocity Verlet for the time stepping.

    The model is specific to the N-body system in one respect only: it reads three field
    names and treats them as positions, velocities and masses. Everything else, including
    the number of bodies, is read from the data it is handed, so the same weights run on
    a cluster of thirty two and on the four body held out regime without change.

    Args:
        context: What the model is being built for.
        hidden: Width of every embedding and message.
        rounds: Message passing rounds.
        softening: Softening in units of the position scale, used by the network's
            distance features.
        gravitational_constant: Used only by the optional energy penalty.
        physical_softening: Softening of the true force law, in physical units, used only
            by the optional energy penalty.

    Raises:
        ValidationError: If the data does not carry the three N-body fields, if mass is
            not the static one, or if the positions are not two dimensional.
    """

    def __init__(  # noqa: PLR0913
        # The five settings a configuration may carry, plus the context. Grouping them
        # into an object would only move the same five somewhere else, and two of them
        # exist solely for a penalty that is off by default.
        self,
        context: ModelContext,
        *,
        hidden: int = _DEFAULT_HIDDEN,
        rounds: int = _DEFAULT_ROUNDS,
        softening: float = _DEFAULT_SOFTENING,
        gravitational_constant: float = 1.0,
        physical_softening: float = 0.0,
    ) -> None:
        super().__init__(GRAPH_NAME, context)
        _check_fields(context)
        self._hidden = hidden
        self._rounds = rounds
        self._softening = softening
        self._gravitational_constant = gravitational_constant
        self._physical_softening = physical_softening

        stats = context.normalisation.stats
        self._length = stats[POSITION_FIELD].scale
        self._mass = abs(stats[MASS_FIELD].mean) or stats[MASS_FIELD].scale
        # The acceleration that changes a velocity by one standard deviation in one
        # stored interval. Scaling the output this way makes the network's own number the
        # normalised velocity change it is being trained to produce, which is the
        # quantity the loss is measured in, so a typical target is near one.
        #
        # The obvious alternative, a velocity scale squared over a length, is the
        # acceleration the virial theorem gives for a cluster in equilibrium. It is a
        # hundred times too small here, because the interval a surrogate learns is ten
        # solver steps and the regimes it learns from spend much of their time collapsing
        # rather than in equilibrium. A network asked for an output of a hundred from
        # bounded features does not train: its gradients vanish and its loss stops at
        # what predicting nothing achieves.
        self._acceleration = stats[VELOCITY_FIELD].scale / context.dt
        self.network = InteractionNetwork(hidden, rounds, context.generator(), softening=softening)

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        """The settings this model was built with."""
        return {
            "hidden": self._hidden,
            "rounds": self._rounds,
            "softening": self._softening,
            "gravitational_constant": self._gravitational_constant,
            "physical_softening": self._physical_softening,
        }

    def accelerations(self, position: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """Predict the acceleration of every body, in physical units.

        Args:
            position: Positions, shape `(batch, bodies, 2)`.
            mass: Masses, shape `(batch, bodies)`.

        Returns:
            Accelerations, shape `(batch, bodies, 2)`.
        """
        scaled: torch.Tensor = self.network(position / self._length, mass / self._mass)
        return scaled * self._acceleration

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Take one velocity Verlet step with the learned acceleration.

        Args:
            fields: The batch, in physical units.
            carry: The acceleration at this state, if the previous step produced it.

        Returns:
            The new positions and velocities, and the acceleration at them.
        """
        position = fields[POSITION_FIELD]
        velocity = fields[VELOCITY_FIELD]
        mass = fields[MASS_FIELD]
        dt = self.dt

        current = (
            carry[ACCELERATION_KEY]
            if carry is not None and ACCELERATION_KEY in carry
            else self.accelerations(position, mass)
        )
        moved = position + velocity * dt + 0.5 * current * dt**2
        following = self.accelerations(moved, mass)
        sped = velocity + 0.5 * (current + following) * dt
        return (
            {POSITION_FIELD: moved, VELOCITY_FIELD: sped},
            {ACCELERATION_KEY: following},
        )

    def physics_penalty(
        self, fields: Mapping[str, torch.Tensor], predicted: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """The squared relative change in total energy over one step.

        Off by default. A model that conserves energy because the loss punished it for
        not doing so has been told the answer, and the invariant drift metric can no
        longer distinguish that from having learned it. This exists so the difference can
        be measured, not so it can be switched on and forgotten.

        The energy is the softened one the true dynamics conserve, so the two constants it
        needs are hyperparameters of this model. They are unused while the weight on this
        term is zero.

        Args:
            fields: The input states, in physical units.
            predicted: What the model produced from them, in physical units.

        Returns:
            A scalar, meaned over the batch.
        """
        mass = fields[MASS_FIELD]
        before = self._energy(fields[POSITION_FIELD], fields[VELOCITY_FIELD], mass)
        after = self._energy(predicted[POSITION_FIELD], predicted[VELOCITY_FIELD], mass)
        return ((after - before) / before.abs().clamp_min(_ENERGY_FLOOR)).pow(2).mean()

    def _energy(
        self, position: torch.Tensor, velocity: torch.Tensor, mass: torch.Tensor
    ) -> torch.Tensor:
        """Total energy of every state in a batch, under the softened potential."""
        kinetic = 0.5 * (mass * velocity.pow(2).sum(-1)).sum(-1)
        separation = position.unsqueeze(1) - position.unsqueeze(2)
        squared = separation.pow(2).sum(-1) + self._physical_softening**2
        inverse = squared.clamp_min(_DISTANCE_FLOOR).rsqrt()
        pairs = mass.unsqueeze(1) * mass.unsqueeze(2)
        diagonal = torch.eye(
            position.shape[1], dtype=position.dtype, device=position.device
        ).unsqueeze(0)
        potential = (
            -0.5
            * self._gravitational_constant
            * (pairs * inverse * (1.0 - diagonal)).sum(dim=(1, 2))
        )
        return kinetic + potential


_ENERGY_FLOOR = 1e-12
"""Keeps a configuration whose total energy passes through zero from dividing by it."""

_DISTANCE_FLOOR = 1e-24
"""Keeps an unsoftened coincident pair from producing an infinity in the penalty."""


def _check_fields(context: ModelContext) -> None:
    """Reject data this model cannot read as an N-body system."""
    expected = (MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD)
    if context.names != tuple(sorted(expected)):
        raise ValidationError(
            f"the graph model reads the N-body fields {sorted(expected)}, but the data "
            f"carries {list(context.names)}"
        )
    if context.static_fields != (MASS_FIELD,):
        raise ValidationError(
            f"the graph model carries mass and predicts the rest, but the data holds "
            f"{list(context.static_fields)} constant along a trajectory"
        )
    for name in (POSITION_FIELD, VELOCITY_FIELD):
        shape = context.field_shapes[name]
        if len(shape) != _DIMENSIONS or shape[-1] != _DIMENSIONS:
            raise ValidationError(
                f"the graph model needs {name!r} of shape (bodies, 2), got {shape}"
            )


def build_graph(context: ModelContext, hyperparameters: ModelHyperparameters) -> SurrogateModel:
    """Build the N-body graph network.

    Args:
        context: What the model is being built for.
        hyperparameters: Accepts `hidden`, `rounds`, `softening`,
            `gravitational_constant` and `physical_softening`.

    Returns:
        The model.

    Raises:
        ValidationError: If a hyperparameter is unknown or out of range, or the data is
            not an N-body dataset.
    """
    check_parameter_names(hyperparameters, _GRAPH_PARAMETERS, context=GRAPH_NAME)
    return NBodyGraphModel(
        context,
        hidden=int_parameter(hyperparameters, "hidden", _DEFAULT_HIDDEN, context=GRAPH_NAME),
        rounds=int_parameter(hyperparameters, "rounds", _DEFAULT_ROUNDS, context=GRAPH_NAME),
        softening=float_parameter(
            hyperparameters, "softening", _DEFAULT_SOFTENING, context=GRAPH_NAME
        ),
        gravitational_constant=float_parameter(
            hyperparameters, "gravitational_constant", 1.0, context=GRAPH_NAME
        ),
        physical_softening=float_parameter(
            hyperparameters, "physical_softening", 0.0, context=GRAPH_NAME
        ),
    )

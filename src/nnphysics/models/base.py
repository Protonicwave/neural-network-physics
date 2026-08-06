"""The model interface: a PyTorch module that is also a predictor.

A surrogate has to be two things at once. To the training loop it is a module with
parameters and a differentiable forward pass over batches. To the evaluation harness it
is a `Predictor`, advancing one state by one stored interval, indistinguishable from a
solver or from a baseline that is broken on purpose. This base class is the join, and it
owns the three things that would otherwise be reimplemented by every model.

**Normalisation.** The dataset is read in physical units and every model works in
physical units at its boundary, so a state that goes in and a state that comes out mean
what they say. Centring and scaling happen inside a model, on the inputs its network
actually sees and on the loss the loop computes. Statistics are fitted to the training
split alone, and they arrive here already fitted: nothing in this layer can refit them.

**Static fields.** Some fields are carried rather than predicted. Mass is constant in
every N-body trajectory, so a model that predicted it would be spending capacity to
reproduce its own input and would be scored for succeeding. Which fields those are is
measured from the training split and arrives in the context, so no model has to name a
field it happens to know about.

**The carry.** An integrator that needs a derivative at the state it just produced would
otherwise evaluate the network twice per step. `advance` may return anything it wants to
reuse, and both the batched unroll and the single stepping path thread it through. The
cache is keyed on object identity and a miss only costs an evaluation, so it can never
change an answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.core.registry import Registry
from nnphysics.core.seeding import torch_generator
from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "INIT_SEED_STREAM",
    "MODELS",
    "Carry",
    "ModelContext",
    "ModelFactory",
    "ModelHyperparameters",
    "SurrogateModel",
    "build_model",
    "load_model",
    "model_payload",
    "save_model",
]

INIT_SEED_STREAM = "model.init"
"""Stream every weight initialisation is drawn from. It depends on the run seed alone, so
adding a model to the registry cannot shift the weights another model starts from."""

CHECKPOINT_SCHEMA_VERSION = 1
"""Bumped when the shape of a checkpoint payload changes incompatibly."""

type Carry = Mapping[str, torch.Tensor]
"""Whatever a model wants to reuse from one step at the next one."""

type ModelHyperparameters = Mapping[str, object]
"""Model specific settings, as they arrive from configuration."""

type ModelFactory = Callable[[ModelContext, ModelHyperparameters], SurrogateModel]
"""Builds one model from the context and the hyperparameters a configuration carried."""

MODELS: Registry[ModelFactory] = Registry("model")
"""Every registered model factory. Importing `nnphysics.models` fills it."""


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Everything a model factory is allowed to know.

    A model is handed the shape of the data and the interval it must learn, and nothing
    that would tell it which system produced them. A model that needs more than this is
    a model for one system, and says so by reading the field names it expects.

    Attributes:
        field_shapes: Field name to the shape of one state of that field, taken from the
            dataset the model will be trained on.
        static_fields: Fields that do not change along a trajectory, measured from the
            training split. A model carries these rather than predicting them.
        normalisation: Statistics fitted to the training split.
        dt: The stored interval, which is the step the model learns to take.
        seed: Root seed of the run, for weight initialisation.
    """

    field_shapes: Mapping[str, tuple[int, ...]]
    static_fields: tuple[str, ...]
    normalisation: Normalisation
    dt: float
    seed: int

    def __post_init__(self) -> None:
        if not self.field_shapes:
            raise ValidationError("a model context must describe at least one field")
        if self.dt <= 0.0:
            raise ValidationError(f"the stored interval must be positive, got {self.dt}")
        missing = sorted(set(self.field_shapes) - set(self.normalisation.stats))
        if missing:
            raise ValidationError(f"no normalisation statistics for fields {missing}")
        unknown = sorted(set(self.static_fields) - set(self.field_shapes))
        if unknown:
            raise ValidationError(f"static fields {unknown} are not fields of the data")
        if not self.predicted_fields:
            raise ValidationError(
                f"every field {list(self.names)} is constant along a trajectory, so there "
                f"is nothing for a model to predict"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Field names, sorted, which is the order a flattening model concatenates in."""
        return tuple(sorted(self.field_shapes))

    @property
    def predicted_fields(self) -> tuple[str, ...]:
        """Fields a model must produce: every field that is not static, sorted."""
        return tuple(name for name in self.names if name not in set(self.static_fields))

    def generator(self) -> torch.Generator:
        """The generator every weight of this run is drawn from."""
        return torch_generator(self.seed, INIT_SEED_STREAM)


class SurrogateModel(nn.Module, ABC):
    """A learned predictor: a module to the training loop, a `Predictor` to the harness.

    Args:
        name: Registered model name, reported in every result file.
        context: The shapes, statistics and interval the model was built for. It also
            decides which fields are predicted and which are carried, because that is a
            fact about the data rather than a choice a model gets to make.
    """

    def __init__(self, name: str, context: ModelContext) -> None:
        super().__init__()
        self._name = name
        self._context = context
        self._predicted_fields = context.predicted_fields
        self._static_fields = tuple(sorted(context.static_fields))
        self._cached_output: State | None = None
        self._cached_carry: Carry | None = None

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return self._name

    @property
    def dt(self) -> float:
        """The stored interval this model advances by."""
        return self._context.dt

    @property
    def context(self) -> ModelContext:
        """What the model was built for."""
        return self._context

    @property
    def predicted_fields(self) -> tuple[str, ...]:
        """Fields the network produces."""
        return self._predicted_fields

    @property
    def static_fields(self) -> tuple[str, ...]:
        """Fields carried from the input unchanged."""
        return self._static_fields

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        """The settings this model was built with, as a checkpoint must record them."""
        return {}

    @property
    def n_parameters(self) -> int:
        """Trainable parameters, for the run record."""
        return sum(int(item.numel()) for item in self.parameters() if item.requires_grad)

    @abstractmethod
    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Advance a batch of states by one stored interval.

        Args:
            fields: Field name to a batched tensor in physical units, shape
                `(batch, *field_shape)`. Every declared field is present.
            carry: What the previous call returned, or `None` when there was none. A
                model that returns a carry must still be correct when handed `None`.

        Returns:
            The predicted fields, in physical units and the same shapes, and whatever
            should be carried to the next step.
        """

    def physics_penalty(
        self, fields: Mapping[str, torch.Tensor], predicted: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """A penalty on unphysical predictions, zero unless a model defines one.

        Off by default and weighted to zero by default. A model that conserves energy
        because the loss punished it for not doing so is a weaker result than one that
        learns to, so this exists to be measured against, not to be switched on quietly.

        Args:
            fields: The input states, in physical units.
            predicted: What the model produced from them, in physical units.

        Returns:
            A scalar, zero for a model that declares no penalty.
        """
        del fields, predicted
        return torch.zeros((), dtype=torch.float32)

    def normalise(self, fields: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Centre and scale fields with the training split's statistics.

        Args:
            fields: Field name to a tensor in physical units.

        Returns:
            The normalised tensors.

        Raises:
            ValidationError: If a field has no statistics.
        """
        return {
            name: (tensor - self._stats(name).mean) / self._stats(name).scale
            for name, tensor in fields.items()
        }

    def denormalise(self, fields: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Undo `normalise`.

        Args:
            fields: Field name to a normalised tensor.

        Returns:
            The tensors in physical units.

        Raises:
            ValidationError: If a field has no statistics.
        """
        return {
            name: tensor * self._stats(name).scale + self._stats(name).mean
            for name, tensor in fields.items()
        }

    def forward(self, fields: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """One step, predicted fields only.

        Args:
            fields: Field name to a batched tensor in physical units.

        Returns:
            The predicted fields one stored interval later.
        """
        predicted, _ = self.advance(fields, None)
        return predicted

    def unroll(self, fields: Mapping[str, torch.Tensor], steps: int) -> dict[str, torch.Tensor]:
        """Advance a batch repeatedly, feeding each prediction back in.

        The graph spans the whole window, so gradients reach every step of it. The window
        is what truncates them: the curriculum lengthens it stage by stage rather than
        backpropagating through a rollout of arbitrary length.

        Args:
            fields: Field name to a batched tensor in physical units.
            steps: Steps to take, at least one.

        Returns:
            Each predicted field stacked over the steps taken, shape
            `(batch, steps, *field_shape)`.

        Raises:
            ValidationError: If fewer than one step was asked for.
        """
        if steps < 1:
            raise ValidationError(f"an unroll must take at least one step, got {steps}")
        current = dict(fields)
        carry: Carry | None = None
        taken: list[dict[str, torch.Tensor]] = []
        for _ in range(steps):
            predicted, carry = self.advance(current, carry)
            taken.append(predicted)
            current = {name: current[name] for name in self._static_fields} | predicted
        return {
            name: torch.stack([entry[name] for entry in taken], dim=1)
            for name in self._predicted_fields
        }

    def step(self, state: State) -> State:
        """Advance one state by one stored interval.

        This is the `Predictor` side of the model, the only side the evaluation harness
        sees. A carry produced by the previous call is reused when this call is handed
        exactly the state that call returned, which is what a rollout does, so an
        integrator that needs a derivative at its own output pays for one network
        evaluation per step rather than two.

        Args:
            state: The current state, in physical units.

        Returns:
            The state one stored interval later.

        Raises:
            ValidationError: If the state does not carry the fields the model was built
                for.
        """
        batch = self._as_batch(state)
        carry = self._cached_carry if state is self._cached_output else None
        with torch.no_grad():
            predicted, carry = self.advance(batch, carry)
        fields = {name: state.fields[name] for name in self._static_fields}
        for name in self._predicted_fields:
            fields[name] = predicted[name].squeeze(0).detach().numpy().astype(np.float64)
        result = State(fields=fields, time=state.time + self.dt)
        self._cached_output = result
        self._cached_carry = carry
        return result

    def _stats(self, name: str) -> FieldStats:
        """Statistics of one field."""
        try:
            return self._context.normalisation.stats[name]
        except KeyError:
            raise ValidationError(
                f"model {self._name!r} has no normalisation statistics for field {name!r}"
            ) from None

    def _as_batch(self, state: State) -> dict[str, torch.Tensor]:
        """Read a state into single precision tensors with a leading batch axis."""
        missing = sorted(set(self._context.field_shapes) - set(state.fields))
        if missing:
            raise ValidationError(f"model {self._name!r} was handed a state missing {missing}")
        return {
            name: torch.from_numpy(
                np.ascontiguousarray(state.fields[name], dtype=np.float32)
            ).unsqueeze(0)
            for name in self._context.field_shapes
        }


def build_model(
    name: str, context: ModelContext, hyperparameters: ModelHyperparameters | None = None
) -> SurrogateModel:
    """Resolve a registered model name and build it.

    Args:
        name: Registered model name, for example `graph`.
        context: What the model is being built for.
        hyperparameters: Model specific settings, or `None` for the defaults.

    Returns:
        The model, in evaluation mode.

    Raises:
        UnknownNameError: If no model is registered under that name.
        ValidationError: If a hyperparameter is unknown or out of range.
    """
    model = MODELS.get(name)(context, {} if hyperparameters is None else hyperparameters)
    model.eval()
    return model


def model_payload(model: SurrogateModel) -> dict[str, Any]:
    """Everything a model must record about itself to be rebuilt later.

    The weights alone would not be enough. A model is only meaningful beside the
    normalisation it was trained with and the interval it was trained to take, so this
    carries both rather than trusting a configuration to still say the same thing when
    the file is read.

    A training checkpoint adds the optimiser to this and nothing else is different, which
    is what lets `load_model` read either file.

    Args:
        model: The model.

    Returns:
        A payload of primitives and tensors, safe to load with weights only.
    """
    context = model.context
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.name,
        "hyperparameters": dict(model.hyperparameters),
        "dt": context.dt,
        "seed": context.seed,
        "field_shapes": {name: list(shape) for name, shape in sorted(context.field_shapes.items())},
        "static_fields": list(context.static_fields),
        "normalisation": {
            name: {"mean": stats.mean, "std": stats.std, "count": stats.count}
            for name, stats in sorted(context.normalisation.stats.items())
        },
        "state_dict": model.state_dict(),
    }


def save_model(path: Path, model: SurrogateModel) -> None:
    """Write a model and everything needed to rebuild it.

    Args:
        path: File to write. Its parent must exist.
        model: The model.
    """
    torch.save(model_payload(model), path)


def load_model(path: Path) -> SurrogateModel:
    """Read a model back, rebuilt from what the checkpoint recorded.

    Args:
        path: File to read.

    Returns:
        The model, in evaluation mode.

    Raises:
        ConfigurationError: If the file is missing, is of another schema version, or is
            missing something the model cannot be rebuilt without.
        UnknownNameError: If the recorded model name is not registered.
    """
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ConfigurationError(f"cannot read model checkpoint {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"{path} is not a version {CHECKPOINT_SCHEMA_VERSION} model checkpoint"
        )
    try:
        context = ModelContext(
            field_shapes={
                str(name): tuple(int(extent) for extent in shape)
                for name, shape in payload["field_shapes"].items()
            },
            static_fields=tuple(str(name) for name in payload["static_fields"]),
            normalisation=Normalisation(
                {
                    str(name): FieldStats(
                        mean=float(entry["mean"]),
                        std=float(entry["std"]),
                        count=int(entry["count"]),
                    )
                    for name, entry in payload["normalisation"].items()
                }
            ),
            dt=float(payload["dt"]),
            seed=int(payload["seed"]),
        )
        model = build_model(str(payload["model"]), context, payload["hyperparameters"])
        model.load_state_dict(payload["state_dict"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} is a malformed model checkpoint: {error}") from error
    except RuntimeError as error:
        raise ConfigurationError(f"{path} does not fit the model it names: {error}") from error
    model.eval()
    return model

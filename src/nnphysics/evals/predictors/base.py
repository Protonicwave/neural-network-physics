"""Naming a predictor in configuration and building it.

A predictor specification is a name and optional parameters, `noise:scale=0.02`, so a
suite can be written down and re-run. Everything a factory might need arrives in a
context built by the runner: the reference solver already wrapped to the stored interval,
the system's declared state specification and symmetries, and a seed. A factory that
reaches for anything else would be reaching into a system, which this layer may not see.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.core.registry import Registry

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor, Symmetry
    from nnphysics.core.types import StateSpec

__all__ = [
    "PREDICTORS",
    "PredictorContext",
    "PredictorFactory",
    "PredictorParameters",
    "PredictorSpec",
    "build_predictor",
    "check_parameters",
    "parse_spec",
    "read_parameter",
]


@dataclass(frozen=True, slots=True)
class PredictorContext:
    """Everything a predictor factory is allowed to know.

    Attributes:
        reference: The system's own solver, already advancing by one stored interval per
            step, which is the interval every predictor in a suite is compared over.
        state_spec: The fields a state carries, with their declared dimensions.
        symmetries: Transformations the system declares its dynamics commute with.
        seed: Root seed of the run, for any factory that needs a generator.
        stream: Name of the seed stream this predictor's generator is drawn from.
    """

    reference: Predictor
    state_spec: StateSpec
    symmetries: tuple[Symmetry, ...]
    seed: int
    stream: str


type PredictorParameters = Mapping[str, float]
"""Parameters parsed out of a specification. Every one is a real number: a predictor that
needed a string would be selecting behaviour, which is what the name is for."""

type PredictorFactory = Callable[[PredictorContext, PredictorParameters], Predictor]
"""Builds one predictor from the context and the parameters its specification carried."""

PREDICTORS: Registry[PredictorFactory] = Registry("predictor")
"""Every registered predictor factory. Importing `nnphysics.evals` fills it."""


@dataclass(frozen=True, slots=True)
class PredictorSpec:
    """A parsed predictor specification.

    Attributes:
        name: Registered predictor name.
        parameters: Parameters given, in the order they were written.
        text: The specification as it was written, kept so a result file records exactly
            what was asked for rather than a re-rendering of it.
    """

    name: str
    parameters: PredictorParameters
    text: str


def parse_spec(text: str) -> PredictorSpec:
    """Parse a predictor specification.

    Args:
        text: `name` or `name:key=value,key=value`.

    Returns:
        The parsed specification.

    Raises:
        ValidationError: If the name is empty, a parameter is malformed, a parameter is
            given twice or a value is not a real number.
    """
    name, separator, tail = text.partition(":")
    name = name.strip()
    if not name:
        raise ValidationError(f"predictor specification {text!r} names no predictor")
    if not separator:
        return PredictorSpec(name, {}, text)

    parameters: dict[str, float] = {}
    for item in tail.split(","):
        key, assigned, value = item.partition("=")
        key = key.strip()
        if not assigned or not key:
            raise ValidationError(
                f"predictor specification {text!r} has a malformed parameter {item!r}, "
                f"expected key=value"
            )
        if key in parameters:
            raise ValidationError(f"predictor specification {text!r} sets {key!r} twice")
        try:
            parameters[key] = float(value)
        except ValueError as error:
            raise ValidationError(
                f"predictor specification {text!r} gives {key!r} the non numeric value {value!r}"
            ) from error
    return PredictorSpec(name, parameters, text)


def build_predictor(spec: PredictorSpec | str, context: PredictorContext) -> Predictor:
    """Resolve a predictor specification against the registry and build it.

    Args:
        spec: A parsed specification, or the text of one.
        context: What the factory is allowed to know.

    Returns:
        The predictor.

    Raises:
        UnknownNameError: If no predictor is registered under that name.
        ValidationError: If the specification is malformed or a parameter is rejected.
    """
    parsed = parse_spec(spec) if isinstance(spec, str) else spec
    return PREDICTORS.get(parsed.name)(context, parsed.parameters)


def read_parameter(
    parameters: PredictorParameters, name: str, default: float, *, context: str
) -> float:
    """Read one parameter of a predictor specification.

    Args:
        parameters: Parameters the specification carried.
        name: Parameter to read.
        default: Value used when the parameter is absent.
        context: Name of the predictor, used in the error message.

    Returns:
        The value.

    Raises:
        ValidationError: If the value is not finite.
    """
    value = parameters.get(name, default)
    if not math.isfinite(value):
        raise ValidationError(f"{context} parameter {name!r} must be finite, got {value}")
    return value


def check_parameters(
    parameters: PredictorParameters, allowed: tuple[str, ...], *, context: str
) -> None:
    """Reject parameters a predictor does not understand.

    A misspelled parameter that is silently ignored looks like a predictor that is not
    as broken as it was meant to be, which is the one thing a fixture must not be.

    Args:
        parameters: Parameters the specification carried.
        allowed: Every name the predictor accepts.
        context: Name of the predictor, used in the error message.

    Raises:
        ValidationError: If any parameter name is not allowed.
    """
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise ValidationError(
            f"{context} does not accept the parameters {unknown}, accepted: {sorted(allowed)}"
        )

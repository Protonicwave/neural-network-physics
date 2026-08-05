"""Reading loosely typed parameters at a boundary.

A system is configured by a mapping of parameters, and so is a model. Both mappings
arrive from YAML or from a declared regime, so both hold whatever the file held. These
readers are where that becomes a number, and where a name nobody understands becomes an
error rather than a default silently taking effect.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from nnphysics.core.errors import ValidationError

__all__ = [
    "ParameterMap",
    "bool_parameter",
    "check_parameter_names",
    "float_parameter",
    "int_parameter",
]

type ParameterMap = Mapping[str, object]
"""Loosely typed parameters as they arrive from configuration or from a regime."""


def check_parameter_names(
    parameters: ParameterMap, allowed: Collection[str], *, context: str
) -> None:
    """Reject parameters that the reader does not understand.

    A misspelled parameter that is silently ignored looks like a physics bug later.

    Args:
        parameters: The parameters given.
        allowed: Every name the reader accepts.
        context: What is being configured, used in the error message.

    Raises:
        ValidationError: If any parameter name is not allowed.
    """
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise ValidationError(
            f"unknown {context} parameters {unknown}, accepted: {sorted(allowed)}"
        )


def float_parameter(
    parameters: ParameterMap, name: str, default: float | None = None, *, context: str
) -> float:
    """Read one real valued parameter.

    Args:
        parameters: The parameters given.
        name: Parameter to read.
        default: Value used when the parameter is absent. `None` makes it required.
        context: What is being configured, used in the error message.

    Returns:
        The value as a float.

    Raises:
        ValidationError: If the parameter is required and absent, or is not a real number.
    """
    if name not in parameters:
        if default is None:
            raise ValidationError(f"{context} requires the parameter {name!r}")
        return default
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{context} parameter {name!r} must be a number, got {value!r}")
    return float(value)


def int_parameter(
    parameters: ParameterMap, name: str, default: int | None = None, *, context: str
) -> int:
    """Read one integer valued parameter.

    Regime parameters are carried as floats, so an integral float is accepted and a
    fractional one is not.

    Args:
        parameters: The parameters given.
        name: Parameter to read.
        default: Value used when the parameter is absent. `None` makes it required.
        context: What is being configured, used in the error message.

    Returns:
        The value as an int.

    Raises:
        ValidationError: If the parameter is required and absent, is not a number, or is
            not integral.
    """
    if name not in parameters and default is not None:
        return default
    value = float_parameter(parameters, name, context=context)
    if value != int(value):
        raise ValidationError(f"{context} parameter {name!r} must be a whole number, got {value}")
    return int(value)


def bool_parameter(
    parameters: ParameterMap, name: str, default: bool | None = None, *, context: str
) -> bool:
    """Read one boolean parameter.

    Only a genuine boolean is accepted. A parameter that selects behaviour is worth
    spelling out in the configuration, and reading `1` as true would mean a typed number
    silently changed what a model does.

    Args:
        parameters: The parameters given.
        name: Parameter to read.
        default: Value used when the parameter is absent. `None` makes it required.
        context: What is being configured, used in the error message.

    Returns:
        The value.

    Raises:
        ValidationError: If the parameter is required and absent, or is not a boolean.
    """
    if name not in parameters:
        if default is None:
            raise ValidationError(f"{context} requires the parameter {name!r}")
        return default
    value = parameters[name]
    if not isinstance(value, bool):
        raise ValidationError(f"{context} parameter {name!r} must be true or false, got {value!r}")
    return value

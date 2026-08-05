"""Declarations shared by every system: the registry and parameter reading.

A system is registered as a factory rather than an instance so that a configuration can
carry system level parameters, for example a softening length, and still resolve a name
to a fully built system at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import System
from nnphysics.core.registry import Registry

__all__ = [
    "SYSTEMS",
    "SystemFactory",
    "SystemParameters",
    "build_system",
    "check_parameter_names",
    "float_parameter",
    "int_parameter",
]

type SystemParameters = Mapping[str, object]
"""Loosely typed parameters as they arrive from configuration or from a regime."""

type SystemFactory = Callable[[SystemParameters], System]
"""Builds a system from its system level parameters."""

SYSTEMS: Registry[SystemFactory] = Registry("system")
"""Every registered system factory. Importing `nnphysics.systems` fills it."""


def build_system(name: str, parameters: SystemParameters | None = None) -> System:
    """Resolve a registered system name and build it.

    Args:
        name: Registered system name, for example `nbody`.
        parameters: System level parameters, or `None` for the defaults.

    Returns:
        The built system.

    Raises:
        UnknownNameError: If no system is registered under that name.
        ValidationError: If a parameter is unknown or of the wrong type.
    """
    return SYSTEMS.get(name)({} if parameters is None else parameters)


def check_parameter_names(
    parameters: SystemParameters, allowed: Collection[str], *, context: str
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
    parameters: SystemParameters, name: str, default: float | None = None, *, context: str
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
    parameters: SystemParameters, name: str, default: int | None = None, *, context: str
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

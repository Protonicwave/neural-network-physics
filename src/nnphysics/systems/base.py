"""Declarations shared by every system: the registry and parameter reading.

A system is registered as a factory rather than an instance so that a configuration can
carry system level parameters, for example a softening length, and still resolve a name
to a fully built system at the boundary.

The parameter readers live in `core` and are re-exported here. Models are configured the
same way systems are, so the readers belong to neither layer alone.
"""

from __future__ import annotations

from collections.abc import Callable

from nnphysics.core.params import (
    ParameterMap,
    check_parameter_names,
    float_parameter,
    int_parameter,
)
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

type SystemParameters = ParameterMap
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

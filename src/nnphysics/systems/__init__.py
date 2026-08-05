"""Systems layer: the physical systems and their reference solvers.

This layer imports only `core`. Importing it registers every system, so a name in a
configuration resolves without the caller knowing which module implements it.
"""

from nnphysics.systems.base import (
    SYSTEMS,
    SystemFactory,
    SystemParameters,
    build_system,
    check_parameter_names,
    float_parameter,
    int_parameter,
)
from nnphysics.systems.nbody import build_nbody

SYSTEMS.add("nbody", build_nbody)

__all__ = [
    "SYSTEMS",
    "SystemFactory",
    "SystemParameters",
    "build_system",
    "check_parameter_names",
    "float_parameter",
    "int_parameter",
]

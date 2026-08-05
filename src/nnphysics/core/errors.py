"""Exception hierarchy for the package.

Every failure raised by `nnphysics` derives from `NNPhysicsError`, so a caller can
distinguish our failures from those of a dependency.
"""

__all__ = [
    "ConfigurationError",
    "DuplicateRegistrationError",
    "NNPhysicsError",
    "NumericalError",
    "RegistryError",
    "UnknownNameError",
    "ValidationError",
]


class NNPhysicsError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(NNPhysicsError):
    """A configuration file or override is missing, malformed or invalid."""


class ValidationError(NNPhysicsError):
    """A value does not satisfy the contract of the boundary it crossed."""


class NumericalError(NNPhysicsError):
    """A numerical routine failed: divergence, a non finite value or no convergence."""


class RegistryError(NNPhysicsError):
    """Base class for registry failures."""


class DuplicateRegistrationError(RegistryError):
    """A name was registered twice in the same registry."""


class UnknownNameError(RegistryError):
    """A name was looked up that was never registered."""

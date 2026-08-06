"""Core layer: protocols, types, registry, config, seeding, errors and units.

This layer imports nothing else from the package. Everything else depends on it.
"""

from nnphysics.core.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    RunConfig,
    SystemConfig,
    TrainingConfig,
    load_run_config,
)
from nnphysics.core.errors import (
    ConfigurationError,
    DuplicateRegistrationError,
    NNPhysicsError,
    NumericalError,
    RegistryError,
    UnknownNameError,
    ValidationError,
)
from nnphysics.core.protocols import (
    Conservation,
    Invariant,
    Metric,
    Predictor,
    Refinement,
    Symmetry,
    System,
)
from nnphysics.core.registry import Registry
from nnphysics.core.seeding import (
    make_deterministic,
    numpy_generator,
    seed_sequence,
    torch_generator,
)
from nnphysics.core.types import (
    FieldSpec,
    MetricResult,
    Regime,
    Rollout,
    State,
    StateSpec,
    Trajectory,
)
from nnphysics.core.units import Dimension

__all__ = [
    "ConfigurationError",
    "Conservation",
    "DataConfig",
    "Dimension",
    "DuplicateRegistrationError",
    "EvaluationConfig",
    "FieldSpec",
    "Invariant",
    "Metric",
    "MetricResult",
    "ModelConfig",
    "NNPhysicsError",
    "NumericalError",
    "Predictor",
    "Refinement",
    "Regime",
    "Registry",
    "RegistryError",
    "Rollout",
    "RunConfig",
    "State",
    "StateSpec",
    "Symmetry",
    "System",
    "SystemConfig",
    "TrainingConfig",
    "Trajectory",
    "UnknownNameError",
    "ValidationError",
    "load_run_config",
    "make_deterministic",
    "numpy_generator",
    "seed_sequence",
    "torch_generator",
]

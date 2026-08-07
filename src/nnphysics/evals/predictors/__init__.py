"""Reference and deliberately broken predictors, and the registry that names them.

Importing this module registers every predictor, so a specification in a configuration
or on the command line resolves without the caller knowing which class implements it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.core.seeding import numpy_generator
from nnphysics.evals.predictors.base import (
    PREDICTORS,
    PredictorContext,
    PredictorFactory,
    PredictorParameters,
    PredictorSpec,
    build_predictor,
    check_parameters,
    parse_spec,
    read_parameter,
)
from nnphysics.evals.predictors.broken import (
    EnergyInjection,
    LinearExtrapolation,
    NoiseInjection,
    Persistence,
    SymmetryBreak,
    rate_fields,
)
from nnphysics.evals.predictors.reference import REFERENCE_NAME, Substepped
from nnphysics.evals.predictors.uncertain import (
    CALIBRATED_NAME,
    OVERCONFIDENT_NAME,
    UncertainNoise,
)

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor, Symmetry

__all__ = [
    "BROKEN_PREDICTORS",
    "CALIBRATED_NAME",
    "DEFAULT_INJECTION_FACTOR",
    "DEFAULT_NOISE_SCALE",
    "DEFAULT_OVERCONFIDENCE",
    "OVERCONFIDENT_NAME",
    "PREDICTORS",
    "REFERENCE_NAME",
    "UNCERTAIN_PREDICTORS",
    "EnergyInjection",
    "LinearExtrapolation",
    "NoiseInjection",
    "Persistence",
    "PredictorContext",
    "PredictorFactory",
    "PredictorParameters",
    "PredictorSpec",
    "Substepped",
    "SymmetryBreak",
    "UncertainNoise",
    "build_predictor",
    "parse_spec",
    "rate_fields",
]

DEFAULT_NOISE_SCALE = 0.01
"""One per cent of each field's magnitude per step: the size of error a plausible
surrogate makes, rather than a size chosen to be easy to catch."""

DEFAULT_INJECTION_FACTOR = 1.0e-3
"""Two competing requirements meet here. One step must be more accurate than any trained
model is likely to be, which is the point the fixture exists to make, and the energy it
adds must outrun the reference solver's own energy error over a rollout, or the metric is
right to stay quiet and the fixture has taught nothing. A tenth of a per cent per step
satisfies both: a single step is three orders of magnitude better than persistence, and
a few hundred steps compound it into tens of per cent."""

DEFAULT_OVERCONFIDENCE = 100.0
"""How much smaller the overconfident fixture's stated uncertainty is than its real one.
Two orders of magnitude: far enough that no reasonable calibration measure could call it
acceptable, close enough that it is the mistake a real ensemble of models that agree with
each other actually makes."""

_PREFERRED_BREAKING_SYMMETRY = "rotation"
"""Both supported systems declare one, and it is the example the plan names."""


def _reference(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """Ground truth: the system's own solver at the stored interval."""
    check_parameters(parameters, (), context=REFERENCE_NAME)
    return context.reference


def _persistence(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """The input state, unchanged."""
    check_parameters(parameters, (), context="persistence")
    return Persistence(context.reference.dt)


def _linear_extrapolation(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """A straight line through the previous two states."""
    check_parameters(parameters, (), context="linear_extrapolation")
    return LinearExtrapolation(context.reference)


def _noise(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """The true solver with Gaussian noise at each step."""
    check_parameters(parameters, ("scale",), context="noise")
    return NoiseInjection(
        context.reference,
        read_parameter(parameters, "scale", DEFAULT_NOISE_SCALE, context="noise"),
        numpy_generator(context.seed, context.stream),
    )


def _energy_injection(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """The true solver with a constant amplification of every declared rate."""
    check_parameters(parameters, ("factor",), context="energy_injection")
    return EnergyInjection(
        context.reference,
        rate_fields(context.state_spec),
        read_parameter(parameters, "factor", DEFAULT_INJECTION_FACTOR, context="energy_injection"),
    )


def _symmetry_break(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """The true solver with a declared symmetry applied after each step."""
    check_parameters(parameters, ("index",), context="symmetry_break")
    return SymmetryBreak(context.reference, _breaking_symmetry(context, parameters))


def _breaking_symmetry(context: PredictorContext, parameters: PredictorParameters) -> Symmetry:
    """Choose the symmetry to apply each step.

    Selection is by declared name, not by any knowledge of the system: a rotation is
    preferred because that is the case the plan calls for, and the last declared symmetry
    is the fallback. An `index` parameter overrides both.
    """
    symmetries = context.symmetries
    if not symmetries:
        raise ValidationError("symmetry_break needs a system that declares a symmetry")
    if "index" in parameters:
        index = read_parameter(parameters, "index", 0.0, context="symmetry_break")
        if index != int(index) or not 0 <= int(index) < len(symmetries):
            raise ValidationError(
                f"symmetry_break index must be a whole number in [0, {len(symmetries)}), "
                f"got {index}"
            )
        return symmetries[int(index)]
    for symmetry in symmetries:
        if symmetry.name == _PREFERRED_BREAKING_SYMMETRY:
            return symmetry
    return symmetries[-1]


def _calibrated(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """Noise, with an honest statement of how far off it has drifted."""
    check_parameters(parameters, ("scale",), context=CALIBRATED_NAME)
    return UncertainNoise(
        context.reference,
        read_parameter(parameters, "scale", DEFAULT_NOISE_SCALE, context=CALIBRATED_NAME),
        numpy_generator(context.seed, context.stream),
        confidence=1.0,
        name=CALIBRATED_NAME,
    )


def _overconfident(context: PredictorContext, parameters: PredictorParameters) -> Predictor:
    """The same noise, understating its own uncertainty by a fixed factor."""
    check_parameters(parameters, ("scale", "factor"), context=OVERCONFIDENT_NAME)
    return UncertainNoise(
        context.reference,
        read_parameter(parameters, "scale", DEFAULT_NOISE_SCALE, context=OVERCONFIDENT_NAME),
        numpy_generator(context.seed, context.stream),
        confidence=read_parameter(
            parameters, "factor", DEFAULT_OVERCONFIDENCE, context=OVERCONFIDENT_NAME
        ),
        name=OVERCONFIDENT_NAME,
    )


BROKEN_PREDICTORS = (
    "persistence",
    "linear_extrapolation",
    "noise",
    "energy_injection",
    "symmetry_break",
)
"""Every fixture that is wrong on purpose, in the order the plan introduces them."""

UNCERTAIN_PREDICTORS = (CALIBRATED_NAME, OVERCONFIDENT_NAME)
"""The two that state an uncertainty, one truthfully and one not. They are what the
calibration metric's numbers are read against."""

PREDICTORS.add(REFERENCE_NAME, _reference)
PREDICTORS.add("persistence", _persistence)
PREDICTORS.add("linear_extrapolation", _linear_extrapolation)
PREDICTORS.add("noise", _noise)
PREDICTORS.add("energy_injection", _energy_injection)
PREDICTORS.add("symmetry_break", _symmetry_break)
PREDICTORS.add(CALIBRATED_NAME, _calibrated)
PREDICTORS.add(OVERCONFIDENT_NAME, _overconfident)

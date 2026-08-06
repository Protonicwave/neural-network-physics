from __future__ import annotations

import numpy as np
import pytest
import torch

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State
from nnphysics.models import ModelContext, build_model
from nnphysics.models.baselines import ConstantModel, MultilayerPerceptron
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD, make_state


def test_the_constant_model_is_persistence_before_it_is_trained(
    context: ModelContext, state: State
) -> None:
    stepped = build_model("constant", context).step(state)
    for field in (POSITION_FIELD, VELOCITY_FIELD):
        assert np.allclose(stepped.fields[field], state.fields[field], atol=1e-6)


def test_the_constant_model_learns_one_number_per_predicted_field(
    context: ModelContext,
) -> None:
    model = build_model("constant", context)
    assert model.n_parameters == len(model.predicted_fields)


def test_the_constant_offset_is_the_same_for_every_element_and_in_normalised_units(
    context: ModelContext, state: State
) -> None:
    model = build_model("constant", context)
    assert isinstance(model, ConstantModel)
    with torch.no_grad():
        model.offset[0] = 0.25
    stepped = model.step(state)
    moved = stepped.fields[POSITION_FIELD] - state.fields[POSITION_FIELD]
    assert np.allclose(moved, 0.25 * context.normalisation.stats[POSITION_FIELD].scale, atol=1e-6)
    assert np.allclose(stepped.fields[VELOCITY_FIELD], state.fields[VELOCITY_FIELD], atol=1e-6)


def test_the_perceptron_is_persistence_before_it_is_trained(
    context: ModelContext, state: State
) -> None:
    stepped = build_model("mlp", context).step(state)
    for field in (POSITION_FIELD, VELOCITY_FIELD):
        assert np.allclose(stepped.fields[field], state.fields[field], atol=1e-6)


def test_the_perceptron_records_the_settings_it_was_built_with(context: ModelContext) -> None:
    model = build_model("mlp", context, {"hidden": 16, "layers": 2})
    assert model.hyperparameters == {"hidden": 16, "layers": 2}
    assert model.n_parameters < build_model("mlp", context).n_parameters


def test_the_perceptron_refuses_a_state_of_another_size(
    context: ModelContext, state: State
) -> None:
    """A flattened state model cannot transfer across a change of size, and says so."""
    model = build_model("mlp", context)
    smaller = make_state(np.zeros((3, 2)), np.zeros((3, 2)), np.full(3, 1.0 / 3), time=state.time)
    with pytest.raises(ValidationError, match="does not transfer"):
        model.step(smaller)


def test_the_perceptron_needs_two_layers(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="at least two layers"):
        MultilayerPerceptron(context, layers=1)


@pytest.mark.parametrize("name", ["constant", "mlp"])
def test_a_baseline_rejects_a_hyperparameter_it_does_not_understand(
    name: str, context: ModelContext
) -> None:
    with pytest.raises(ValidationError, match="unknown"):
        build_model(name, context, {"rounds": 2})


def test_the_perceptron_sees_the_static_field_even_though_it_does_not_predict_it(
    context: ModelContext, state: State
) -> None:
    """Mass is an input to the network and an output of the model, unchanged."""
    model = build_model("mlp", context)
    assert MASS_FIELD not in model.predicted_fields
    assert np.array_equal(model.step(state).fields[MASS_FIELD], state.fields[MASS_FIELD])
    # A change of mass alone must be able to change what the network predicts, which it
    # can only do if mass reaches the input layer.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.2, 0.2)
    heavier = make_state(
        state.fields[POSITION_FIELD],
        state.fields[VELOCITY_FIELD],
        state.fields[MASS_FIELD] * 2.0,
        time=state.time,
    )
    assert not np.allclose(
        model.step(heavier).fields[VELOCITY_FIELD], model.step(state).fields[VELOCITY_FIELD]
    )

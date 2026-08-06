from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.models import MODELS, ModelContext, build_model, load_model, save_model
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD

NAMES = ("constant", "mlp", "graph")

Batcher = Callable[..., dict[str, torch.Tensor]]
Exciter = Callable[..., None]


def test_every_model_the_phase_delivers_is_registered() -> None:
    assert set(NAMES) <= set(MODELS.names())


def test_a_context_needs_statistics_for_every_field(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="no normalisation statistics"):
        ModelContext(
            field_shapes=context.field_shapes,
            static_fields=(MASS_FIELD,),
            normalisation=Normalisation({POSITION_FIELD: FieldStats(0.0, 1.0, 10)}),
            dt=context.dt,
            seed=0,
        )


def test_a_static_field_must_be_a_field_of_the_data(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="not fields of the data"):
        ModelContext(
            field_shapes=context.field_shapes,
            static_fields=("charge",),
            normalisation=context.normalisation,
            dt=context.dt,
            seed=0,
        )


def test_a_dataset_that_never_moves_leaves_a_model_nothing_to_do(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="nothing for a model to predict"):
        ModelContext(
            field_shapes=context.field_shapes,
            static_fields=context.names,
            normalisation=context.normalisation,
            dt=context.dt,
            seed=0,
        )


def test_a_context_needs_a_positive_interval(context: ModelContext) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        ModelContext(
            field_shapes=context.field_shapes,
            static_fields=context.static_fields,
            normalisation=context.normalisation,
            dt=0.0,
            seed=0,
        )


@pytest.mark.parametrize("name", NAMES)
def test_a_model_predicts_what_moves_and_carries_what_does_not(
    name: str, context: ModelContext
) -> None:
    model = build_model(name, context)
    assert model.static_fields == (MASS_FIELD,)
    assert model.predicted_fields == (POSITION_FIELD, VELOCITY_FIELD)


@pytest.mark.parametrize("name", NAMES)
def test_one_step_advances_by_the_stored_interval_and_leaves_mass_alone(
    name: str, context: ModelContext, state: State, excite: Exciter
) -> None:
    model = build_model(name, context)
    excite(model)
    stepped = model.step(state)
    assert stepped.time == pytest.approx(state.time + context.dt)
    assert np.array_equal(stepped.fields[MASS_FIELD], state.fields[MASS_FIELD])
    stepped.require_finite()


@pytest.mark.parametrize("name", NAMES)
def test_a_state_missing_a_field_is_refused(name: str, context: ModelContext, state: State) -> None:
    model = build_model(name, context)
    partial = State(fields={POSITION_FIELD: state.fields[POSITION_FIELD]}, time=state.time)
    with pytest.raises(ValidationError, match="missing"):
        model.step(partial)


def test_normalisation_round_trips(context: ModelContext, batch: dict[str, torch.Tensor]) -> None:
    model = build_model("constant", context)
    restored = model.denormalise(model.normalise(batch))
    for name, tensor in batch.items():
        assert torch.allclose(restored[name], tensor, atol=1e-5)


def test_normalisation_uses_the_statistics_it_was_given(
    context: ModelContext, batch: dict[str, torch.Tensor]
) -> None:
    model = build_model("constant", context)
    stats = context.normalisation.stats[POSITION_FIELD]
    expected = (batch[POSITION_FIELD] - stats.mean) / stats.scale
    assert torch.allclose(model.normalise(batch)[POSITION_FIELD], expected)


@pytest.mark.parametrize("name", NAMES)
def test_an_unroll_stacks_one_entry_per_step(
    name: str, context: ModelContext, batch: dict[str, torch.Tensor], excite: Exciter
) -> None:
    model = build_model(name, context)
    excite(model)
    rolled = model.unroll(batch, 5)
    assert set(rolled) == set(model.predicted_fields)
    assert rolled[POSITION_FIELD].shape == (1, 5, *context.field_shapes[POSITION_FIELD])


@pytest.mark.parametrize("name", NAMES)
def test_an_unroll_is_the_same_as_stepping_that_many_times(
    name: str,
    context: ModelContext,
    state: State,
    as_batch: Batcher,
    excite: Exciter,
) -> None:
    model = build_model(name, context)
    excite(model)
    stepped = state
    for _ in range(4):
        stepped = model.step(stepped)
    with torch.no_grad():
        rolled = model.unroll(as_batch(state), 4)
    for field in model.predicted_fields:
        assert np.allclose(
            rolled[field][0, -1].numpy(), stepped.fields[field], rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize("name", NAMES)
def test_an_unroll_takes_at_least_one_step(
    name: str, context: ModelContext, batch: dict[str, torch.Tensor]
) -> None:
    model = build_model(name, context)
    with pytest.raises(ValidationError, match="at least one step"):
        model.unroll(batch, 0)


@pytest.mark.parametrize("name", NAMES)
def test_a_saved_model_predicts_what_the_saved_one_did(
    name: str, context: ModelContext, state: State, tmp_path: Path, excite: Exciter
) -> None:
    model = build_model(name, context)
    excite(model)
    expected = model.step(state)
    path = tmp_path / "model.pt"
    save_model(path, model)

    restored = load_model(path)
    assert restored.name == model.name
    assert restored.dt == model.dt
    assert restored.context.static_fields == model.context.static_fields
    actual = restored.step(state)
    for field in model.predicted_fields:
        assert np.array_equal(actual.fields[field], expected.fields[field])


def test_a_checkpoint_carries_the_statistics_rather_than_trusting_a_configuration(
    context: ModelContext, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    save_model(path, build_model("graph", context))
    assert load_model(path).context.normalisation.stats == context.normalisation.stats


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"schema_version": 99}, path)
    with pytest.raises(ConfigurationError, match="not a version"):
        load_model(path)


def test_a_missing_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_model(tmp_path / "absent.pt")


@pytest.mark.parametrize("name", NAMES)
def test_a_model_reports_the_parameters_it_would_train(name: str, context: ModelContext) -> None:
    assert build_model(name, context).n_parameters > 0


@pytest.mark.parametrize("name", NAMES)
def test_the_carry_cannot_change_an_answer(
    name: str, context: ModelContext, state: State, excite: Exciter
) -> None:
    """A cache miss must cost an evaluation and nothing else."""
    model = build_model(name, context)
    excite(model)
    chained = model.step(model.step(state))

    fresh = build_model(name, context)
    fresh.load_state_dict(model.state_dict())
    first = fresh.step(state)
    # A state the model did not produce, so the carry is dropped and recomputed.
    detached = State(fields=dict(first.fields), time=first.time)
    for field in model.predicted_fields:
        assert np.allclose(
            fresh.step(detached).fields[field], chained.fields[field], rtol=1e-5, atol=1e-8
        )

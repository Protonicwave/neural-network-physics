"""The Fourier neural operator, and the convolutional baseline it is measured against.

The two are tested together because the point of the baseline is the contrast. Both must
be trainable, both must start from persistence, and exactly one of them should still mean
the same thing when the grid is refined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from tests.models.operator.conftest import SIZE, make_context, smooth_field, smooth_state
from torch import nn

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Predictor
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.models import (
    ModelContext,
    SurrogateModel,
    build_model,
    load_model,
    save_model,
)
from nnphysics.systems.fluid.grid import VORTICITY_FIELD
from nnphysics.systems.fluid.refinement import SpectralRefinement
from nnphysics.systems.fluid.system import build_fluid

if TYPE_CHECKING:
    from pathlib import Path

NAMES = ("operator", "convolution")

SMALL = {
    "operator": {"modes": 4, "width": 6, "layers": 2, "projection": 8},
    "convolution": {"width": 8, "layers": 2},
}
"""Settings small enough for a test, and the same shape as the defaults."""


def build(name: str, context: ModelContext) -> SurrogateModel:
    """One of the two grid models, at test size."""
    return build_model(name, context, SMALL[name])


def excite(module: torch.nn.Module, seed: int = 1) -> None:
    """Give the zero initialised head some weight, so the model actually predicts."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.2, 0.2, generator=generator)


@pytest.mark.parametrize("name", NAMES)
class TestBoth:
    def test_it_is_a_predictor(self, name: str, context: ModelContext) -> None:
        assert isinstance(build(name, context), Predictor)

    def test_an_untrained_model_is_persistence(self, name: str, context: ModelContext) -> None:
        """The head starts at zero, so training begins from the identity, not from noise."""
        model = build(name, context)
        field = smooth_field(SIZE)

        predicted, carry = model.advance({VORTICITY_FIELD: field}, None)

        assert carry is None
        assert torch.allclose(predicted[VORTICITY_FIELD], field, atol=1e-6)

    def test_it_predicts_something_once_it_has_weights(
        self, name: str, context: ModelContext
    ) -> None:
        model = build(name, context)
        excite(model)
        field = smooth_field(SIZE)

        predicted, _ = model.advance({VORTICITY_FIELD: field}, None)

        assert float((predicted[VORTICITY_FIELD] - field).detach().abs().max()) > 1e-3

    def test_gradients_reach_every_parameter(self, name: str, context: ModelContext) -> None:
        model = build(name, context)
        excite(model)

        predicted = model.unroll({VORTICITY_FIELD: smooth_field(SIZE)}, 2)[VORTICITY_FIELD]
        torch.autograd.backward(predicted.pow(2).mean())

        starved = sorted(
            key
            for key, parameter in model.named_parameters()
            if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        )
        assert not starved

    def test_it_steps_a_state_through_the_predictor_interface(
        self, name: str, context: ModelContext
    ) -> None:
        model = build(name, context)
        excite(model)
        state = smooth_state(SIZE)

        stepped = model.step(state)

        assert stepped.time == pytest.approx(state.time + model.dt)
        assert stepped.fields[VORTICITY_FIELD].shape == (SIZE, SIZE)
        assert np.isfinite(stepped.fields[VORTICITY_FIELD]).all()

    def test_a_checkpoint_round_trips(
        self, name: str, context: ModelContext, tmp_path: Path
    ) -> None:
        model = build(name, context)
        excite(model)
        path = tmp_path / "model.pt"
        save_model(path, model)

        reloaded = load_model(path)

        assert reloaded.name == model.name
        assert reloaded.hyperparameters == model.hyperparameters
        assert torch.allclose(
            reloaded.forward({VORTICITY_FIELD: smooth_field(SIZE)})[VORTICITY_FIELD],
            model.forward({VORTICITY_FIELD: smooth_field(SIZE)})[VORTICITY_FIELD],
        )

    def test_the_same_seed_gives_the_same_weights(self, name: str, context: ModelContext) -> None:
        first = build(name, context)
        second = build(name, context)
        other = build(name, make_context(SIZE))

        for (_, left), (_, right) in zip(
            first.named_parameters(), second.named_parameters(), strict=True
        ):
            assert torch.equal(left, right)
        assert other is not first

    def test_data_that_is_not_a_grid_is_refused(self, name: str) -> None:
        """These models read the shape of the data, not the name of the system.

        A field of masses, one number per body, is not an image and never will be, so
        the model says so at construction rather than reshaping something into a grid.
        """
        context = ModelContext(
            field_shapes={"position": (8, 2), "mass": (8,)},
            static_fields=("mass",),
            normalisation=Normalisation(
                {
                    "position": FieldStats(mean=0.0, std=1.0, count=10),
                    "mass": FieldStats(mean=1.0, std=0.1, count=10),
                }
            ),
            dt=0.01,
            seed=0,
        )

        with pytest.raises(ValidationError, match="two dimensional grid"):
            build(name, context)


class TestSettings:
    @pytest.mark.parametrize("hyperparameters", [{"layers": 0}, {"width": 0}, {"projection": 0}])
    def test_an_operator_setting_out_of_range_is_refused(
        self, context: ModelContext, hyperparameters: dict[str, int]
    ) -> None:
        with pytest.raises(ValidationError):
            build_model("operator", context, {**SMALL["operator"], **hyperparameters})

    def test_an_unknown_operator_setting_is_refused(self, context: ModelContext) -> None:
        with pytest.raises(ValidationError, match="unknown"):
            build_model("operator", context, {"depth": 3})

    def test_an_even_stencil_is_refused(self, context: ModelContext) -> None:
        """A stencil that is not centred would shift the field a half cell every step."""
        with pytest.raises(ValidationError, match="odd"):
            build_model("convolution", context, {"kernel_size": 4})

    def test_the_settings_a_model_reports_are_the_ones_it_was_built_with(
        self, context: ModelContext
    ) -> None:
        model = build_model("operator", context, SMALL["operator"])

        assert model.hyperparameters == SMALL["operator"]

    def test_the_defaults_put_the_two_models_at_the_same_size(self) -> None:
        """Without a matched parameter count the comparison says nothing about structure."""
        context = make_context(64)
        operator = build_model("operator", context, {"width": 12})
        convolution = build_model("convolution", context, {"width": 74})

        ratio = operator.n_parameters / convolution.n_parameters
        assert 0.99 < ratio < 1.01


class TestResolution:
    """The claim the phase turns on, and the control that gives it meaning."""

    def test_the_operator_runs_on_a_grid_it_was_not_built_for(self) -> None:
        model = build_model("operator", make_context(SIZE), SMALL["operator"])
        excite(model)

        predicted, _ = model.advance({VORTICITY_FIELD: smooth_field(2 * SIZE)}, None)

        assert predicted[VORTICITY_FIELD].shape == (1, 2 * SIZE, 2 * SIZE)

    def test_the_operator_gives_the_same_answer_on_a_finer_grid(self) -> None:
        model = build_model("operator", make_context(SIZE), SMALL["operator"])
        excite(model)

        assert _refined_disagreement(model) < 1e-5

    def test_a_stencil_that_has_learned_a_derivative_does_not(self) -> None:
        """The control, without which passing the test above shows only that it cannot fail.

        A convolution is set to the five point Laplacian, which is what a model of a
        viscous flow has to learn some version of. The stencil is a statement about
        neighbouring grid points, so on a grid of half the spacing it computes four times
        the second derivative it did before, and its answer changes by a factor the grid
        chose rather than the physics.
        """
        model = build_model("convolution", make_context(SIZE), {"width": 1, "layers": 1})
        stencil = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        block = cast("nn.Conv2d", cast("nn.ModuleList", model.blocks)[0])
        head = cast("nn.Conv2d", model.head)
        with torch.no_grad():
            block.weight.copy_(0.02 * stencil.reshape(1, 1, 3, 3))
            cast("torch.Tensor", block.bias).zero_()
            head.weight.fill_(1.0)
            cast("torch.Tensor", head.bias).zero_()

        assert _refined_disagreement(model) > 0.5

    def test_a_randomly_initialised_stencil_is_not_yet_a_control(self) -> None:
        """Worth asserting, because it is why the control above sets the weights by hand.

        Before training, a stencil's response is dominated by the sum of its weights,
        which is the same on any grid. Resolution dependence is something a convolutional
        model acquires by learning the physics, not something it has at initialisation,
        so an untrained one would make the comparison look better than it is.
        """
        model = build_model("convolution", make_context(SIZE), SMALL["convolution"])
        excite(model)

        assert _refined_disagreement(model) < 0.05

    def test_a_grid_smaller_than_the_retained_band_is_refused(self) -> None:
        model = build_model("operator", make_context(SIZE), {**SMALL["operator"], "modes": 8})

        with pytest.raises(ValidationError, match="at least"):
            model.advance({VORTICITY_FIELD: smooth_field(8)}, None)


def _refined_disagreement(model: SurrogateModel) -> float:
    """How much a model's one step update changes when the grid is refined.

    Measured against the size of the update rather than of the state, because these
    models predict an update and a state that is mostly carried through would hide the
    whole difference. Measured through the system's own refinement rather than by
    subsampling, so that it is the same transformation the resolution metric applies.
    """
    system = build_fluid({"grid_size": SIZE})
    refinement = SpectralRefinement(system.grid, 2)
    state = smooth_state(SIZE)
    before = state.fields[VORTICITY_FIELD]

    native = model.step(state).fields[VORTICITY_FIELD] - before
    refined = (
        refinement.coarsen(model.step(refinement.refine(state))).fields[VORTICITY_FIELD] - before
    )

    return float(np.sqrt(np.sum((refined - native) ** 2)) / np.sqrt(np.sum(native**2)))

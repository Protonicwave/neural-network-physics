from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ValidationError
from nnphysics.data.dataset import TrajectoryWindows
from nnphysics.data.manifest import Manifest, Split
from nnphysics.models import ModelContext, SurrogateModel, build_model
from nnphysics.training import rollout_error, rollout_loss, rollout_residual

Dataset = tuple[Path, Manifest, RunConfig]
STEPS = 3


@pytest.fixture
def batch(dataset: Dataset) -> dict[str, dict[str, torch.Tensor]]:
    """One batch of windows long enough for a multi step loss."""
    directory, manifest, _ = dataset
    windows = TrajectoryWindows(
        directory, Split.TRAIN, sequence_length=STEPS, stride=4, manifest=manifest
    )
    return next(iter(DataLoader(windows, batch_size=4, shuffle=False)))


@pytest.fixture
def model(context: ModelContext) -> SurrogateModel:
    return build_model("constant", context)


def test_a_perfect_predictor_has_no_error(
    context: ModelContext, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    """Scored against its own output, any model reads zero, which anchors the scale."""
    model = build_model("constant", context)
    with torch.no_grad():
        targets = model.unroll(batch["inputs"], STEPS)
        error = rollout_error(model, batch["inputs"], targets, STEPS)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-12)


def test_there_is_one_error_per_step_and_it_grows(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    with torch.no_grad():
        error = rollout_error(model, batch["inputs"], batch["targets"], STEPS)
    assert error.shape == (STEPS,)
    assert float(error[0]) < float(error[-1])


def test_a_window_shorter_than_the_unroll_is_refused(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    with pytest.raises(ValidationError, match="needs"):
        rollout_error(model, batch["inputs"], batch["targets"], STEPS + 1)


def test_one_step_training_has_no_later_steps_to_be_wrong_about(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    terms = rollout_loss(
        model, batch["inputs"], batch["targets"], 1, multi_step_weight=1.0, physics_weight=0.0
    )
    assert terms.multi_step == 0.0
    assert float(terms.total.detach()) == pytest.approx(terms.one_step)


def test_the_multi_step_weight_is_what_it_says(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    weighted = rollout_loss(
        model, batch["inputs"], batch["targets"], STEPS, multi_step_weight=2.0, physics_weight=0.0
    )
    assert float(weighted.total.detach()) == pytest.approx(
        weighted.one_step + 2.0 * weighted.multi_step, rel=1e-5
    )


def test_a_weight_of_zero_leaves_the_one_step_term_alone(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    terms = rollout_loss(
        model, batch["inputs"], batch["targets"], STEPS, multi_step_weight=0.0, physics_weight=0.0
    )
    assert float(terms.total.detach()) == pytest.approx(terms.one_step)
    assert terms.multi_step > 0.0


def test_the_physics_penalty_is_off_unless_it_is_weighted(
    context: ModelContext, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    """A model that declares a penalty must not pay it while the weight is zero."""
    model = build_model("graph", context, {"physical_softening": 0.05})
    unweighted = rollout_loss(
        model, batch["inputs"], batch["targets"], 1, multi_step_weight=1.0, physics_weight=0.0
    )
    weighted = rollout_loss(
        model, batch["inputs"], batch["targets"], 1, multi_step_weight=1.0, physics_weight=1.0
    )
    assert unweighted.physics == 0.0
    assert weighted.physics > 0.0
    assert float(weighted.total.detach()) > float(unweighted.total.detach())


def test_the_loss_carries_a_gradient(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    terms = rollout_loss(
        model, batch["inputs"], batch["targets"], STEPS, multi_step_weight=1.0, physics_weight=0.0
    )
    torch.autograd.backward(terms.total)
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("steps", "multi", "physics"), [(0, 1.0, 0.0), (1, -1.0, 0.0), (1, 1.0, -1.0)]
)
def test_a_malformed_loss_is_refused(
    model: SurrogateModel,
    batch: dict[str, dict[str, torch.Tensor]],
    steps: int,
    multi: float,
    physics: float,
) -> None:
    with pytest.raises(ValidationError):
        rollout_loss(
            model,
            batch["inputs"],
            batch["targets"],
            steps,
            multi_step_weight=multi,
            physics_weight=physics,
        )


def test_a_diverging_rollout_cannot_swamp_a_batch(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    """The failure this prevents: one lengthening of the curriculum wrecking the model.

    A squared error on residuals of tens of standard deviations is a gradient thousands
    of times anything else in the batch. The objective counts them linearly instead.
    """
    wrecked = {name: value + 100.0 for name, value in batch["targets"].items()}
    with torch.no_grad():
        squared = rollout_error(model, batch["inputs"], wrecked, 1)
        robust = rollout_residual(model, batch["inputs"], wrecked, 1)
    assert float(robust[0]) < float(squared[0]) / 10.0


def test_the_objective_and_the_metric_agree_while_the_model_is_working(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    """Below one standard deviation the robust loss is the squared error exactly."""
    with torch.no_grad():
        targets = model.unroll(batch["inputs"], STEPS)
        nudged = {name: value + 0.001 for name, value in targets.items()}
        squared = rollout_error(model, batch["inputs"], nudged, STEPS)
        robust = rollout_residual(model, batch["inputs"], nudged, STEPS)
    assert torch.allclose(squared, robust, rtol=1e-5)


def test_the_robust_threshold_must_be_positive(
    model: SurrogateModel, batch: dict[str, dict[str, torch.Tensor]]
) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        rollout_residual(model, batch["inputs"], batch["targets"], 1, delta=0.0)

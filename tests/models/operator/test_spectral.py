"""The spectral convolution layer.

The claim being tested is narrow and it is the one everything above rests on: the layer
computes a function of the field, not of the array holding it. Sample the same band
limited function on two grids, run the layer on both, and the answers must agree where
the grids do.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.models.operator.conftest import smooth_field

from nnphysics.core.errors import ValidationError
from nnphysics.models.operator.spectral import SpectralConv2d

MODES = 4
WIDTH = 3


def peak(tensor: torch.Tensor) -> float:
    """Largest magnitude in a tensor, detached so reading it is not a graph operation."""
    return float(tensor.detach().abs().max())


def layer(seed: int = 0, modes: int = MODES, channels: int = WIDTH) -> SpectralConv2d:
    return SpectralConv2d(channels, channels, modes, torch.Generator().manual_seed(seed))


class TestShapeAndSettings:
    def test_it_preserves_the_grid_and_maps_the_channels(self) -> None:
        result = SpectralConv2d(2, 5, MODES, torch.Generator().manual_seed(0))(
            torch.randn(3, 2, 16, 16)
        )

        assert result.shape == (3, 5, 16, 16)

    @pytest.mark.parametrize(("fan_in", "fan_out"), [(0, 1), (1, 0)])
    def test_a_layer_with_no_channels_is_refused(self, fan_in: int, fan_out: int) -> None:
        with pytest.raises(ValidationError, match="channel"):
            SpectralConv2d(fan_in, fan_out, MODES, torch.Generator())

    def test_a_layer_retaining_no_mode_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="retain a mode"):
            SpectralConv2d(1, 1, 0, torch.Generator())

    def test_a_grid_too_small_for_the_retained_band_is_refused(self) -> None:
        """The two corners would overlap, and the layer would count a mode twice."""
        with pytest.raises(ValidationError, match="at least"):
            layer(modes=8)(torch.randn(1, WIDTH, 8, 8))

    def test_the_weights_come_from_the_generator_it_was_given(self) -> None:
        assert torch.equal(layer(seed=1).weight_real, layer(seed=1).weight_real)
        assert not torch.equal(layer(seed=1).weight_real, layer(seed=2).weight_real)


class TestResolutionIndependence:
    def test_the_same_function_on_two_grids_gives_the_same_answer(self) -> None:
        """The whole claim: mode four is mode four on either grid."""
        block = layer()
        coarse = smooth_field(16).repeat(WIDTH, 1, 1).unsqueeze(0)
        fine = smooth_field(32).repeat(WIDTH, 1, 1).unsqueeze(0)

        on_coarse = block(coarse)
        on_fine = block(fine)

        assert torch.allclose(on_fine[..., ::2, ::2], on_coarse, atol=1e-5)

    def test_it_is_not_trivially_independent_because_it_outputs_nothing(self) -> None:
        """A layer that returned zero would pass the test above."""
        block = layer()

        assert peak(block(smooth_field(16).repeat(WIDTH, 1, 1).unsqueeze(0))) > 0.01


class TestTruncation:
    def test_content_above_the_cutoff_is_dropped(self) -> None:
        """Truncation is the parameterisation, so it must be visible."""
        block = layer(modes=2, channels=1)
        size = 16
        axis = torch.arange(size, dtype=torch.float32) * (2.0 * np.pi / size)
        high = torch.sin(6.0 * axis).reshape(1, 1, size, 1).repeat(1, 1, 1, size)

        assert peak(block(high)) == pytest.approx(0.0, abs=1e-6)

    def test_content_below_the_cutoff_survives(self) -> None:
        block = layer(modes=2, channels=1)
        size = 16
        axis = torch.arange(size, dtype=torch.float32) * (2.0 * np.pi / size)
        low = torch.sin(axis).reshape(1, 1, size, 1).repeat(1, 1, 1, size)

        assert peak(block(low)) > 0.01


class TestOutput:
    def test_the_output_is_real(self) -> None:
        """Both corners of the half spectrum are retained, so the filter is a real one."""
        result = layer()(smooth_field(16).repeat(WIDTH, 1, 1).unsqueeze(0))

        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_gradients_reach_both_halves_of_every_weight(self) -> None:
        block = layer()
        block(smooth_field(16).repeat(WIDTH, 1, 1).unsqueeze(0)).pow(2).sum().backward()

        assert block.weight_real.grad is not None
        assert block.weight_imaginary.grad is not None
        assert float(block.weight_real.grad.abs().sum()) > 0.0
        assert float(block.weight_imaginary.grad.abs().sum()) > 0.0

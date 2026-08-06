"""The spectral convolution: a learned multiplier on a truncated set of Fourier modes.

A convolution on a periodic domain is a multiplication in Fourier space. This layer
learns that multiplication directly, one complex matrix per retained mode mapping the
input channels to the output ones, and drops every mode above a fixed cutoff.

Two properties follow from that, and they are the reason the layer exists.

**The cutoff is in modes, not in grid points.** Mode eight is the same physical
wavenumber on a grid of sixty four and on a grid of one hundred and twenty eight, so the
weights mean the same thing on both and the layer can run on a grid it never saw. That is
the whole resolution independence claim, and it is a property of this file.

**Truncation is not a side effect, it is the parameterisation.** Keeping every mode would
be one weight per grid point and would tie the layer to the grid again. Keeping a fixed
low band assumes the operator being learned is smooth, which for a viscous flow it is:
the modes that are dropped are the ones viscosity is busy removing.

Weights are held as separate real and imaginary tensors rather than as complex
parameters. The arithmetic is written out because of that, which costs four einsums
instead of one and buys an optimiser and a checkpoint that only ever see real numbers.

The transform is the real one, so the half spectrum holds non negative wavenumbers along
the last axis only. Two corners of it are retained: low wavenumbers along the first axis,
and the high indices of that axis, which are the negative wavenumbers of the same size.
Retaining only the first corner would keep half of each mode pair and learn a filter that
was not real.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nnphysics.core.errors import ValidationError

__all__ = ["SpectralConv2d"]

_CORNERS = 2
"""Low and high indices along the first axis: positive and negative wavenumbers."""


class SpectralConv2d(nn.Module):
    """A learned multiplier on the lowest modes of a two dimensional real transform.

    Args:
        in_channels: Channels read.
        out_channels: Channels written.
        modes: Modes retained along each axis, counted from the lowest.
        generator: Generator every weight is drawn from.

    Raises:
        ValidationError: If a channel count or the mode count is not positive.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes: int, generator: torch.Generator
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValidationError(
                f"a spectral convolution needs at least one channel each way, got "
                f"{in_channels} and {out_channels}"
            )
        if modes < 1:
            raise ValidationError(f"a spectral convolution must retain a mode, got {modes}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Unit gain: the sum over input channels of a uniform weight with this bound has
        # unit variance, so a retained mode leaves the layer about the size it entered it
        # and a stack of these neither dies nor explodes before any training happens.
        bound = math.sqrt(3.0 / in_channels)
        shape = (_CORNERS, in_channels, out_channels, modes, modes)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imaginary = nn.Parameter(torch.empty(shape))
        with torch.no_grad():
            self.weight_real.uniform_(-bound, bound, generator=generator)
            self.weight_imaginary.uniform_(-bound, bound, generator=generator)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Filter a batch of images through the retained modes.

        Args:
            image: Shape `(batch, in_channels, height, width)`. Both extents must be at
                least twice the mode count, so that the retained band exists and its two
                corners do not overlap.

        Returns:
            Shape `(batch, out_channels, height, width)`.

        Raises:
            ValidationError: If the grid is too small for the retained band.
        """
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if min(height, width) < _CORNERS * self.modes:
            raise ValidationError(
                f"a spectral convolution retaining {self.modes} modes needs a grid of at "
                f"least {_CORNERS * self.modes} points each way, got {height} by {width}"
            )
        spectrum = torch.fft.rfft2(image, norm="forward")
        filtered = torch.zeros(
            (image.shape[0], self.out_channels, height, width // 2 + 1),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        cut = self.modes
        filtered[..., :cut, :cut] = self._multiply(spectrum[..., :cut, :cut], corner=0)
        filtered[..., -cut:, :cut] = self._multiply(spectrum[..., -cut:, :cut], corner=1)
        transformed: torch.Tensor = torch.fft.irfft2(filtered, s=(height, width), norm="forward")
        return transformed

    def _multiply(self, block: torch.Tensor, *, corner: int) -> torch.Tensor:
        """One corner of the retained band, times its complex weight matrix."""
        real, imaginary = block.real, block.imag
        weight_real = self.weight_real[corner]
        weight_imaginary = self.weight_imaginary[corner]
        pattern = "bixy,ioxy->boxy"
        return torch.complex(
            torch.einsum(pattern, real, weight_real)
            - torch.einsum(pattern, imaginary, weight_imaginary),
            torch.einsum(pattern, real, weight_imaginary)
            + torch.einsum(pattern, imaginary, weight_real),
        )

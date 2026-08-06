"""Models over fields sampled on a two dimensional grid.

The Fourier neural operator and the convolutional network it is measured against. They
share their channel packing and differ in one thing: whether the learned map is a
multiplier on Fourier modes or a stencil on grid points.

Neither names a field, so neither knows it is looking at a fluid. What makes them
applicable is the shape of the data.
"""

from nnphysics.models.operator.convolution import (
    CONVOLUTION_NAME,
    ConvolutionalBaseline,
    build_convolution,
)
from nnphysics.models.operator.fields import GridFields
from nnphysics.models.operator.model import (
    OPERATOR_NAME,
    FourierNeuralOperator,
    build_operator,
)
from nnphysics.models.operator.spectral import SpectralConv2d

__all__ = [
    "CONVOLUTION_NAME",
    "OPERATOR_NAME",
    "ConvolutionalBaseline",
    "FourierNeuralOperator",
    "GridFields",
    "SpectralConv2d",
    "build_convolution",
    "build_operator",
]

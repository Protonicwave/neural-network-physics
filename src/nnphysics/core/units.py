"""Physical dimensions.

Dimensions are tracked, units are not. A quantity is described by the exponents of
mass, length and time, which is enough to catch a state field declared as a velocity
being filled with an energy, and cheap enough to carry on every field specification.
Simulations run in their own natural units, so a scale factor would carry no
information here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "ACCELERATION",
    "AREA",
    "DIMENSIONLESS",
    "ENERGY",
    "LENGTH",
    "MASS",
    "MOMENTUM",
    "TIME",
    "VELOCITY",
    "Dimension",
]


@dataclass(frozen=True, slots=True)
class Dimension:
    """The dimension of a physical quantity as exponents of mass, length and time.

    Exponents are fractions so that square roots of dimensioned quantities, which
    appear in error norms, stay exact.

    Attributes:
        mass: Exponent of mass.
        length: Exponent of length.
        time: Exponent of time.
    """

    mass: Fraction = Fraction(0)
    length: Fraction = Fraction(0)
    time: Fraction = Fraction(0)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
        )

    def __pow__(self, exponent: int | Fraction) -> Dimension:
        factor = Fraction(exponent)
        return Dimension(self.mass * factor, self.length * factor, self.time * factor)

    @property
    def is_dimensionless(self) -> bool:
        """Whether every exponent is zero."""
        return self.mass == 0 and self.length == 0 and self.time == 0

    @property
    def symbol(self) -> str:
        """A compact human readable form such as `M L^2 T^-2`."""
        if self.is_dimensionless:
            return "1"
        parts = [
            f"{base}^{exponent}" if exponent != 1 else base
            for base, exponent in (("M", self.mass), ("L", self.length), ("T", self.time))
            if exponent != 0
        ]
        return " ".join(parts)

    def __str__(self) -> str:
        return self.symbol


DIMENSIONLESS = Dimension()
MASS = Dimension(mass=Fraction(1))
LENGTH = Dimension(length=Fraction(1))
TIME = Dimension(time=Fraction(1))
AREA = LENGTH**2
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
MOMENTUM = MASS * VELOCITY
ENERGY = MOMENTUM * VELOCITY

from fractions import Fraction

from nnphysics.core.units import (
    DIMENSIONLESS,
    ENERGY,
    LENGTH,
    MASS,
    MOMENTUM,
    TIME,
    VELOCITY,
    Dimension,
)


def test_energy_is_mass_length_squared_per_time_squared() -> None:
    assert Dimension(Fraction(1), Fraction(2), Fraction(-2)) == ENERGY


def test_momentum_is_mass_times_velocity() -> None:
    assert MOMENTUM == MASS * VELOCITY


def test_dividing_a_dimension_by_itself_is_dimensionless() -> None:
    assert (ENERGY / ENERGY).is_dimensionless


def test_fractional_powers_stay_exact() -> None:
    assert (LENGTH**2) ** Fraction(1, 2) == LENGTH


def test_symbols_read_as_expected() -> None:
    assert DIMENSIONLESS.symbol == "1"
    assert LENGTH.symbol == "L"
    assert str(ENERGY) == "M L^2 T^-2"
    assert (LENGTH / TIME).symbol == "L T^-1"


def test_dimensions_are_hashable_and_comparable() -> None:
    assert len({VELOCITY, LENGTH / TIME, ENERGY}) == 2

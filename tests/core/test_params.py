import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.params import (
    bool_parameter,
    check_parameter_names,
    float_parameter,
    int_parameter,
)

CONTEXT = "a reader"


def test_known_names_are_accepted() -> None:
    check_parameter_names({"a": 1, "b": 2}, ("a", "b", "c"), context=CONTEXT)


def test_an_unknown_name_is_rejected_by_name() -> None:
    with pytest.raises(ValidationError, match="hiddne"):
        check_parameter_names({"hiddne": 64}, ("hidden",), context=CONTEXT)


def test_a_float_is_read_and_an_integer_is_widened() -> None:
    assert float_parameter({"x": 2}, "x", context=CONTEXT) == 2.0
    assert float_parameter({"x": 2.5}, "x", context=CONTEXT) == 2.5


def test_a_missing_float_falls_back_or_is_required() -> None:
    assert float_parameter({}, "x", 1.5, context=CONTEXT) == 1.5
    with pytest.raises(ValidationError, match="requires"):
        float_parameter({}, "x", context=CONTEXT)


@pytest.mark.parametrize("value", [True, False, "3", None, [1]])
def test_a_non_number_is_not_a_float(value: object) -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        float_parameter({"x": value}, "x", context=CONTEXT)


def test_an_integral_float_is_an_integer_and_a_fractional_one_is_not() -> None:
    assert int_parameter({"n": 4.0}, "n", context=CONTEXT) == 4
    with pytest.raises(ValidationError, match="whole number"):
        int_parameter({"n": 4.5}, "n", context=CONTEXT)


def test_only_a_genuine_boolean_is_a_boolean() -> None:
    assert bool_parameter({"on": True}, "on", context=CONTEXT) is True
    assert bool_parameter({}, "on", False, context=CONTEXT) is False
    with pytest.raises(ValidationError, match="true or false"):
        bool_parameter({"on": 1}, "on", context=CONTEXT)
    with pytest.raises(ValidationError, match="requires"):
        bool_parameter({}, "on", context=CONTEXT)

import pytest

from nnphysics.core.errors import DuplicateRegistrationError, UnknownNameError
from nnphysics.core.registry import Registry


def test_add_then_get_round_trips() -> None:
    registry: Registry[int] = Registry("widget")
    registry.add("one", 1)
    assert registry.get("one") == 1


def test_duplicate_registration_is_an_error() -> None:
    registry: Registry[int] = Registry("widget")
    registry.add("one", 1)
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        registry.add("one", 2)
    assert registry.get("one") == 1


def test_unknown_name_is_an_error_and_lists_what_is_known() -> None:
    registry: Registry[int] = Registry("widget")
    registry.add("one", 1)
    with pytest.raises(UnknownNameError, match="unknown widget 'two'") as caught:
        registry.get("two")
    assert "one" in str(caught.value)


def test_unknown_name_on_an_empty_registry_says_so() -> None:
    registry: Registry[int] = Registry("widget")
    with pytest.raises(UnknownNameError, match="nothing"):
        registry.get("one")


def test_empty_name_is_rejected() -> None:
    registry: Registry[int] = Registry("widget")
    with pytest.raises(ValueError, match="empty name"):
        registry.add("", 1)


def test_register_decorator_returns_the_object_unchanged() -> None:
    registry: Registry[type[object]] = Registry("widget")

    @registry.register("thing")
    class Thing:
        pass

    assert registry.get("thing") is Thing


def test_listing_is_sorted_and_complete() -> None:
    registry: Registry[int] = Registry("widget")
    for name, value in (("b", 2), ("a", 1), ("c", 3)):
        registry.add(name, value)
    assert registry.names() == ("a", "b", "c")
    assert list(registry) == ["a", "b", "c"]
    assert len(registry) == 3
    assert "a" in registry
    assert "z" not in registry

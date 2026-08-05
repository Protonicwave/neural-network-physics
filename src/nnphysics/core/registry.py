"""A name to implementation registry, used later for systems, models and metrics.

Registration is loud on both sides: registering a name twice and looking up a name that
was never registered are both errors. A silent overwrite would make a run's meaning
depend on import order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from nnphysics.core.errors import DuplicateRegistrationError, UnknownNameError

__all__ = ["Registry"]


class Registry[T]:
    """A mapping from name to implementation for one kind of thing.

    Args:
        kind: What this registry holds, used in error messages, for example `metric`.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @property
    def kind(self) -> str:
        """What this registry holds."""
        return self._kind

    def add(self, name: str, item: T) -> T:
        """Register an implementation under a name.

        Args:
            name: Unique, non empty name.
            item: The implementation.

        Returns:
            The item, so this can be used inside a decorator.

        Raises:
            DuplicateRegistrationError: If the name is already taken.
            ValueError: If the name is empty.
        """
        if not name:
            raise ValueError(f"cannot register a {self._kind} under an empty name")
        if name in self._items:
            raise DuplicateRegistrationError(
                f"{self._kind} {name!r} is already registered as {self._items[name]!r}"
            )
        self._items[name] = item
        return item

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under `name`.

        Args:
            name: Unique, non empty name.

        Returns:
            A decorator returning its argument unchanged.
        """

        def decorator(item: T) -> T:
            return self.add(name, item)

        return decorator

    def get(self, name: str) -> T:
        """Look up an implementation.

        Args:
            name: The registered name.

        Returns:
            The registered implementation.

        Raises:
            UnknownNameError: If nothing is registered under that name.
        """
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "nothing"
            raise UnknownNameError(f"unknown {self._kind} {name!r}, registered: {known}") from None

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted."""
        return tuple(sorted(self._items))

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, names={list(self.names())!r})"

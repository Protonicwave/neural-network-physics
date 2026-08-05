"""What machine and what code produced a run.

A number is only reproducible if what produced it is written down. The interpreter, the
platform and the versions of the libraries that do the arithmetic all change results, and
none of them is visible in a configuration file. The commit is recorded too, read from
the git directory rather than by running git, so that recording it costs nothing and
works where git is not installed.

Nothing here fails. A missing version or an absent repository is recorded as unknown,
because a report that refused to render because it could not find a commit hash would be
worse than a report that says it does not know.
"""

from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RECORDED_PACKAGES", "EnvironmentRecord", "describe_environment", "git_commit"]

RECORDED_PACKAGES = ("numpy", "scipy", "torch", "h5py", "pydantic", "matplotlib")
"""Libraries whose version can change a number. Recorded by name so that reading a record
does not require importing any of them."""

UNKNOWN = "unknown"
"""Written where a value could not be determined, so that a field is never quietly empty."""

_GIT_DIR = ".git"
_HEAD = "HEAD"
_REF_PREFIX = "ref: "
_PACKED_REFS = "packed-refs"


class EnvironmentRecord(BaseModel):
    """The machine and the libraries a run used.

    Attributes:
        python: Interpreter version.
        implementation: Interpreter implementation, for example CPython.
        platform: Operating system and release.
        machine: Processor architecture.
        cpu_count: Logical cores visible to the process, which bounds every timing in the
            record.
        packages: Version of each recorded library.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    python: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    cpu_count: int = Field(ge=1)
    packages: dict[str, str] = {}


def describe_environment() -> EnvironmentRecord:
    """Describe the interpreter, the machine and the installed libraries.

    Returns:
        The record. Every field that cannot be determined is filled with `unknown`.
    """
    return EnvironmentRecord(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=f"{platform.system()} {platform.release()}" if platform.system() else UNKNOWN,
        machine=platform.machine() or UNKNOWN,
        cpu_count=_cpu_count(),
        packages={name: _version(name) for name in RECORDED_PACKAGES},
    )


def git_commit(start: Path | None = None) -> str:
    """Read the commit the working tree is on, without running git.

    Args:
        start: Directory to search upwards from. Defaults to this package's location,
            which is where the code being recorded actually lives.

    Returns:
        The full commit hash, or an empty string if there is no repository, the
        repository has no commit yet, or the reference cannot be resolved.
    """
    directory = _find_git_dir(start or Path(__file__).resolve().parent)
    if directory is None:
        return ""
    head = _read(directory / _HEAD)
    if head is None:
        return ""
    if not head.startswith(_REF_PREFIX):
        return head
    reference = head[len(_REF_PREFIX) :].strip()
    loose = _read(directory / reference)
    if loose is not None:
        return loose
    return _read_packed(directory / _PACKED_REFS, reference)


def _cpu_count() -> int:
    """Logical cores the process may use, never less than one."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def _version(name: str) -> str:
    """Installed version of one package, or `unknown`."""
    try:
        return version(name)
    except PackageNotFoundError:
        return UNKNOWN


def _find_git_dir(start: Path) -> Path | None:
    """The nearest `.git` directory at or above a path."""
    for directory in (start, *start.parents):
        candidate = directory / _GIT_DIR
        if candidate.is_dir():
            return candidate
    return None


def _read(path: Path) -> str | None:
    """First line of a file, stripped, or `None` if it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    line = text.strip()
    return line or None


def _read_packed(path: Path, reference: str) -> str:
    """Resolve a reference out of the packed references file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("^"):
            continue
        commit, _, name = line.partition(" ")
        if name.strip() == reference:
            return commit
    return ""

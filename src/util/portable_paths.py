"""Portable dataset path helpers used by format and evaluation stages.

Converted records store paths relative to ``ANYSCALEGRASP_DATA_ROOT``.  Absolute
paths are accepted for older producer artifacts, but are never written to new
records.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "ANYSCALEGRASP_DATA_ROOT"
LEGACY_DATA_ROOT_ENV = "AnyScaleGraspDataset"


def dataset_root() -> Path | None:
    """Return the configured dataset root, if one is available."""

    value = os.environ.get(DATA_ROOT_ENV) or os.environ.get(LEGACY_DATA_ROOT_ENV)
    return Path(value).expanduser() if value else None


def portable_reference(path: str | os.PathLike[str]) -> str:
    """Convert a producer path to a dataset-relative POSIX reference.

    Args:
        path: Absolute or relative scene/object path from a producer artifact.

    Returns:
        A path relative to the external dataset root whenever a known producer
        or dataset marker is present.  Unknown relative paths are preserved.
    """

    text = str(path).replace("\\", "/")
    for marker in ("/src/curobo/content/", "/AnyScaleGrasp/"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    if text.startswith("assets/object/"):
        text = text[len("assets/") :]
    elif text.startswith("./assets/object/"):
        text = text[len("./assets/") :]

    root = dataset_root()
    if root is not None:
        try:
            text = Path(text).resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return text.lstrip("/") if not Path(text).is_absolute() else text


def candidate_paths(reference: str | os.PathLike[str]) -> list[Path]:
    """Return ordered local candidates for a portable or legacy reference."""

    raw = Path(str(reference).expanduser())
    candidates: list[Path] = [raw]
    root = dataset_root()
    if root is not None and not raw.is_absolute():
        candidates.extend((root / raw, root / "assets" / raw))
    if not raw.is_absolute():
        candidates.extend((Path.cwd() / raw, Path.cwd() / "assets" / raw))
    # Legacy records sometimes contain ``assets/object/...`` after the producer
    # prefix was stripped.  Resolve that spelling against the dataset root too.
    if root is not None and str(raw).startswith("assets/object/"):
        candidates.append(root / str(raw)[len("assets/") :])
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def resolve_path(reference: str | os.PathLike[str], *, required: bool = False) -> str:
    """Resolve a portable reference to an existing local path when possible."""

    candidates = candidate_paths(reference)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if required:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Cannot resolve dataset path {reference!r}; checked: {checked}")
    return str(candidates[0])

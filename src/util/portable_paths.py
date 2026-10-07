"""Portable dataset path helpers used by format and evaluation stages.

Converted records store paths relative to ``HUGS_DATASET_ROOT``.  Absolute
paths are accepted for older producer artifacts, but are never written to new
records.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "HUGS_DATASET_ROOT"


def dataset_root() -> Path | None:
    """Return the configured dataset root, if one is available."""

    value = os.environ.get(DATA_ROOT_ENV)
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
    for marker in ("/src/curobo/content/",):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    if text.startswith("assets/object/"):
        text = text[len("assets/") :]
    elif text.startswith("./assets/object/"):
        text = text[len("./assets/") :]

    root = dataset_root()
    if root is not None and Path(text).is_absolute():
        try:
            text = Path(os.path.abspath(text)).relative_to(Path(os.path.abspath(root))).as_posix()
        except (OSError, ValueError):
            try:
                text = Path(text).resolve().relative_to(root.resolve()).as_posix()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Dataset path is outside {DATA_ROOT_ENV}: {path}") from exc
    if Path(text).is_absolute():
        raise ValueError(f"Set {DATA_ROOT_ENV} before exporting absolute paths: {path}")
    if ".." in Path(text).parts:
        raise ValueError(f"Dataset reference must stay within {DATA_ROOT_ENV}: {path}")
    return Path(text).as_posix()


def candidate_paths(reference: str | os.PathLike[str]) -> list[Path]:
    """Return ordered local candidates for a portable or legacy reference."""

    raw = Path(str(reference)).expanduser()
    candidates: list[Path] = []
    root = dataset_root()
    if root is not None and not raw.is_absolute():
        if ".." in raw.parts:
            raise ValueError(f"Dataset reference must stay within {DATA_ROOT_ENV}: {reference}")
        candidates.append(root / raw)
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.extend((Path.cwd() / raw, Path.cwd() / "assets" / raw))
    # Legacy records sometimes contain ``assets/object/...`` after the producer
    # prefix was stripped.  Resolve that spelling against the dataset root too.
    if root is not None and str(raw).startswith("assets/object/"):
        candidates.append(root / str(raw)[len("assets/") :])
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        # Preserve the public bundle layout when a dataset subtree is a symlink.
        normalized = Path(os.path.abspath(candidate))
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

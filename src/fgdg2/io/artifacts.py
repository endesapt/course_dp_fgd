"""Artifact schema v3 — provenance-rich run records with collision-safe writes.

Layout (top level):
    artifact_type   — "gang_churn" | "gang" | "inflation" | ...
    schema_version  — 3 (artifact schema; independent of the stream schema)
    config_sha256   — sha256 of the canonical config (the run's identity)
    provenance      — git_sha + package versions
    config          — full run configuration (all flags, stream provenance)
    metrics         — result scalars + summaries
    timeline        — downsampled event series (optional)

Collision policy (audit A2 — silent overwrites): `write_artifact` refuses to
overwrite an existing file whose canonical config differs — same config is an
idempotent rewrite, different config under the same name is an error, not a
silent clobber.
"""

import hashlib
import json
import os
from pathlib import Path

from .provenance import provenance

ARTIFACT_SCHEMA_VERSION = 3


class ArtifactCollisionError(RuntimeError):
    """Raised when a different run config would silently overwrite an artifact."""


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_sha(config: dict) -> str:
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()


def make_artifact(artifact_type: str, config: dict, metrics: dict,
                  timeline: dict | None = None, prov: dict | None = None) -> dict:
    return {
        "artifact_type": artifact_type,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": config_sha(config),
        "provenance": prov if prov is not None else provenance(),
        "config": config,
        "metrics": metrics,
        "timeline": timeline,
    }


def write_artifact(path, artifact: dict) -> Path:
    """Atomic write (tmp + os.replace) with the collision guard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_artifact(path, require_schema=False)
        if _canonical(existing.get("config")) != _canonical(artifact["config"]):
            raise ArtifactCollisionError(
                f"{path}: existing artifact was written with a different "
                "config — refusing silent overwrite (audit A2)"
            )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_artifact(path, require_schema: bool = True) -> dict:
    """Read an artifact; require_schema=True rejects unknown schema versions
    (artifact v3 only — no silent reads of legacy formats)."""
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_schema:
        v = art.get("schema_version")
        if v != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported artifact schema_version {v!r} "
                f"(this reader knows {ARTIFACT_SCHEMA_VERSION})"
            )
    return art

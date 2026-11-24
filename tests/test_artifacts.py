"""Artifact schema v3 + provenance: collision-safe atomic writes (audit A2),
schema gating, provenance block (audit A8)."""

import json

import pytest

from fgdg2.io.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCollisionError,
    config_sha,
    load_artifact,
    make_artifact,
    write_artifact,
)
from fgdg2.io.provenance import package_versions, provenance


def _art(cfg=None):
    return make_artifact("gang_churn", cfg or {"policy": "dp", "seed": 0},
                         {"gangs_placed": 10})


def test_schema_and_shape(tmp_path):
    p = write_artifact(tmp_path / "a.json", _art())
    art = json.loads(p.read_text())
    assert art["schema_version"] == ARTIFACT_SCHEMA_VERSION == 3
    assert art["artifact_type"] == "gang_churn"
    assert set(art) >= {"config_sha256", "provenance", "config", "metrics",
                      "timeline"}
    assert art["config_sha256"] == config_sha(art["config"])
    assert art["metrics"]["gangs_placed"] == 10


def test_provenance_block(tmp_path):
    prov = provenance()
    assert prov["git_sha"]            # real repo -> short sha
    vs = package_versions()
    assert vs["python"] and vs["numpy"] and vs["pandas"]
    p = write_artifact(tmp_path / "a.json", _art())
    art = load_artifact(p)
    assert art["provenance"]["git_sha"] == prov["git_sha"]


def test_idempotent_rewrite_same_config(tmp_path):
    p = tmp_path / "a.json"
    write_artifact(p, _art())
    write_artifact(p, _art())  # same config -> allowed (re-run overwrites)
    assert p.exists()


def test_collision_refuses_different_config(tmp_path):
    p = tmp_path / "a.json"
    write_artifact(p, _art({"policy": "dp"}))
    with pytest.raises(ArtifactCollisionError):
        write_artifact(p, _art({"policy": "atomic_fgd"}))
    # the original survived untouched
    assert load_artifact(p)["config"]["policy"] == "dp"


def test_load_rejects_unknown_schema(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"schema_version": 2, "config": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        load_artifact(p)
    # require_schema=False reads anything (used by the collision check itself)
    assert load_artifact(p, require_schema=False)["schema_version"] == 2


def test_config_sha_order_insensitive():
    a = {"b": 1, "a": [1, 2]}
    b = {"a": [1, 2], "b": 1}
    assert config_sha(a) == config_sha(b)


def test_write_is_atomic(tmp_path):
    """The tmp file must not linger; parent dirs are created."""
    p = tmp_path / "sub" / "dir" / "a.json"
    write_artifact(p, _art())
    assert p.exists()
    assert not (tmp_path / "sub" / "dir" / "a.json.tmp").exists()

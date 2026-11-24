"""Provenance — single home for git_sha / package versions (was triplicated
across scripts, audit A8)."""

import platform
import subprocess

import numpy as np
import pandas as pd


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def provenance() -> dict:
    """The provenance block embedded in every artifact v3."""
    return {"git_sha": git_sha(), "versions": package_versions()}

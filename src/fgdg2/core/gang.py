"""Gang plan data types (AGENTS.md §12.4): pure data, no algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One worker placement inside a gang plan (anchor GPU of the worker's block)."""

    node_id: str
    gpu_idx: int | None = None


GangPlan = tuple[Step, ...]

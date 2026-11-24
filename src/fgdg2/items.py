"""StreamItem — THE stream item schema (replaces duck-typed dicts).

One item = one gang (singletons are gang_size=1). Field set matches the
schema-v2 JSONL stream files exactly: optional fields serialize only when
set, so synthetic streams (no temporal/extra fields) stay byte-identical.
"""

from dataclasses import dataclass

from .core.tasks import TaskClass

_ITEM_FIELDS = (
    "gang_id", "gang_size", "worker_cpu_milli", "worker_gpu_demand",
    "arrival_index", "is_gang", "submit_time_s", "duration_s",
    "job_type", "gpu_model",
)


@dataclass(frozen=True)
class StreamItem:
    gang_id: str
    gang_size: int
    worker_cpu_milli: int
    worker_gpu_demand: float
    arrival_index: int
    is_gang: bool
    submit_time_s: float | None = None
    duration_s: float | None = None
    job_type: str | None = None
    gpu_model: str | None = None

    def __post_init__(self):
        if int(self.gang_size) < 1:
            raise ValueError(f"gang_size must be >= 1, got {self.gang_size}")
        if bool(self.is_gang) != (int(self.gang_size) > 1):
            raise ValueError(
                f"is_gang={self.is_gang} inconsistent with gang_size={self.gang_size}"
            )
        if float(self.worker_gpu_demand) < 0:
            raise ValueError(f"worker_gpu_demand must be >= 0, got {self.worker_gpu_demand}")
        object.__setattr__(self, "gang_size", int(self.gang_size))
        object.__setattr__(self, "worker_cpu_milli", int(self.worker_cpu_milli))
        object.__setattr__(self, "worker_gpu_demand", float(self.worker_gpu_demand))
        object.__setattr__(self, "arrival_index", int(self.arrival_index))
        object.__setattr__(self, "is_gang", bool(self.is_gang))

    # -- derived accessors -------------------------------------------------
    @property
    def class_key(self) -> tuple[int, float]:
        """The (cpu_milli, gpu_demand) workload-class key — same convention as
        workload_from_counts and every M estimator."""
        return (self.worker_cpu_milli, self.worker_gpu_demand)

    @property
    def t_submit(self) -> float:
        """submit_time_s when present (real streams); arrival_index ticks otherwise."""
        return float(self.submit_time_s) if self.submit_time_s is not None else float(self.arrival_index)

    @property
    def gpu_demand_total(self) -> float:
        return self.gang_size * self.worker_gpu_demand

    def task_class(self) -> TaskClass:
        """Worker class of this item (memory is not part of the class key)."""
        d = self.worker_gpu_demand
        return TaskClass(
            name=f"w{self.worker_cpu_milli}_{d:g}",
            cpu_milli=self.worker_cpu_milli,
            gpu_demand=d,
            whole_gpu=(d >= 1.0),
            memory_mib=0,
        )

    def duration(self) -> float:
        """Replay duration; raises unless duration_s was assigned."""
        if self.duration_s is None:
            raise ValueError(
                f"item {self.gang_id} has no duration_s — assign durations "
                "before Layer-2 replay (durations mode)"
            )
        if self.duration_s <= 0.0:
            raise ValueError(f"item {self.gang_id}: duration_s must be > 0")
        return float(self.duration_s)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        """JSONL row: required fields always; optional fields only when set
        (keeps synthetic streams byte-identical to schema-v2 files)."""
        d = {
            "gang_id": self.gang_id,
            "gang_size": self.gang_size,
            "worker_cpu_milli": self.worker_cpu_milli,
            "worker_gpu_demand": self.worker_gpu_demand,
            "arrival_index": self.arrival_index,
            "is_gang": self.is_gang,
        }
        for f in ("submit_time_s", "duration_s", "job_type", "gpu_model"):
            v = getattr(self, f)
            if v is not None:
                d[f] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StreamItem":
        unknown = set(d) - set(_ITEM_FIELDS)
        if unknown:
            raise ValueError(f"stream item has unknown fields {sorted(unknown)}")
        missing = [f for f in _ITEM_FIELDS[:6] if f not in d]
        if missing:
            raise ValueError(f"stream item missing required fields {missing}")
        return cls(**{f: d[f] for f in _ITEM_FIELDS if f in d})


def items_to_dicts(items) -> list[dict]:
    return [it.to_dict() if isinstance(it, StreamItem) else dict(it) for it in items]


def as_items(items) -> list[StreamItem]:
    """Coerce a sequence of dicts / StreamItems to StreamItems."""
    return [it if isinstance(it, StreamItem) else StreamItem.from_dict(it) for it in items]

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaskClass:
    name: str
    cpu_milli: int = 0
    memory_mib: int = 0
    gpu_demand: float = 0.0
    whole_gpu: bool = False
    gpu_spec: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.whole_gpu:
            if self.gpu_demand <= 0 or self.gpu_demand != int(self.gpu_demand):
                raise ValueError(f"whole_gpu requires a positive integer demand, got {self.gpu_demand}")
            self.gpu_demand = int(self.gpu_demand)
        if self.gpu_demand >= 1.0 and self.gpu_demand != int(self.gpu_demand):
            raise ValueError(f"combined demands like 1.5 GPUs are out of model scope, got {self.gpu_demand}")
        if not (0.0 <= self.gpu_demand):
            raise ValueError(f"gpu_demand must be >= 0, got {self.gpu_demand}")

    def __hash__(self) -> int:
        return hash((self.name, self.cpu_milli, self.memory_mib, float(self.gpu_demand), self.whole_gpu, self.gpu_spec))


@dataclass
class Workload:
    classes: tuple[TaskClass, ...]
    popularities: np.ndarray = field(default=None)

    def __post_init__(self):
        if len(self.classes) == 0:
            raise ValueError("workload needs at least one class")
        if self.popularities is None:
            self.popularities = np.full(len(self.classes), 1.0 / len(self.classes))
        if not np.isclose(self.popularities.sum(), 1.0):
            raise ValueError(f"popularities must sum to 1, got {self.popularities.sum()}")
        if (self.popularities < 0).any():
            raise ValueError("popularities must be non-negative")

    def sample(self, rng: np.random.Generator) -> TaskClass:
        return self.classes[rng.choice(len(self.classes), p=self.popularities)]

    def with_popularities(self, popularities) -> "Workload":
        """Copy with new popularities (validated by __post_init__); the source stays intact."""
        p = np.asarray(popularities, dtype=float)
        if p.shape != (len(self.classes),):
            raise ValueError(f"popularities must have shape ({len(self.classes)},), got {p.shape}")
        return Workload(self.classes, p)

    @property
    def demand_array(self) -> np.ndarray:
        return np.array([c.gpu_demand for c in self.classes], dtype=float)

    @property
    def cpu_array(self) -> np.ndarray:
        return np.array([c.cpu_milli for c in self.classes], dtype=float)


def workload_from_counts(counts) -> Workload:
    """Workload from a {(cpu_milli, gpu_demand): weight} mapping — the shared
    constructor behind the empirical/occupancy/sliding M arms."""
    if not counts:
        raise ValueError("empty class histogram")
    keys = sorted(counts)
    total = sum(counts.values())
    classes = tuple(
        TaskClass(
            name=f"w{cpu}_{d:g}",
            cpu_milli=cpu,
            gpu_demand=d,
            whole_gpu=(d >= 1.0),
            memory_mib=0,
        )
        for cpu, d in keys
    )
    pops = np.array([counts[k] / total for k in keys], dtype=float)
    return Workload(classes, pops)

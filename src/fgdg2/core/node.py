from dataclasses import dataclass
from typing import Literal

import numpy as np

from .rmath import TOL, is_empty, is_full


@dataclass
class Placement:
    gpu_indices: tuple[int, ...]
    gpu_share: float = 0.0
    kind: Literal["cpu_only", "whole", "frac"] = "cpu_only"


class Node:
    def __init__(self, node_id, num_gpus, model=None, cpu_milli=0, memory_mib=0):
        if num_gpus < 0:
            raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")
        self.id = node_id
        self.num_gpus = num_gpus
        self.model = model
        self.cpu_milli = cpu_milli
        self.memory_mib = memory_mib
        self._gpu_free = np.full(num_gpus, 1.0)
        self._used_cpu = 0
        self._used_memory = 0

    @classmethod
    def with_free(cls, node_id, free, model=None, cpu_milli=0, memory_mib=0):
        shares = np.asarray(free, dtype=float)
        if shares.ndim != 1:
            raise ValueError(f"free must be a 1-D sequence of shares, got shape {shares.shape}")
        if np.isnan(shares).any() or (shares < 0.0).any() or (shares > 1.0).any():
            raise ValueError(f"free shares must be within [0, 1], got {shares}")
        node = cls(node_id, len(shares), model=model, cpu_milli=cpu_milli, memory_mib=memory_mib)
        node._gpu_free = shares
        return node

    @property
    def gpu_free(self) -> np.ndarray:
        v = self._gpu_free.view()
        v.flags.writeable = False
        return v

    @property
    def free_slots(self) -> int:
        return int(np.count_nonzero(self._gpu_free >= 1.0 - TOL))

    @property
    def is_empty(self) -> bool:
        """Fully releasable: every GPU share free AND no cpu/memory held — the
        elastic-cloud unit (a 7/8-free node cannot be deallocated)."""
        return (
            self.free_slots == self.num_gpus
            and self.cpu_free >= self.cpu_milli
            and self.memory_free >= self.memory_mib
        )

    @property
    def used_slots(self) -> int:
        return self.num_gpus - self.free_slots

    def fits_cpu(self, cls) -> bool:
        return self._used_cpu + cls.cpu_milli <= self.cpu_milli

    def fits_memory(self, cls) -> bool:
        return self._used_memory + cls.memory_mib <= self.memory_mib

    @property
    def cpu_free(self) -> int:
        return self.cpu_milli - self._used_cpu

    @property
    def memory_free(self) -> int:
        return self.memory_mib - self._used_memory

    def fits_model(self, cls) -> bool:
        return cls.gpu_spec is None or self.model in cls.gpu_spec

    def fits_whole(self, cls) -> bool:
        return self.free_slots >= cls.gpu_demand

    def fits_non_gpu(self, cls) -> bool:
        """CPU + memory + GPU-model constraints; GPU capacity is checked separately."""
        return self.fits_cpu(cls) and self.fits_memory(cls) and self.fits_model(cls)

    def fits(self, cls) -> bool:
        if not self.fits_non_gpu(cls):
            return False
        d = cls.gpu_demand
        if d == 0:
            return True
        if d < 1.0:
            return bool((self._gpu_free >= d - TOL).any())
        return self.fits_whole(cls)

    def max_whole_workers(self, cls) -> int:
        """How many whole workers of `cls` fit on this node right now (GPU/CPU/memory).

        The single home of gang-worker capacity (AGENTS §12.4): feeds the DP
        capacities and the packed gang baselines. Raises for non-whole classes.
        """
        d = cls.gpu_demand
        if d < 1.0 or d != int(d):
            raise ValueError(f"max_whole_workers requires a whole-GPU class, got gpu_demand={d}")
        if not self.fits_model(cls):
            return 0
        caps = [self.free_slots // int(d)]
        if cls.cpu_milli > 0:
            caps.append(self.cpu_free // cls.cpu_milli)
        if cls.memory_mib > 0:
            caps.append(self.memory_free // cls.memory_mib)
        return max(0, min(caps))

    def max_frac_workers(self, cls) -> int:
        """How many FRACTIONAL workers (0 < gpu_demand < 1) of `cls` fit on this
        node right now: per-GPU floor(free_share / d), summed, capped by CPU and
        memory. Workers of one gang MAY share a GPU (the simulator allows it and
        PAI ran fractional multi-instance tasks). Raises for non-fractional classes."""
        d = float(cls.gpu_demand)
        if not 0.0 < d < 1.0:
            raise ValueError(f"max_frac_workers requires a fractional class, got gpu_demand={d}")
        if not self.fits_model(cls):
            return 0
        caps = [int(np.floor((self._gpu_free + TOL) / d).sum())]
        if cls.cpu_milli > 0:
            caps.append(self.cpu_free // cls.cpu_milli)
        if cls.memory_mib > 0:
            caps.append(self.memory_free // cls.memory_mib)
        return max(0, int(min(caps)))

    def max_workers(self, cls) -> int:
        """Gang-worker capacity for any GPU class (whole or fractional)."""
        d = float(cls.gpu_demand)
        if 0.0 < d < 1.0:
            return self.max_frac_workers(cls)
        return self.max_whole_workers(cls)

    def place(self, cls, gpu_idx=None) -> Placement:
        if not self.fits(cls):
            raise ValueError(f"task {cls.name} does not fit on node {self.id}")
        self._used_cpu += cls.cpu_milli
        self._used_memory += cls.memory_mib
        d = cls.gpu_demand
        if d >= 1.0:
            k = int(d)
            candidates = [int(g) for g in np.flatnonzero(self._gpu_free >= 1.0 - TOL)]
            if gpu_idx is not None:
                if gpu_idx not in candidates:
                    raise ValueError(f"GPU {gpu_idx} is not fully free on node {self.id}")
                chosen = [gpu_idx] + [g for g in candidates if g != gpu_idx][: k - 1]
            else:
                chosen = candidates[:k]
            self._gpu_free[chosen] = 0.0
            return Placement(tuple(chosen), kind="whole")
        if d > 0.0:
            if gpu_idx is None:
                raise ValueError(
                    "fractional placement requires an explicit gpu_idx; the policy layer decides the GPU"
                )
            if self._gpu_free[gpu_idx] < d - TOL:
                raise ValueError(f"GPU {gpu_idx} cannot host {d} on node {self.id}")
            self._gpu_free[gpu_idx] -= d
            return Placement((int(gpu_idx),), gpu_share=d, kind="frac")
        return Placement((), kind="cpu_only")

    def release(self, placement: Placement, cls) -> None:
        self._used_cpu -= cls.cpu_milli
        self._used_memory -= cls.memory_mib
        if self._used_cpu < 0 or self._used_memory < 0:
            raise ValueError(f"CPU/memory underflow on node {self.id}")
        if placement.kind == "whole":
            for g in placement.gpu_indices:
                if not is_empty(self._gpu_free[g]):
                    raise ValueError(f"GPU {g} on node {self.id} is not fully used")
                self._gpu_free[g] = 1.0
        elif placement.kind == "frac":
            g = placement.gpu_indices[0]
            if is_full(self._gpu_free[g]):
                raise ValueError(f"GPU {g} on node {self.id} is fully free, nothing to release")
            self._gpu_free[g] = min(1.0, self._gpu_free[g] + placement.gpu_share)
        else:
            return

    def total_free_gpu_shares(self) -> float:
        return float(self._gpu_free.sum())

    def copy(self) -> "Node":
        n = Node(self.id, self.num_gpus, model=self.model, cpu_milli=self.cpu_milli, memory_mib=self.memory_mib)
        n._gpu_free = self._gpu_free.copy()
        n._used_cpu = self._used_cpu
        n._used_memory = self._used_memory
        return n

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Node({self.id!r}, gpus={self.num_gpus}, model={self.model}, free={self.free_slots})"

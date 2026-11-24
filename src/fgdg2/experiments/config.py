"""Serializable experiment configs — every CLI flag is a field, so a run is
fully described by one object (and `io.artifacts.config_sha` can fingerprint
it). Scripts are thin argparse→config→runner adapters over `experiments.runner`.
"""

from dataclasses import asdict, dataclass

from ..paths import GANG_STREAMS_GLOB, RESULTS
from ..paths import NODES_CSV as _NODES
from ..paths import PODS_CSV as _PODS

PODS_CSV = str(_PODS)
NODES_CSV = str(_NODES)


@dataclass(frozen=True)
class InflationConfig:
    """Layer-1 paper calibration (was scripts/calibrate.py)."""
    policy: str
    seeds: int = 1
    fast: bool = False
    sampling: str = "replacement"          # replacement|sequential
    stop_fraction: float = 1.0             # 1.3 replicates the paper's -tune 1.3
    out: str = str(RESULTS / "experiments")
    pods_csv: str = PODS_CSV
    nodes_csv: str = NODES_CSV

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GangConfig:
    """Layer-1 gang streams (was scripts/calibrate_gang.py)."""
    policy: str
    background: str = "fgd_q"
    streams: str = GANG_STREAMS_GLOB
    seeds: int = 1
    out: str = str(RESULTS / "gang")
    pods_csv: str = PODS_CSV
    nodes_csv: str = NODES_CSV
    policy_m: str = "stream"               # stream|empirical (Layer-1 arms)
    consolidation_beta: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChurnConfig:
    """Layer-2 temporal replay (was scripts/run_churn.py)."""
    policy: str
    background: str = "fgd_q"
    streams: str = ""                      # glob (required)
    seeds: int = 1
    out: str = str(RESULTS / "churn")
    pods_csv: str = PODS_CSV
    nodes_csv: str = NODES_CSV
    policy_m: str = "stream"               # any MSpec string (workload/spec.py)
    consolidation_beta: float = 0.0        # Phase-2 elasticity knob (dp/atomic_fgd/seq_fgd)
    node_replicas: int = 1                 # oversupply regime (elasticity study)
    queue_mode: str = "scan"               # scan|strict|blocking
    queue_scan_limit: int = 256
    durations: str = "field"               # field|empirical|constant|resample
    duration_source_csv: str = PODS_CSV
    duration_s: float = 3600.0
    duration_scale: float = 1.0            # synthetic congestion knob
    # Stage-1 elasticity: empty-node eviction + reactive scale-up against a
    # finite blueprint pool (docs/ELASTICITY_PLAN.md); False = static fleet
    elastic: bool = False
    evict_after_s: float = 3600.0          # idle>T eviction timeout
    launch_latency_s: float = 300.0        # cold-start delay; billing starts at ready
    pool_max_nodes: int = 0                # supply cap; 0 = len(nodes_csv pool)
    reserve_empty: int = 0                 # keep-R-hot floor (never evict below R)
    cold_start: bool = True                # pool starts empty; False = all online
    reserve_top_n: int = 0                 # M1: pin empties for top-N queue gangs
    pin_max_hold_s: float = 21600.0        # pin timeout: expired pins drop and
                                           # the gang is embargoed from
                                           # re-pinning for one hold period
    relaunch_cooldown_s: float = 3600.0    # per-item relaunch backoff
    provision_binding: bool = False        # Kueue ProvReq emulation: launched
                                           # nodes arrive bound to their item

    def to_dict(self) -> dict:
        return asdict(self)

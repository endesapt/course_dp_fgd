"""Policy registry — the single source of truth for policy names.

Replaces the name→constructor maps that lived inside scripts (audit: registry
logic embedded in run scripts meant CLI choices and code drifted apart).
Scripts/tests must import names and builders from here.
"""

import numpy as np

from .baselines import (
    BestFitPolicy,
    ClusteringPolicy,
    DotProdPolicy,
    FirstFitPolicy,
    PackingPolicy,
    RandomPolicy,
)
from .fgd import FastFGDPolicy, FGDPolicy
from .gang import (
    AtomicSequentialFGDGang,
    AtomicSequentialRawFGDGang,
    BestFitGang,
    DPFGDGang,
    DPPackGang,
    FirstFitGang,
    MixedPolicy,
    RandomGang,
    SequentialGangPolicy,
)

# -- gang-layer policies (Layer-1/2 streams, decide_gang interface) -----------
GANG_POLICIES = (
    "dp",              # DP-FGD-Gang: optimal one-step gang placement under FGD
    "seq_fgd",         # sequential RAW FGD (dp's objective, greedy per worker)
    "atomic_fgd",      # sequential FGD-integer, all-or-nothing (spread)
    "atomic_bestfit",  # sequential BestFit (spread)
    "atomic_firstfit", # sequential FirstFit (spread)
    "atomic_random",   # sequential Random (spread)
    "atomic_bestfit_gpu",  # sequential BestFit, GPU-only residual (spread)
    "dp_pack",         # exact one-step DP, M-free packing cost, lam=1 (spread)
    "dp_pack0",        # same, lam=0: pure joint GPU-leftover BestFit (spread)
    "atomic_dotprod",  # sequential DotProd (spread; CAFGD-paper baseline)
    "atomic_packing",  # sequential GPU Packing (spread; CAFGD-paper baseline)
    "firstfit_gang",   # packed FirstFit
    "bestfit_gang",    # packed BestFit
    "random_gang",     # packed Random
)

#: spread = workers admitted atomically, may span nodes; packed = one node.
GANG_SEMANTICS = {
    "dp": "spread",
    "seq_fgd": "spread",
    "atomic_fgd": "spread",
    "atomic_bestfit": "spread",
    "atomic_firstfit": "spread",
    "atomic_random": "spread",
    "atomic_bestfit_gpu": "spread",
    "dp_pack": "spread",
    "dp_pack0": "spread",
    "atomic_dotprod": "spread",
    "atomic_packing": "spread",
    "firstfit_gang": "packed",
    "bestfit_gang": "packed",
    "random_gang": "packed",
}

# -- fractional-background policies (single-task decide) ----------------------
BACKGROUND_POLICIES = ("fgd_q", "fgd", "bestfit", "firstfit", "random", "gang")
#: "gang" is not a policy: it routes ALL fractional items (singletons and
#: fractional gangs) to the gang policy under test (no separate background).

# -- Layer-1 single-task policies (inflation benchmark, paper baselines) ------
INFLATION_POLICIES = (
    "fgd", "fgd_q", "bestfit", "firstfit", "random",
    "dotprod", "packing", "clustering",
)


def build_gang_policy(name: str, workload, cluster, rng: np.random.Generator,
                      beta: float = 0.0):
    """name in GANG_POLICIES → gang policy (spread or packed)."""
    if name == "dp":
        return DPFGDGang(workload, beta=beta)
    if name == "seq_fgd":
        return AtomicSequentialRawFGDGang(workload, beta=beta)
    if name == "atomic_fgd":
        return AtomicSequentialFGDGang(workload, beta=beta)
    if name == "atomic_bestfit":
        return SequentialGangPolicy(BestFitPolicy.from_cluster(cluster))
    if name == "atomic_firstfit":
        return SequentialGangPolicy(FirstFitPolicy())
    if name == "atomic_random":
        return SequentialGangPolicy(RandomPolicy(rng))
    if name == "atomic_bestfit_gpu":
        return SequentialGangPolicy(BestFitPolicy.from_cluster(cluster, cpu_weight=0.0))
    if name == "atomic_dotprod":
        return SequentialGangPolicy(DotProdPolicy.from_cluster(cluster))
    if name == "atomic_packing":
        return SequentialGangPolicy(PackingPolicy())
    if name == "dp_pack":
        return DPPackGang(workload, lam=1.0)
    if name == "dp_pack0":
        return DPPackGang(workload, lam=0.0)
    if name == "firstfit_gang":
        return FirstFitGang()
    if name == "bestfit_gang":
        return BestFitGang.from_cluster(cluster)
    if name == "random_gang":
        return RandomGang(rng)
    raise ValueError(f"unknown gang policy {name!r} (want one of {GANG_POLICIES})")


def build_background_policy(name: str, workload, cluster, rng):
    """name in BACKGROUND_POLICIES → single-task policy for fractional items."""
    if name == "fgd_q":
        return FastFGDPolicy(workload, quantized=True)
    if name == "fgd":
        return FastFGDPolicy(workload)
    if name == "bestfit":
        return BestFitPolicy.from_cluster(cluster)
    if name == "firstfit":
        return FirstFitPolicy()
    if name == "random":
        return RandomPolicy(rng)
    raise ValueError(
        f"unknown background policy {name!r} (want one of {BACKGROUND_POLICIES})")


def build_mixed_policy(gang_name: str, workload, cluster, rng_gang, rng_bg,
                       background: str = "fgd_q", beta: float = 0.0) -> MixedPolicy:
    """The dispatch pair: gang policy for whole items, background policy for
    fractional singletons (MixedPolicy handles the split)."""
    gang = build_gang_policy(gang_name, workload, cluster, rng_gang, beta=beta)
    if background == "gang":
        return MixedPolicy(gang, None)
    return MixedPolicy(gang, build_background_policy(background, workload, cluster, rng_bg))


def build_inflation_policy(name: str, workload, cluster, seed: int, fast=False):
    """name in INFLATION_POLICIES → single-task policy (calibrate.py surface)."""
    if name == "fgd":
        return FastFGDPolicy(workload) if fast else FGDPolicy(workload)
    if name == "fgd_q":
        return (FastFGDPolicy(workload, quantized=True) if fast
                else FGDPolicy(workload, quantized=True))
    if name == "bestfit":
        return BestFitPolicy.from_cluster(cluster)
    if name == "firstfit":
        return FirstFitPolicy()
    if name == "random":
        return RandomPolicy(np.random.default_rng(seed))
    if name == "dotprod":
        return DotProdPolicy.from_cluster(cluster)
    if name == "packing":
        return PackingPolicy()
    if name == "clustering":
        return ClusteringPolicy.from_cluster(cluster)
    raise ValueError(
        f"unknown policy {name!r} (want one of {INFLATION_POLICIES})")

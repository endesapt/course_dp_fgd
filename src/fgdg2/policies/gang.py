"""Gang policies (AGENTS.md §12.4).

Two gang semantics, never conflated (AGENTS §4.5):
- SPREAD: k workers, all-or-nothing, may span nodes (DPFGDGang, SequentialGangPolicy);
- PACKED: all k workers on ONE node (FirstFitGang, BestFitGang, RandomGang).

DPFGDGang is "optimal ONE-STEP gang placement under the FGD objective" — never
"optimal gang scheduling": the online process stays greedy in time.

Two objectives, also never conflated:
- DPFGDGang minimizes the raw CPU-aware F_n(M) delta from `gang_deltas`
  (this IS the FGD objective);
- FGDIntegerPolicy uses the theory-mode integer specialization φ(x) = x·P(K>x)
  (AGENTS §2.2; CPU enters only as a feasibility filter).
`seq_fgd` (AtomicSequentialRawFGDGang) is the clean solver ablation of DPFGDGang:
SAME raw objective, applied greedily one worker at a time — identical to DP at
k=1 by construction, divergent only for k>=2 (joint vs sequential allocation).
"""

import numpy as np

from ..core.cluster import Cluster
from ..core.gang import GangPlan, Step
from ..core.objective import raw_node_workload_fragmentation_batch
from ..core.rmath import TOL
from .baselines import BestFitPolicy


def _free_gpu_indices(node) -> np.ndarray:
    return np.flatnonzero(node.gpu_free >= 1.0 - TOL)


def _packed_plan(node, worker_cls, k: int) -> GangPlan:
    d = int(worker_cls.gpu_demand)
    free = _free_gpu_indices(node)
    return tuple(Step(node.id, int(free[j * d])) for j in range(int(k)))


def plan_for_allocation(cluster: Cluster, worker_cls, alloc) -> GangPlan:
    """Deterministic spread plan for node-wise worker counts: anchors are the
    ascending free GPUs in blocks of d; Node.place fills each worker's block."""
    d = int(worker_cls.gpu_demand)
    steps: list[Step] = []
    for node, a in zip(cluster.nodes, alloc):
        if not a:
            continue
        free = _free_gpu_indices(node)
        steps.extend(Step(node.id, int(free[j * d])) for j in range(int(a)))
    return tuple(steps)


def _cluster_arrays(cluster: Cluster, worker_cls, caps=None):
    """(R, valid, cpu_free, mem_free, num_gpus, caps) — fresh per-node state
    arrays shared by `gang_deltas` and the incremental sequential path; safe
    to mutate (copies, not views of node state). Pass `caps` to reuse an
    already-computed max_whole_workers sweep (early-feasibility checks)."""
    nodes = cluster.nodes
    n = len(nodes)
    num_gpus = np.array([nd.num_gpus for nd in nodes], dtype=int)
    gmax = int(num_gpus.max()) if n else 0
    R = np.full((n, gmax), np.nan)
    if n and gmax:
        R[np.arange(gmax)[None, :] < num_gpus[:, None]] = np.concatenate(
            [nd.gpu_free for nd in nodes]
        )
    if caps is None:
        caps = np.array([nd.max_whole_workers(worker_cls) for nd in nodes], dtype=int)
    return (
        R,
        ~np.isnan(R),
        np.array([nd.cpu_free for nd in nodes], dtype=float),
        np.array([nd.memory_free for nd in nodes], dtype=float),
        num_gpus,
        caps,
    )


def gang_deltas(cluster: Cluster, workload, worker_cls, kmax: int) -> np.ndarray:
    """Δ[i, x] = F_n(x workers placed on node i) − F_n(current), raw CPU-aware FGD
    objective, computed with kmax+1 batched calls (AGENTS §12.4).

    The per-state Δ cache: one (N, kmax+1) matrix per call. States change after
    every commit, so nothing is reused across decisions; +inf where x exceeds the
    node's whole-worker capacity.
    """
    nodes = cluster.nodes
    n = len(nodes)
    d = int(worker_cls.gpu_demand)
    delta = np.full((n, kmax + 1), np.inf)
    delta[:, 0] = 0.0
    R, valid, cpu_free, _mem, _gnum, caps = _cluster_arrays(cluster, worker_cls)
    if n == 0 or R.shape[1] == 0:
        return delta
    full = valid & (R >= 1.0 - TOL)
    full_rank = np.cumsum(full, axis=1)
    F0 = raw_node_workload_fragmentation_batch(R, valid, cpu_free, workload)
    for x in range(1, kmax + 1):
        selected = full & (full_rank <= x * d)
        R_x = np.where(selected, 0.0, R)
        cpu_x = cpu_free - x * worker_cls.cpu_milli
        F_x = raw_node_workload_fragmentation_batch(R_x, valid, cpu_x, workload)
        reachable = caps >= x
        delta[reachable, x] = (F_x - F0)[reachable]
    return delta


def _is_frac(worker_cls) -> bool:
    return 0.0 < float(worker_cls.gpu_demand) < 1.0


def _frac_best_moves(R, valid, cpu, mem, worker_cls, workload, rows, F_cur):
    """Best single fractional-worker move on each node in `rows` under the raw
    FGD objective: for every GPU with free share >= d, F after taking d from
    it; argmin over GPUs (ties -> lowest GPU index). Returns (dF, gpu, F_after)
    aligned with `rows`; dF = +inf where no GPU fits or CPU/memory are short.
    Arrays are not mutated."""
    rows = np.asarray(rows, dtype=int)
    m = len(rows)
    best = np.full(m, np.inf)
    bgpu = np.full(m, -1, dtype=int)
    fafter = np.full(m, np.nan)
    if m == 0:
        return best, bgpu, fafter
    d = float(worker_cls.gpu_demand)
    Rr, vr = R[rows], valid[rows]
    cpu_after = cpu[rows] - worker_cls.cpu_milli
    ok_res = (cpu_after >= 0) & (mem[rows] - worker_cls.memory_mib >= 0)
    for g in range(R.shape[1]):
        ok = ok_res & vr[:, g] & (Rr[:, g] >= d - TOL)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        Rt = Rr[idx].copy()
        Rt[:, g] = np.maximum(Rt[:, g] - d, 0.0)
        Ft = raw_node_workload_fragmentation_batch(Rt, vr[idx], cpu_after[idx], workload)
        dF = Ft - F_cur[rows[idx]]
        better = dF < best[idx]
        best[idx[better]] = dF[better]
        bgpu[idx[better]] = g
        fafter[idx[better]] = Ft[better]
    return best, bgpu, fafter


def gang_deltas_frac(cluster: Cluster, workload, worker_cls, kmax: int, caps):
    """Fractional analogue of `gang_deltas`: Δ[i, x] = F_n(node i after x
    fractional workers) − F_n(now), where the x workers are placed on node i
    by the WITHIN-NODE greedy FGD rule (each worker to the GPU that minimizes
    F_n). Returns (delta, gpu_seq) with gpu_seq[i, j] = GPU of the (j+1)-th
    worker on node i, so placements are nested in x (plan for x is the first x
    greedy moves). With this rule the DP over nodes is exact; the within-node
    choice is greedy — hence "optimal across nodes given the within-node rule".

    Greedy never runs out before caps[i]: taking d from a GPU with free share f
    lowers floor(f/d) by exactly one."""
    nodes = cluster.nodes
    n = len(nodes)
    delta = np.full((n, kmax + 1), np.inf)
    delta[:, 0] = 0.0
    caps = np.asarray(caps, dtype=int)
    cmax = int(min(kmax, caps.max())) if n else 0
    gpu_seq = np.full((n, max(cmax, 1)), -1, dtype=int)
    if n == 0 or cmax == 0:
        return delta, gpu_seq
    R, valid, cpu, mem, _gnum, _ = _cluster_arrays(cluster, worker_cls, caps=caps)
    d = float(worker_cls.gpu_demand)
    F0 = raw_node_workload_fragmentation_batch(R, valid, cpu, workload)
    Fcur = F0.copy()
    active = np.flatnonzero(caps >= 1)
    for x in range(1, cmax + 1):
        rows = active[caps[active] >= x]
        if rows.size == 0:
            break
        dF, g, Fa = _frac_best_moves(R, valid, cpu, mem, worker_cls, workload, rows, Fcur)
        ok = np.isfinite(dF)
        rows, g, Fa = rows[ok], g[ok], Fa[ok]
        R[rows, g] = np.maximum(R[rows, g] - d, 0.0)
        cpu[rows] -= worker_cls.cpu_milli
        mem[rows] -= worker_cls.memory_mib
        Fcur[rows] = Fa
        delta[rows, x] = Fcur[rows] - F0[rows]
        gpu_seq[rows, x - 1] = g
        active = rows
    return delta, gpu_seq


class FGDIntegerPolicy:
    """FGD-integer: φ(x) = x·P(K > x) potential descent (AGENTS §2.2).

    Theory-mode specialization of FGD for whole-GPU workloads. The potential is
    computed over the WHOLE-integer subset of the workload only: fractional classes
    do not enter φ (they do enter the raw F_n(M) objective used by DPFGDGang). CPU
    enters only as a feasibility filter, not the potential. Provides the usual
    single-task `decide`; it is the base of AtomicSequentialFGDGang.
    """

    def __init__(self, workload, beta: float = 0.0):
        """beta: consolidation penalty in GPU-share units added to the score
        when the placement leaves the node PARTIALLY free (0 < free < G) —
        the provider-binary objective (full or empty nodes are releasable-
        neutral; a partial slab is stranded capacity). beta=0 = pure FGD.
        Phase-2 elasticity knob."""
        whole = [
            (c, float(p))
            for c, p in zip(workload.classes, workload.popularities)
            if c.gpu_demand >= 1.0 and float(c.gpu_demand) == int(c.gpu_demand)
        ]
        if not whole:
            raise ValueError(
                "FGDIntegerPolicy requires at least one whole-integer class "
                f"(theory-mode specialization, AGENTS §2.2); got {len(workload.classes)} classes"
            )
        self.workload = workload
        self.beta = float(beta)
        self._demands = np.array([float(c.gpu_demand) for c, _ in whole])
        pops = np.array([p for _, p in whole])
        self._pops = pops / pops.sum()
        self._p_gt_cache: dict[int, float] = {}

    def _p_gt(self, x: int) -> float:
        v = self._p_gt_cache.get(x)
        if v is None:
            v = float(self._pops[self._demands > x].sum())
            self._p_gt_cache[x] = v
        return v

    def _phi(self, x: int) -> float:
        return x * self._p_gt(x)

    def decide(self, cluster: Cluster, cls):
        d = cls.gpu_demand
        if d < 1.0 or float(d) != int(d):
            raise ValueError(f"FGDIntegerPolicy only places whole-GPU classes, got {cls.name}")
        d = int(d)
        best = None
        best_delta = None
        for node in cluster.nodes:
            if not node.fits(cls):
                continue
            x = node.free_slots
            delta = self._phi(x - d) - self._phi(x)
            if self.beta and 0 < x - d < node.num_gpus:
                delta += self.beta
            if best_delta is None or delta < best_delta:
                best_delta, best = delta, node
        if best is None:
            return (None, None)
        return (best, int(_free_gpu_indices(best)[0]))


class SequentialGangPolicy:
    """Spread gang policy: base.decide k times on simulated placements; commit
    iff all k workers fit. Commit itself is two-phase in the runner (AGENTS §12.4) —
    the policy only proposes.

    Simulation mutates the REAL cluster and rolls back every receipt in
    reverse before returning (O(k) place/release instead of an O(N) deep copy
    per call — the copy dominated runtime on large domains like A10). The
    returned plan is then re-applied by the runner's commit on the restored
    cluster; semantics are identical to the old dry-copy version."""

    def __init__(self, base):
        self.base = base

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        steps: list[Step] = []
        receipts = []
        try:
            for _ in range(int(k)):
                node, gpu = self.base.decide(cluster, worker_cls)
                if node is None:
                    return None
                receipts.append((node, node.place(worker_cls, gpu_idx=gpu)))
                steps.append(Step(node.id, gpu))
        finally:
            for node, pl in reversed(receipts):
                node.release(pl, worker_cls)
        return tuple(steps)


class AtomicSequentialFGDGang(SequentialGangPolicy):
    """Spec preset (§3.3/§10.3): sequential FGD-integer placements, all-or-nothing."""

    def __init__(self, workload, beta: float = 0.0):
        super().__init__(FGDIntegerPolicy(workload, beta=beta))


class RawFGDTaskPolicy:
    """Single-task decider over the RAW FGD objective: argmin of the
    `gang_deltas` x=1 column (CPU-aware F_n(M) over the full workload, incl.
    fractional classes). The DPFGDGang score restricted to one worker — exists
    so `seq_fgd` is a pure solver ablation (same objective, sequential).

    beta: same partial-state penalty as DPFGDGang/FGDIntegerPolicy, charged per
    worker step on nodes the placement leaves partially free (0 < free_after <
    num_gpus; sequential semantics — a multi-worker gang on one node can pay it
    more than once, where DP charges once per plan).
    """

    def __init__(self, workload, beta: float = 0.0):
        self.workload = workload
        self.beta = float(beta)

    def decide(self, cluster: Cluster, cls):
        d = cls.gpu_demand
        if d < 1.0 or float(d) != int(d):
            raise ValueError(f"RawFGDTaskPolicy only places whole-GPU classes, got {cls.name}")
        nodes = cluster.nodes
        caps = np.array([nd.max_whole_workers(cls) for nd in nodes], dtype=int)
        feas = np.flatnonzero(caps > 0)
        if feas.size == 0:
            return (None, None)
        scores = gang_deltas(cluster, self.workload, cls, 1)[:, 1]
        if self.beta:
            free = np.array([nd.free_slots for nd in nodes], dtype=int)
            gnum = np.array([nd.num_gpus for nd in nodes], dtype=int)
            after = free - int(d)
            scores = scores + np.where((after > 0) & (after < gnum), self.beta, 0.0)
        node = nodes[int(feas[int(np.argmin(scores[feas]))])]
        return (node, int(_free_gpu_indices(node)[0]))


class AtomicSequentialRawFGDGang(SequentialGangPolicy):
    """`seq_fgd`: sequential placements under the raw FGD objective (same
    `gang_deltas` as DPFGDGang), greedy per worker, all-or-nothing. Decision-
    identical to DPFGDGang at k=1; differs only for k>=2 gangs.

    decide_gang runs the k steps on a per-node array snapshot: each step argmins
    the same x=1 marginal `gang_deltas` column RawFGDTaskPolicy.decide uses,
    then re-scores ONLY the touched node — F_n is node-separable (AGENTS §2.2),
    so every other row's marginal is provably unchanged. The generic
    place/rollback path re-scored the whole cluster per worker (O(k) batch
    evals + O(k) O(N) Python sweeps vs 2 + k 1-row evals here). Plans are
    bit-identical to the generic path (test_seq_fgd_fast_matches_generic)."""

    def __init__(self, workload, beta: float = 0.0):
        super().__init__(RawFGDTaskPolicy(workload, beta=beta))

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        d = worker_cls.gpu_demand
        if _is_frac(worker_cls):
            return self._decide_frac(cluster, worker_cls, k)
        if d < 1.0 or float(d) != int(d):
            raise ValueError(
                f"AtomicSequentialRawFGDGang only places whole-GPU classes, got {worker_cls.name}"
            )
        d = int(d)
        nodes = cluster.nodes
        if not nodes:
            return None
        # exact feasibility pre-check (same as DP's): identical workers place
        # greedily iff total whole-worker capacity >= k — doomed queue retries
        # exit after this one O(N) sweep.
        caps = np.array([nd.max_whole_workers(worker_cls) for nd in nodes], dtype=int)
        if int(caps.sum()) < k:
            return None
        wl, beta = self.base.workload, self.base.beta
        R, valid, cpu, mem, num_gpus, _ = _cluster_arrays(cluster, worker_cls, caps=caps)
        vcap = caps.copy()
        vfree = (valid & (R >= 1.0 - TOL)).sum(axis=1)
        vcpu = cpu.copy()
        vmem = mem.copy()
        Fvirt = raw_node_workload_fragmentation_batch(R, valid, cpu, wl)
        full = valid & (R >= 1.0 - TOL)
        sel = full & (np.cumsum(full, axis=1) <= d)
        Fnext = raw_node_workload_fragmentation_batch(
            np.where(sel, 0.0, R), valid, vcpu - worker_cls.cpu_milli, wl)
        score = np.where(vcap >= 1, Fnext - Fvirt, np.inf)
        steps: list[Step] = []
        for step in range(k):
            feas = np.flatnonzero(vcap >= 1)
            if feas.size == 0:
                return None
            sc = score
            if beta:
                after = vfree - d
                sc = score + np.where((after > 0) & (after < num_gpus), beta, 0.0)
            j = int(feas[int(np.argmin(sc[feas]))])
            free_idx = np.flatnonzero(R[j] >= 1.0 - TOL)
            steps.append(Step(nodes[j].id, int(free_idx[0])))
            if step == k - 1:
                break  # last worker: no next-step score needed
            R[j, free_idx[:d]] = 0.0
            vcpu[j] -= worker_cls.cpu_milli
            vmem[j] -= worker_cls.memory_mib
            vfree[j] -= d
            Fvirt[j] = Fnext[j]
            cap = int(vfree[j]) // d
            if worker_cls.cpu_milli > 0:
                cap = min(cap, int(vcpu[j]) // worker_cls.cpu_milli)
            if worker_cls.memory_mib > 0:
                cap = min(cap, int(vmem[j]) // worker_cls.memory_mib)
            vcap[j] = min(cap, caps[j])  # caps[j] carries the static model gate
            if vcap[j] >= 1:
                row = R[j].copy()
                row[np.flatnonzero(row >= 1.0 - TOL)[:d]] = 0.0
                Fnext[j] = raw_node_workload_fragmentation_batch(
                    row[None, :], valid[j : j + 1],
                    vcpu[j : j + 1] - worker_cls.cpu_milli, wl)[0]
                score[j] = Fnext[j] - Fvirt[j]
            else:
                score[j] = np.inf
        return tuple(steps)

    def _decide_frac(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        """Fractional workers: each step takes the global argmin over
        (node, GPU) of the raw-FGD marginal — exactly FGD's single-task rule
        for a fractional pod — then re-scores only the touched node.
        Ties: lowest node index, then lowest GPU index. beta is not defined
        for fractional states and must be 0."""
        if self.base.beta:
            raise ValueError("consolidation beta is undefined for fractional gang workers")
        nodes = cluster.nodes
        if not nodes:
            return None
        caps = np.array([nd.max_frac_workers(worker_cls) for nd in nodes], dtype=int)
        if int(caps.sum()) < k:
            return None
        wl = self.base.workload
        R, valid, cpu, mem, _gnum, _ = _cluster_arrays(cluster, worker_cls, caps=caps)
        d = float(worker_cls.gpu_demand)
        F = raw_node_workload_fragmentation_batch(R, valid, cpu, wl)
        score = np.full(len(nodes), np.inf)
        gsel = np.full(len(nodes), -1, dtype=int)
        Fnext = np.full(len(nodes), np.nan)
        rows = np.flatnonzero(caps >= 1)
        s, g, fa = _frac_best_moves(R, valid, cpu, mem, worker_cls, wl, rows, F)
        score[rows], gsel[rows], Fnext[rows] = s, g, fa
        steps: list[Step] = []
        for step in range(k):
            if not np.isfinite(score).any():
                return None
            j = int(np.argmin(score))
            steps.append(Step(nodes[j].id, int(gsel[j])))
            if step == k - 1:
                break
            R[j, gsel[j]] = max(R[j, gsel[j]] - d, 0.0)
            cpu[j] -= worker_cls.cpu_milli
            mem[j] -= worker_cls.memory_mib
            F[j] = Fnext[j]
            s, g, fa = _frac_best_moves(R, valid, cpu, mem, worker_cls, wl, [j], F)
            score[j], gsel[j], Fnext[j] = s[0], g[0], fa[0]
        return tuple(steps)


class DPFGDGang:
    """Exact DP for one gang decision: optimal ONE-STEP gang placement under the
    FGD objective (raw CPU-aware F_n(M); spread semantics).

    dp[i][j] = min_{0<=x<=min(cap_i, j)} dp[i-1][j-x] + Δ_i(x), O(N·k·C) with the
    per-decision Δ matrix. Deterministic tie-break: node list order, then GPU
    indices ascending.

    beta: consolidation penalty (GPU-share units) added once per node the
    plan leaves PARTIALLY free (0 < free_after < num_gpus) — pushes DP toward
    finish-or-leave-empty states; subsumes the don't-wake-empties rule since
    empty→partial pays beta while empty→full does not. Phase-2 elasticity.
    """

    def __init__(self, workload, beta: float = 0.0):
        self.workload = workload
        self.beta = float(beta)

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        nodes = cluster.nodes
        frac = _is_frac(worker_cls)
        caps = np.array([nd.max_workers(worker_cls) for nd in nodes], dtype=int)
        if int(caps.sum()) < k:
            return None
        gpu_seq = None
        if frac:
            if self.beta:
                raise ValueError("consolidation beta is undefined for fractional gang workers")
            delta, gpu_seq = self._frac_deltas(cluster, worker_cls, k, caps)
        else:
            delta = self._deltas(cluster, worker_cls, k)
        if self.beta:
            free = np.array([nd.free_slots for nd in nodes], dtype=int)
            gnum = np.array([nd.num_gpus for nd in nodes], dtype=int)
            xs = np.arange(delta.shape[1])[None, :]
            after = free[:, None] - xs * int(worker_cls.gpu_demand)
            partial = (after > 0) & (after < gnum[:, None])
            delta[:, 1:] += np.where(partial[:, 1:], self.beta, 0.0)
        feas = np.flatnonzero(caps > 0)
        if feas.size == 0:
            return None
        if k == 1:
            i = int(feas[int(np.argmin(delta[feas, 1]))])
            alloc = [0] * len(nodes)
            alloc[i] = 1
            return self._plan(cluster, worker_cls, alloc, gpu_seq)
        n = len(feas)
        # x beyond the largest node capacity is +inf in every row: restricting
        # the candidate range to 0..X leaves dp/argmin identical (first-min
        # tie-break preserved) and keeps O(N·X·k) for very large gangs.
        X = int(min(k, caps[feas].max()))
        dp = np.full((n + 1, k + 1), np.inf)
        dp[0, 0] = 0.0
        choice = np.zeros((n + 1, k + 1), dtype=int)
        xs = np.arange(X + 1)
        js = np.arange(k + 1)
        jmask = xs[:, None] <= js[None, :]
        prev_idx = np.clip(js[None, :] - xs[:, None], 0, k)
        for pos in range(1, n + 1):
            i = int(feas[pos - 1])
            shifted = np.where(jmask, dp[pos - 1][prev_idx], np.inf)
            vals = shifted + delta[i][: X + 1][:, None]
            dp[pos] = vals.min(axis=0)
            choice[pos] = vals.argmin(axis=0)
        if not np.isfinite(dp[n, k]):
            return None
        alloc = [0] * len(nodes)
        j = k
        for pos in range(n, 0, -1):
            x = int(choice[pos, j])
            alloc[int(feas[pos - 1])] = x
            j -= x
        plan = self._plan(cluster, worker_cls, alloc, gpu_seq)
        return plan if len(plan) == k else None

    def _plan(self, cluster: Cluster, worker_cls, alloc, gpu_seq) -> GangPlan:
        if gpu_seq is None:
            return plan_for_allocation(cluster, worker_cls, alloc)
        steps: list[Step] = []
        for i, (node, a) in enumerate(zip(cluster.nodes, alloc)):
            steps.extend(Step(node.id, int(gpu_seq[i, j])) for j in range(int(a)))
        return tuple(steps)

    def _frac_deltas(self, cluster: Cluster, worker_cls, k: int, caps):
        """Per-node cost Δ[i, x] for fractional workers + their GPU sequence."""
        return gang_deltas_frac(cluster, self.workload, worker_cls, k, caps)

    def _deltas(self, cluster: Cluster, worker_cls, k: int) -> np.ndarray:
        """Per-node cost Δ[i, x] of placing x workers on node i (FGD objective)."""
        return gang_deltas(cluster, self.workload, worker_cls, k)


class DPPackGang(DPFGDGang):
    """M-free joint bin-packing for a spread gang: the SAME exact one-step DP as
    DPFGDGang, with the FGD delta replaced by a workload-independent packing
    cost (abl. "is the FGD score / workload knowledge needed at all?"):

        Δ_i(0) = 0;  Δ_i(x) = after_i(x) / G_max + lam · [node i was empty],
        after_i(x) = whole free GPUs left on node i after x workers,
        G_max = max GPUs per node in the cluster (same normalizer as BestFitPolicy).

    i.e. joint BestFit on GPU leftovers of touched nodes (fill partially used
    nodes first, finish nodes) plus a penalty for opening an empty node. CPU
    and memory enter only through feasibility (max_whole_workers). The
    workload argument is ignored."""

    def __init__(self, workload=None, lam: float = 1.0):
        super().__init__(workload, beta=0.0)
        self.lam = float(lam)

    def _frac_deltas(self, cluster: Cluster, worker_cls, k: int, caps):
        raise ValueError("dp_pack is defined for whole-GPU workers only")

    def _deltas(self, cluster: Cluster, worker_cls, k: int) -> np.ndarray:
        nodes = cluster.nodes
        d = int(worker_cls.gpu_demand)
        caps = np.array([nd.max_whole_workers(worker_cls) for nd in nodes], dtype=int)
        free = np.array([nd.free_slots for nd in nodes], dtype=float)
        gmax = float(max((nd.num_gpus for nd in nodes), default=1) or 1)
        empty = np.array([nd.free_slots == nd.num_gpus for nd in nodes], dtype=bool)
        xs = np.arange(k + 1)[None, :]
        after = free[:, None] - xs * d
        delta = after / gmax + self.lam * empty[:, None]
        delta[:, 0] = 0.0
        delta[xs.repeat(len(nodes), 0) > caps[:, None]] = np.inf
        return delta


class FirstFitGang:
    """Packed gang baseline: first node in list order where all k workers fit on
    ONE node (k fully-free GPUs + CPU/memory). Packed semantics §4.5; with max 8
    GPUs/node in the trace, k=16 is impossible by topology (reported as
    `no_single_node`, not a bug)."""

    packed = True

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        for node in cluster.nodes:
            if node.max_whole_workers(worker_cls) >= int(k):
                return _packed_plan(node, worker_cls, k)
        return None


class BestFitGang:
    """Packed BestFit: argmin of the BestFit normalized residual measured on a node
    copy after hosting the gang; tie-break first node in list order. Implementation
    freedom is documented (paper baselines are 'simple extensions')."""

    packed = True

    def __init__(self, max_cpu_milli: float, max_gpu: float):
        self._bf = BestFitPolicy(max_cpu_milli, max_gpu)

    @classmethod
    def from_cluster(cls, cluster: Cluster):
        return cls(
            max(n.cpu_milli for n in cluster.nodes),
            max(n.num_gpus for n in cluster.nodes),
        )

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        k = int(k)
        best = None
        best_key = None
        for node in cluster.nodes:
            if node.max_whole_workers(worker_cls) < k:
                continue
            dry = node.copy()
            for _ in range(k - 1):
                dry.place(worker_cls)
            key = self._bf._residual(dry, worker_cls)  # residual after the k-th worker
            if best_key is None or key < best_key:
                best_key, best = key, node
        if best is None:
            return None
        return _packed_plan(best, worker_cls, k)


class RandomGang:
    """Packed random baseline: uniform among nodes that can host the whole gang on
    ONE node."""

    packed = True

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        feas = [n for n in cluster.nodes if n.max_whole_workers(worker_cls) >= int(k)]
        if not feas:
            return None
        return _packed_plan(feas[int(self.rng.integers(len(feas)))], worker_cls, k)


class MixedPolicy:
    """Dispatch for streams with fractional background (schema v2).

    Fractional singleton items (gpu_demand < 1) go to the background single-task
    policy; everything whole (gangs and whole singletons) goes to the gang policy.
    Fractional GANG workers remain out of scope and raise.
    """

    def __init__(self, gang_policy, background_policy):
        self.gang_policy = gang_policy
        self.background_policy = background_policy
        self.packed = bool(getattr(gang_policy, "packed", False))

    def decide_gang(self, cluster: Cluster, worker_cls, k: int) -> GangPlan | None:
        if self.background_policy is None:
            # background="gang": every item, fractional or whole, singleton or
            # gang, is placed by the policy under test (no hidden FGD fallback)
            return self.gang_policy.decide_gang(cluster, worker_cls, k)
        if worker_cls.gpu_demand < 1.0:
            if int(k) != 1:
                raise ValueError("fractional gang workers are out of scope (§10.5)")
            node, gpu = self.background_policy.decide(cluster, worker_cls)
            if node is None:
                return None
            return (Step(node.id, gpu),)
        return self.gang_policy.decide_gang(cluster, worker_cls, k)

"""Event heap — the single owner of release ordering in Layer-2 replay.

Releases are real events at their true deadlines, fired BETWEEN arrivals
(the post-2026-09-23 fix: lazy-flushing releases at the next arrival inflated
waits and undercounted empty time). Payloads are opaque; the runner attaches
release records.
"""

import heapq


class EventHeap:
    """Min-heap of (time, seq, payload). `seq` makes ordering deterministic
    for equal deadlines and keeps payloads out of tuple comparison."""

    __slots__ = ("_h", "_seq")

    def __init__(self):
        self._h: list[tuple[float, int, object]] = []
        self._seq = 0

    def push(self, t: float, payload) -> None:
        self._seq += 1
        heapq.heappush(self._h, (t, self._seq, payload))

    def next_time(self) -> float | None:
        return self._h[0][0] if self._h else None

    def pop(self):
        t, _, payload = heapq.heappop(self._h)
        return t, payload

    def pop_due(self, now: float):
        """Drain (time, payload) pairs with time <= now, in deadline order."""
        while self._h and self._h[0][0] <= now:
            yield self.pop()

    def payloads(self):
        """Live payloads (unordered) — the runner's running-set snapshot."""
        return [p for _, _, p in self._h]

    def __len__(self) -> int:
        return len(self._h)

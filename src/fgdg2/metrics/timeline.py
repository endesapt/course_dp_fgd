"""Timeline downsampling for artifacts (single home — was duplicated in
calibrate.py / calibrate_gang.py / run_churn.py)."""


def downsample_rows(rows, max_points: int = 600):
    """(points, stride) — always includes the last row."""
    n = len(rows)
    stride = max(1, n // max_points)
    pts = list(rows[::stride])
    if rows and pts[-1] is not rows[-1]:
        pts.append(rows[-1])
    return pts, stride


def downsample_demand_timeline(timeline, max_points: int = 600) -> dict:
    """Layer-1 (gang/inflation) artifact block: cumulative_demand x unallocated."""
    pts, stride = downsample_rows(timeline, max_points)
    return {
        "stride": stride,
        "cumulative_demand": [round(t["cumulative_demand"], 3) for t in pts],
        "unallocated_shares": [round(t["unallocated_shares"], 3) for t in pts],
        "fully_free": [int(t["fully_free"]) for t in pts],
    }


def downsample_event_timeline(timeline, max_points: int = 600) -> dict:
    """Layer-2 (churn) artifact block: event-sampled point dicts verbatim."""
    pts, stride = downsample_rows(timeline, max_points)
    return {"stride": stride, "points": pts}

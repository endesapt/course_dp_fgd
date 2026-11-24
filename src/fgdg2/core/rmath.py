
TOL = 1e-9


def is_full(r: float) -> bool:
    return r >= 1.0 - TOL


def is_empty(r: float) -> bool:
    return r <= TOL

from __future__ import annotations

import math


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..data_types import VehicleState
from ..helpers import euclidean_distance


@dataclass
class GroupingConfig:
    communication_range: float = 45.0
    merge_distance: float = 20.0
    split_distance: float = 80.0


def _build_proximity_graph(
    vehicles: list[VehicleState],
    communication_range: float,
) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {v.vehicle_id: set() for v in vehicles}
    for i, v1 in enumerate(vehicles):
        for v2 in vehicles[i + 1 :]:
            d = euclidean_distance(v1.x, v1.y, v2.x, v2.y)
            if d <= communication_range:
                graph[v1.vehicle_id].add(v2.vehicle_id)
                graph[v2.vehicle_id].add(v1.vehicle_id)
    return graph


def assign_groups(vehicles: list[VehicleState], config: GroupingConfig) -> dict[str, int]:
    """Connected-component grouping by proximity."""
    graph = _build_proximity_graph(vehicles, config.communication_range)
    visited: set[str] = set()
    out: dict[str, int] = {}
    gid = 0

    for vehicle in vehicles:
        root = vehicle.vehicle_id
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        while stack:
            cur = stack.pop()
            out[cur] = gid
            for nxt in graph[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        gid += 1

    return out


def regroup_with_history(
    vehicles: list[VehicleState],
    previous: dict[str, int] | None,
    config: GroupingConfig,
) -> dict[str, int]:
    """Simple merge/split-aware regrouping.

    - Base grouping from communication range.
    - Keep previous group id if still near at least one previous member.
    """
    current = assign_groups(vehicles, config)
    if not previous:
        return current

    by_prev: dict[int, list[str]] = defaultdict(list)
    for vid, pgid in previous.items():
        by_prev[pgid].append(vid)

    state_map = {v.vehicle_id: v for v in vehicles}
    remapped: dict[str, int] = {}
    next_gid = max(current.values(), default=-1) + 1

    for vid, cg in current.items():
        pv = previous.get(vid)
        if pv is None:
            remapped[vid] = cg
            continue

        # Preserve old group id if still close to a historical neighbor.
        keep_old = False
        for nvid in by_prev.get(pv, []):
            if nvid == vid or nvid not in state_map:
                continue
            d = euclidean_distance(
                state_map[vid].x,
                state_map[vid].y,
                state_map[nvid].x,
                state_map[nvid].y,
            )
            if d <= config.split_distance:
                keep_old = True
                break

        if keep_old:
            remapped[vid] = pv
        else:
            remapped[vid] = next_gid
            next_gid += 1

    return remapped

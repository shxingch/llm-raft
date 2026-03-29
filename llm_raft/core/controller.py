from __future__ import annotations

from dataclasses import dataclass

from ..data_types import ActionCommand, GroupPlan, VehicleState
from ..helpers import clamp


@dataclass
class ControllerConfig:
    max_accel: float = 2.5
    max_decel: float = -5.0
    max_abs_steering: float = 0.35


def _lane_change_allowed(vehicle: VehicleState) -> bool:
    if vehicle.lane_id.startswith(":"):
        return False
    available = [str(lane) for lane in vehicle.meta.get("available_lanes", [])]
    lateral_candidates = [
        lane for lane in available if lane != vehicle.lane_id and not lane.startswith(":")
    ]
    return bool(lateral_candidates)


def plan_to_actions(
    vehicles: list[VehicleState],
    vehicle_to_group: dict[str, int],
    group_plans: dict[int, GroupPlan],
    cfg: ControllerConfig,
) -> list[ActionCommand]:
    out: list[ActionCommand] = []
    for v in vehicles:
        gid = vehicle_to_group[v.vehicle_id]
        gp = group_plans[gid]
        if gp.acceleration_range is not None:
            accel = 0.5 * (gp.acceleration_range[0] + gp.acceleration_range[1])
        else:
            accel = gp.target_speed - v.speed
        accel = clamp(accel, cfg.max_decel, cfg.max_accel)
        if gp.steering_range is not None:
            steer = 0.5 * (gp.steering_range[0] + gp.steering_range[1])
        else:
            steer = gp.steering_hint
        steer = clamp(steer, -cfg.max_abs_steering, cfg.max_abs_steering)
        if not _lane_change_allowed(v):
            steer = 0.0
        out.append(
            ActionCommand(
                vehicle_id=v.vehicle_id,
                acceleration=accel,
                steering=steer,
                source="llm_raft_hybrid_controller",
            )
        )
    return out

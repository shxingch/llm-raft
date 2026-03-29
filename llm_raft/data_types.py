from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    speed: float
    lane_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeProposal:
    vehicle_id: str
    intent: str
    justification: str
    target_speed: float
    steering_hint: float
    acceleration_range: tuple[float, float] | None = None
    steering_range: tuple[float, float] | None = None


@dataclass
class GroupPlan:
    group_id: int
    plan_text: str
    target_speed: float
    steering_hint: float
    proposer_ids: list[str]
    acceleration_range: tuple[float, float] | None = None
    steering_range: tuple[float, float] | None = None


@dataclass
class ActionCommand:
    vehicle_id: str
    acceleration: float
    steering: float
    source: str = "llm_raft"

from __future__ import annotations

from dataclasses import dataclass

from ..data_types import GroupPlan, NarrativeProposal


@dataclass
class NarrativeConfig:
    default_intent: str = "keep lane"
    default_justification: str = "maintain safety and progress"


def build_fallback_proposal(
    vehicle_id: str, speed: float, cfg: NarrativeConfig
) -> NarrativeProposal:
    return NarrativeProposal(
        vehicle_id=vehicle_id,
        intent=cfg.default_intent,
        justification=cfg.default_justification,
        target_speed=max(0.0, speed),
        steering_hint=0.0,
        acceleration_range=(0.0, 2.0),
        steering_range=(0.0, 0.0),
    )


def aggregate_proposals(group_id: int, proposals: list[NarrativeProposal]) -> GroupPlan:
    if not proposals:
        return GroupPlan(
            group_id=group_id,
            plan_text="No valid proposal. Fallback to conservative keep-lane.",
            target_speed=0.0,
            steering_hint=0.0,
            proposer_ids=[],
        )

    avg_speed = sum(p.target_speed for p in proposals) / len(proposals)
    avg_steer = sum(p.steering_hint for p in proposals) / len(proposals)
    accel_ranges = [p.acceleration_range for p in proposals if p.acceleration_range is not None]
    steer_ranges = [p.steering_range for p in proposals if p.steering_range is not None]
    intents = "; ".join(f"{p.vehicle_id}:{p.intent}" for p in proposals)
    return GroupPlan(
        group_id=group_id,
        plan_text=f"Consensus narrative: {intents}",
        target_speed=avg_speed,
        steering_hint=avg_steer,
        proposer_ids=[p.vehicle_id for p in proposals],
        acceleration_range=(
            sum(r[0] for r in accel_ranges) / len(accel_ranges),
            sum(r[1] for r in accel_ranges) / len(accel_ranges),
        )
        if accel_ranges
        else None,
        steering_range=(
            sum(r[0] for r in steer_ranges) / len(steer_ranges),
            sum(r[1] for r in steer_ranges) / len(steer_ranges),
        )
        if steer_ranges
        else None,
    )

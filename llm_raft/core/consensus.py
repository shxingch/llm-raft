from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .narrative import aggregate_proposals, build_fallback_proposal, NarrativeConfig
from ..data_types import GroupPlan, NarrativeProposal, VehicleState


@dataclass
class ConsensusConfig:
    election_timeout_s: float = 0.6
    heartbeat_timeout_s: float = 0.3
    quorum_ratio: float = 0.5


class SemanticConsensus:
    """Raft-inspired, semantic-level consensus for traffic narratives."""

    def __init__(
        self,
        cfg: ConsensusConfig,
        narrative_cfg: NarrativeConfig,
        reconciler: object | None = None,
        scenario_category: str = "normal_road",
    ) -> None:
        self.cfg = cfg
        self.narrative_cfg = narrative_cfg
        self.reconciler = reconciler
        self.scenario_category = scenario_category

    def elect_leader(self, member_ids: list[str]) -> str:
        # Deterministic leader election for repeatability.
        return min(member_ids)

    def commit_group_plan(
        self,
        group_id: int,
        members: list[VehicleState],
        proposed: dict[str, NarrativeProposal],
    ) -> GroupPlan:
        if not members:
            return aggregate_proposals(group_id, [])

        member_ids = [m.vehicle_id for m in members]
        _leader = self.elect_leader(member_ids)

        valid: list[NarrativeProposal] = []
        for m in members:
            p = proposed.get(m.vehicle_id)
            if p is None:
                p = build_fallback_proposal(m.vehicle_id, m.speed, self.narrative_cfg)
            valid.append(p)

        quorum = max(1, int(len(members) * self.cfg.quorum_ratio + 0.999))
        if len(valid) < quorum:
            # Conservative fallback
            return GroupPlan(
                group_id=group_id,
                plan_text="Quorum not reached; fallback keep-lane low-speed.",
                target_speed=min((m.speed for m in members), default=0.0),
                steering_hint=0.0,
                proposer_ids=[p.vehicle_id for p in valid],
            )
        if self.reconciler is not None:
            reconcile = getattr(self.reconciler, "reconcile", None)
            if callable(reconcile):
                plan = reconcile(group_id, members, valid, self.scenario_category)
                if plan is not None:
                    return plan
        return aggregate_proposals(group_id, valid)

    def commit_all(
        self,
        vehicles: list[VehicleState],
        vehicle_to_group: dict[str, int],
        proposed: dict[str, NarrativeProposal],
    ) -> dict[int, GroupPlan]:
        grouped: dict[int, list[VehicleState]] = defaultdict(list)
        for v in vehicles:
            gid = vehicle_to_group[v.vehicle_id]
            grouped[gid].append(v)

        plans: dict[int, GroupPlan] = {}
        for gid, members in grouped.items():
            plans[gid] = self.commit_group_plan(gid, members, proposed)
        return plans

from __future__ import annotations

from dataclasses import dataclass

from .consensus import ConsensusConfig, SemanticConsensus
from .controller import ControllerConfig, plan_to_actions
from .grouping import GroupingConfig, regroup_with_history
from .narrative import NarrativeConfig, build_fallback_proposal
from ..data_types import ActionCommand, NarrativeProposal, VehicleState


@dataclass
class LLMRaftConfig:
    grouping: GroupingConfig
    consensus: ConsensusConfig
    narrative: NarrativeConfig
    controller: ControllerConfig


class LLMRaftEngine:
    def __init__(self, cfg: LLMRaftConfig, consensus_impl: SemanticConsensus | None = None) -> None:
        self.cfg = cfg
        self._prev_grouping: dict[str, int] | None = None
        self._consensus = consensus_impl or SemanticConsensus(cfg.consensus, cfg.narrative)

    def _default_proposals(self, vehicles: list[VehicleState]) -> dict[str, NarrativeProposal]:
        return {
            v.vehicle_id: build_fallback_proposal(v.vehicle_id, v.speed, self.cfg.narrative)
            for v in vehicles
        }

    def step(
        self,
        vehicles: list[VehicleState],
        proposals: dict[str, NarrativeProposal] | None = None,
    ) -> tuple[list[ActionCommand], dict[str, int]]:
        if proposals is None:
            proposals = self._default_proposals(vehicles)

        grouping = regroup_with_history(vehicles, self._prev_grouping, self.cfg.grouping)
        group_plans = self._consensus.commit_all(vehicles, grouping, proposals)
        actions = plan_to_actions(vehicles, grouping, group_plans, self.cfg.controller)
        self._prev_grouping = grouping
        return actions, grouping

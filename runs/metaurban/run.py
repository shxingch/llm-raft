#!/usr/bin/env python3
"""
MetaUrban LLM-Raft runner.

Three-stage LLM-Raft framework:
  Stage 1: Dynamic Vehicle Grouping
  Stage 2: Semantic Consensus
  Stage 3: Hybrid Control — LLM ranges + IDM execution

All designated vehicles participate:
  - Ego: custom IDM controller with LLM range constraints
  - Traffic: MetaUrban IDMPolicy with target_speed modulated by group plan

Methods:
  raft         — Full framework: grouping → consensus → coordinated IDM
  no-consensus — Grouping without consensus
  zero-shot    — Independent LLM + IDM per vehicle

Metrics:
  Task Completion Time — seconds until all designated vehicles arrive
  Average Speed        — mean speed across designated vehicles (m/s)
  Collision Rate       — fraction of trials with any collision
  Success Rate         — all arrive + no collision + within time limit
"""

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent.parent  # runs/metaurban/run.py → llm-raft/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "third_party" / "metaurban"))

from llm_raft.core.consensus import ConsensusConfig, SemanticConsensus
from llm_raft.core.grouping import GroupingConfig, regroup_with_history
from llm_raft.core.narrative import NarrativeConfig
from llm_raft.llm_runtime import (
    GroupPlanReconciler,
    LLMRuntimeConfig,
    NarrativeGenerator,
    build_scene_prompt,
)
from llm_raft.envs.metaurban_builder import (
    MetaUrbanScenarioConfig,
    build_env_config,
    build_vehicle_metaurban_env,
)
from llm_raft.data_types import VehicleState
from llm_raft.helpers import clamp

from metaurban.policy.idm_policy import IDMPolicy

PHYSICS_DT = 0.1


# ═══════════════════════════════════════════════════════════════
# Load scenario configs from YAML
# ═══════════════════════════════════════════════════════════════
import yaml

_CONFIGS_DIR = _ROOT / "configs" / "metaurban"
_ALGO_CONFIG = _ROOT / "configs" / "algorithms" / "llm_raft.yaml"

SCENARIO_MAP = {
    "dynamic_sparse": "sparse.yaml",
    "dynamic_normal": "normal.yaml",
    "dynamic_dense": "dense.yaml",
    "emergency_pedestrian": "emergency.yaml",
}


def _load_scenario(name):
    path = _CONFIGS_DIR / SCENARIO_MAP[name]
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_algo_config():
    with open(_ALGO_CONFIG, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# Ego IDM — Uses MetaUrban's IDMPolicy (with lane change)
# ═══════════════════════════════════════════════════════════════
def _create_ego_idm(ego, target_speed_kmh, seed=0):
    """Create MetaUrban IDMPolicy for ego vehicle.

    This gives ego the SAME driving capability as traffic vehicles:
    PID steering, IDM acceleration, multi-lane detection, lane changes.
    The LLM's strategic guidance is applied by modulating NORMAL_SPEED
    and ACC_FACTOR/DEACC_FACTOR on the policy instance.
    """
    policy = IDMPolicy(control_object=ego, random_seed=seed)
    policy.NORMAL_SPEED = target_speed_kmh
    policy.target_speed = target_speed_kmh
    return policy


# ═══════════════════════════════════════════════════════════════
# Traffic IDM Modulation — hybrid control for non-ego
# ═══════════════════════════════════════════════════════════════
def _modulate_traffic_idm(env, designated_ids, group_plans, grouping, ego_id):
    """Modulate designated traffic vehicles' IDMPolicy target_speed
    from the group plan. Keeps MetaUrban's IDM (lane changes,
    multi-lane detection) while applying LLM's strategic guidance.
    """
    for v in env.engine.traffic_manager.vehicles:
        vid = str(v.id)
        if vid not in designated_ids or vid == ego_id:
            continue
        veh_gid = grouping.get(vid)
        if veh_gid is None:
            continue
        plan = group_plans.get(veh_gid)
        if plan is None:
            continue
        try:
            policy = env.engine.get_policy(v.id)
            if policy is None:
                continue
            # Modulate target speed from consensus plan
            # Must set NORMAL_SPEED (not target_speed) because lane_follow() resets it
            if plan.target_speed > 0:
                policy.NORMAL_SPEED = plan.target_speed * 3.6  # m/s → km/h
                policy.target_speed = plan.target_speed * 3.6
            # Modulate acceleration constraints from plan ranges
            if plan.acceleration_range is not None:
                policy.ACC_FACTOR = max(0.1, plan.acceleration_range[1])
                policy.DEACC_FACTOR = min(-0.1, plan.acceleration_range[0])
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
@dataclass
class EpisodeStats:
    task_completion_time_s: float
    average_speed: float
    collision: bool
    collision_type: str
    success: bool
    steps: int
    arrive_dest: bool
    fail_reason: str
    designated_count: int
    designated_arrived: int


# ═══════════════════════════════════════════════════════════════
def _build_env(scenario_cfg, use_gui):
    cfg = MetaUrbanScenarioConfig(
        scenario_name="experiment",
        env_type="dynamic",
        num_scenarios=scenario_cfg.get("num_scenarios", 400),
        start_seed=scenario_cfg.get("start_seed", 0),
        map_id=scenario_cfg.get("map_id", "SXS"),
        traffic_density=scenario_cfg["traffic_density"],
        object_density=scenario_cfg.get("object_density", 0.15),
        crswalk_density=scenario_cfg.get("crswalk_density", 0.2),
        spawn_human_num=scenario_cfg.get("spawn_human_num", 4),
        max_actor_num=max(40, scenario_cfg["target_vehicle_range"][1] + 10),
        use_render=use_gui,
        horizon=scenario_cfg.get("horizon", 1000),
    )
    env_cls = build_vehicle_metaurban_env("dynamic")

    class SafeEnv(env_cls):
        def _is_out_of_road(self, vehicle):
            if hasattr(vehicle, "navigation") and vehicle.navigation is not None:
                lat = abs(getattr(vehicle.navigation, "current_lateral", 0.0))
                if lat > 15.0:
                    return True
            return False

    ec = build_env_config(cfg)
    ec["crash_vehicle_done"] = True
    ec["crash_object_done"] = False
    ec["crash_human_done"] = True
    ec["out_of_route_done"] = False
    ec["max_lateral_dist"] = 15.0
    return SafeEnv(ec)


def _collect_vehicle_states(env, designated_ids, rng):
    states = []
    ego = env.agent
    ep = ego.position
    states.append(
        VehicleState(
            vehicle_id=str(ego.id),
            x=float(ep[0]),
            y=float(ep[1]),
            speed=float(ego.speed),
            lane_id=str(getattr(ego, "lane_index", "") or ""),
            meta={"is_ego": True, "distance_to_ego": 0.0},
        )
    )
    for v in env.engine.traffic_manager.vehicles:
        vid = str(v.id)
        if vid in designated_ids and vid != str(ego.id):
            try:
                p = v.position
                states.append(
                    VehicleState(
                        vehicle_id=vid,
                        x=float(p[0]) + rng.gauss(0, 0.3),
                        y=float(p[1]) + rng.gauss(0, 0.3),
                        speed=max(0.0, float(v.speed) + rng.gauss(0, 0.2)),
                        lane_id=str(getattr(v, "lane_index", "") or ""),
                        meta={
                            "is_ego": False,
                            "distance_to_ego": math.hypot(
                                float(p[0]) - float(ep[0]), float(p[1]) - float(ep[1])
                            ),
                        },
                    )
                )
            except Exception:
                pass
    return states


def _build_scene_context(env, designated_ids, scenario_name):
    ego = env.agent
    lines = [
        f"## MetaUrban {scenario_name}",
        f"ego at ({ego.position[0]:.1f}, {ego.position[1]:.1f}), speed {ego.speed:.1f} m/s",
    ]
    count = 0
    for v in env.engine.traffic_manager.vehicles:
        if count >= 6:
            break
        try:
            dx = float(v.position[0]) - float(ego.position[0])
            dy = float(v.position[1]) - float(ego.position[1])
            dist = math.hypot(dx, dy)
            if dist < 50:
                tag = "designated" if str(v.id) in designated_ids else "bg"
                lines.append(
                    f"- {v.id} ({tag}) rel=({dx:.1f},{dy:.1f}) sp={v.speed:.1f} d={dist:.0f}m"
                )
                count += 1
        except Exception:
            pass
    return "\n".join(lines)


def _build_framework(mode, scene_prompt: str = ""):
    llm_cfg = LLMRuntimeConfig.from_env(scene_prompt=scene_prompt)
    narrator = NarrativeGenerator(llm_cfg)
    algo = _load_algo_config()
    g = algo.get("grouping", {})
    grouping_cfg = GroupingConfig(
        communication_range=g.get("communication_range", 30.0),
        merge_distance=g.get("merge_distance", 20.0),
        split_distance=g.get("split_distance", 40.0),
    )
    consensus_cfg = ConsensusConfig()
    narrative_cfg = NarrativeConfig()
    if mode == "raft":
        consensus = SemanticConsensus(
            consensus_cfg,
            narrative_cfg,
            reconciler=GroupPlanReconciler(llm_cfg),
            scenario_category="metaurban",
        )
        return narrator, grouping_cfg, consensus
    elif mode == "no-consensus":
        consensus = SemanticConsensus(
            consensus_cfg, narrative_cfg, reconciler=None, scenario_category="metaurban"
        )
        return narrator, grouping_cfg, consensus
    else:
        return narrator, grouping_cfg, None


# ═══════════════════════════════════════════════════════════════
# Episode Runner
# ═══════════════════════════════════════════════════════════════
def run_episode(env, scenario_cfg, mode, trial_seed, decision_interval=10):
    rng = random.Random(trial_seed)
    target_range = scenario_cfg["target_vehicle_range"]
    action_noise = scenario_cfg.get("action_noise", 0.03)
    target_speed = scenario_cfg.get("speed_limit_kmh", 40)
    scene_prompt = build_scene_prompt(scenario_cfg)

    narrator, grouping_cfg, consensus = _build_framework(mode, scene_prompt)
    prev_grouping = None

    # Ego IDM controller — uses MetaUrban's IDMPolicy (with lane change)
    ego_idm = None  # created after env.reset()

    # Reset env
    obs, info = None, {}
    for offset in range(50):
        seed = trial_seed + offset
        obs, info = env.reset(seed=seed)
        all_traffic = list(env.engine.traffic_manager.vehicles)
        if not (target_range[0] <= len(all_traffic) <= target_range[1]):
            continue
        ego_pos = env.agent.position
        if all(
            math.hypot(
                float(v.position[0]) - float(ego_pos[0]),
                float(v.position[1]) - float(ego_pos[1]),
            )
            >= 3.0
            for v in all_traffic
        ):
            break

    ego = env.agent
    ego_id = str(ego.id)

    # Create ego IDMPolicy (after reset so vehicle is initialized)
    ego_idm = _create_ego_idm(ego, target_speed, seed=trial_seed)

    # Designated vehicles (30-70%)
    all_traffic = list(env.engine.traffic_manager.vehicles)
    candidates = [v for v in all_traffic if str(v.id) != ego_id]
    ratio = rng.uniform(0.3, 0.7)
    des_count = max(1, int(round(len(candidates) * ratio)))
    rng.shuffle(candidates)
    designated_ids = {ego_id}
    for v in candidates[:des_count]:
        designated_ids.add(str(v.id))

    # Set traffic IDM base speed
    # Must set NORMAL_SPEED (class default), not target_speed (resets each step)
    traffic_speed = target_speed
    for v in all_traffic:
        try:
            policy = env.engine.get_policy(v.id)
            if policy and hasattr(policy, "NORMAL_SPEED"):
                policy.NORMAL_SPEED = traffic_speed
                policy.target_speed = traffic_speed
        except Exception:
            pass

    ego_accel_range = None

    # ── Main loop ────────────────────────────────────────────
    max_steps = scenario_cfg.get("horizon", 1000)
    speed_sum, speed_n = 0.0, 0
    collision = False
    collision_type = ""
    ego_arrived = False
    fail_reason = ""
    total_steps = 0

    for step in range(max_steps):
        total_steps = step + 1

        # Track ALL designated speeds
        for v in all_traffic:
            if str(v.id) in designated_ids:
                try:
                    speed_sum += float(v.speed)
                    speed_n += 1
                except Exception:
                    pass
        speed_sum += float(ego.speed)
        speed_n += 1

        # ── LLM decision point ──────────────────────────────
        if step % decision_interval == 0:
            if mode in ("raft", "no-consensus") and consensus is not None:
                vehicles = _collect_vehicle_states(env, designated_ids, rng)
                context = _build_scene_context(env, designated_ids, "metaurban")

                proposals = {}
                for vs in vehicles:
                    neighbors = [n for n in vehicles if n.vehicle_id != vs.vehicle_id]
                    proposals[vs.vehicle_id] = narrator.generate(
                        vehicle=vs,
                        neighbors=neighbors,
                        env_context=context,
                    )

                grouping = regroup_with_history(vehicles, prev_grouping, grouping_cfg)
                prev_grouping = grouping
                group_plans = consensus.commit_all(vehicles, grouping, proposals)

                # Ego ranges from group plan
                ego_gid = grouping.get(ego_id, 0)
                ego_plan = group_plans.get(ego_gid)
                if ego_plan is not None:
                    ego_accel_range = ego_plan.acceleration_range

                # Modulate ALL designated traffic IDM from group plans
                _modulate_traffic_idm(env, designated_ids, group_plans, grouping, ego_id)

            elif mode == "zero-shot":
                ego_state = VehicleState(
                    vehicle_id=ego_id,
                    x=float(ego.position[0]),
                    y=float(ego.position[1]),
                    speed=float(ego.speed),
                    lane_id="",
                    meta={"is_ego": True},
                )
                proposal = narrator.generate(
                    vehicle=ego_state,
                    neighbors=[],
                    env_context="",
                )
                ego_accel_range = proposal.acceleration_range

                # Zero-shot: each traffic vehicle gets independent proposal
                for v in all_traffic:
                    vid = str(v.id)
                    if vid not in designated_ids or vid == ego_id:
                        continue
                    try:
                        vs = VehicleState(
                            vehicle_id=vid,
                            x=float(v.position[0]),
                            y=float(v.position[1]),
                            speed=float(v.speed),
                            lane_id="",
                            meta={"is_ego": False},
                        )
                        p = narrator.generate(
                            vehicle=vs,
                            neighbors=[],
                            env_context="",
                        )
                        policy = env.engine.get_policy(v.id)
                        if policy and p.target_speed > 0:
                            policy.target_speed = p.target_speed * 3.6
                    except Exception:
                        pass

        # ── Ego: IDM Hybrid Control ───────────────────────────
        # IDMPolicy (with lane change); LLM ranges modulate ACC/DEACC
        if ego_accel_range is not None:
            ego_idm.ACC_FACTOR = max(0.1, ego_accel_range[1])
            ego_idm.DEACC_FACTOR = min(-0.1, ego_accel_range[0])
        ego_action = list(ego_idm.act())
        ego_action[0] += rng.gauss(0, action_noise)
        ego_action[1] += rng.gauss(0, action_noise * 0.3)
        ego_action[0] = clamp(ego_action[0], -1.0, 1.0)
        ego_action[1] = clamp(ego_action[1], -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(ego_action)

        if info.get("crash_vehicle") or info.get("crash_human"):
            collision = True
            collision_type = "vehicle" if info.get("crash_vehicle") else "human"
            fail_reason = "collision"
            break
        if info.get("crash_object") and not collision:
            collision = True
            collision_type = "object"

        if info.get("arrive_dest"):
            ego_arrived = True
            break
        if terminated or truncated:
            fail_reason = "out_of_road" if info.get("out_of_road") else "terminated"
            break

    # ── Metrics ──────────────────────────────────────────────
    time_s = round(total_steps * PHYSICS_DT, 1)
    avg_speed = round(speed_sum / speed_n, 3) if speed_n else 0.0
    designated_arrived = 0
    for v in all_traffic:
        vid = str(v.id)
        if vid not in designated_ids:
            continue
        try:
            if vid == ego_id:
                if ego_arrived:
                    designated_arrived += 1
            else:
                rc = float(getattr(getattr(v, "navigation", None), "route_completion", 0.0))
                if rc > 0.95:
                    designated_arrived += 1
        except Exception:
            pass

    all_arrived = designated_arrived >= len(designated_ids)
    success = all_arrived and not collision and time_s <= 100.0

    return EpisodeStats(
        task_completion_time_s=time_s,
        average_speed=avg_speed,
        collision=collision,
        collision_type=collision_type,
        success=success,
        steps=total_steps,
        arrive_dest=ego_arrived,
        fail_reason=fail_reason,
        designated_count=len(designated_ids),
        designated_arrived=designated_arrived,
    )


# ═══════════════════════════════════════════════════════════════
def run_scenario(scenario_name, num_trials, mode, use_gui, decision_interval=10):
    scenario_cfg = _load_scenario(scenario_name)
    print(f"\n{'=' * 60}")
    print(f"  {scenario_name} | mode={mode} | trials={num_trials}")
    noise = scenario_cfg.get("action_noise", 0.03)
    print(f"  speed_limit={scenario_cfg.get('speed_limit_kmh')} km/h | noise={noise}")
    print(f"{'=' * 60}")

    env = _build_env(scenario_cfg, use_gui)
    stats_list = []
    for trial in range(num_trials):
        trial_seed = scenario_cfg.get("start_seed", 0) + trial * 7
        try:
            stats = run_episode(
                env,
                scenario_cfg,
                mode,
                trial_seed,
                decision_interval=decision_interval,
            )
            stats_list.append(stats)
            tag = "OK" if stats.success else "FAIL"
            col = f"col={stats.collision_type}" if stats.collision else "col=no"
            print(
                f"  [{tag}] t={stats.task_completion_time_s}s sp={stats.average_speed:.1f} "
                f"{col} arr={stats.arrive_dest} "
                f"des={stats.designated_arrived}/{stats.designated_count} "
                f"reason={stats.fail_reason}"
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()
    env.close()

    if not stats_list:
        print("  No trials completed!")
        return
    n = len(stats_list)
    print(f"\n{'─' * 50}")
    print(f"  {scenario_name} [{mode}] ({n} trials):")
    print(f"  Task Completion Time:  {sum(s.task_completion_time_s for s in stats_list) / n:.1f}s")
    print(f"  Average Speed:         {sum(s.average_speed for s in stats_list) / n:.2f} m/s")
    col_r = sum(1 for s in stats_list if s.collision) / n * 100
    print(
        f"  Collision Rate:        {col_r:.1f}% "
        f"(veh={sum(1 for s in stats_list if s.collision_type == 'vehicle') / n * 100:.0f}% "
        f"obj={sum(1 for s in stats_list if s.collision_type == 'object') / n * 100:.0f}% "
        f"ped={sum(1 for s in stats_list if s.collision_type == 'human') / n * 100:.0f}%)"
    )
    print(f"  Success Rate:          {sum(1 for s in stats_list if s.success) / n * 100:.1f}%")
    print(f"  Ego Arrival Rate:      {sum(1 for s in stats_list if s.arrive_dest) / n * 100:.1f}%")
    print(f"{'─' * 50}")


def main():
    parser = argparse.ArgumentParser(description="MetaUrban LLM-Raft Runner")
    parser.add_argument(
        "--scenario", default="dynamic_sparse", choices=list(SCENARIO_MAP.keys()) + ["all"]
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--mode", default="raft", choices=["raft", "no-consensus", "zero-shot"])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--decision-interval", type=int, default=10)
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export OPENAI_BASE_URL='https://api.openai.com/v1'  # or your endpoint")
        sys.exit(1)
    scenarios = list(SCENARIO_MAP.keys()) if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        run_scenario(
            sc,
            args.trials,
            args.mode,
            args.gui,
            decision_interval=args.decision_interval,
        )


if __name__ == "__main__":
    main()

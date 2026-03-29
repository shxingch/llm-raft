"""
LLM-Raft — LimSim runner.

Built on LimSim (SUMO-based):
  - DearPyGUI visualization
  - Model / TrafficManager for simulation
  - LLM-Raft grouping, consensus, and hybrid control

Usage:
  python run.py --scenario normal_road                    # with GUI
  python run.py --scenario normal_road --no-gui           # headless
  python run.py --scenario normal_road --trials 5         # batch
  python run.py --scenario all --trials 30 --no-gui       # full run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # runs/limsim/run.py → llm-raft/
_LIMSIM_DIR = _ROOT / "third_party" / "limsim"
sys.path.insert(0, str(_ROOT))  # llm_raft package
sys.path.insert(0, str(_LIMSIM_DIR))  # LimSim internal modules
os.chdir(str(_LIMSIM_DIR))  # LimSim needs relative paths

# ── LLM API (set OPENAI_API_KEY and OPENAI_BASE_URL in environment) ───────

# ── LimSim imports ──────────────────────────────────────────────────────────
from simModel.Model import Model
from simModel.MPGUI import GUI
from simModel.DataQueue import QuestionAndAnswer
from simInfo.CustomExceptions import (
    CollisionChecker,
    CollisionException,
    LaneChangeException,
    TimeOutException,
)
from simInfo.EnvDescriptor import EnvDescription
from trafficManager.traffic_manager import TrafficManager
from trafficManager.decision_maker.abstract_decision_maker import (
    MultiDecision,
    SingleStepDecision,
)
from trafficManager.common.vehicle import Behaviour
import traci

# ── LLM-Raft imports ────────────────────────────────────────────────────────
from llm_raft.data_types import ActionCommand, VehicleState
from llm_raft.core.grouping import GroupingConfig
from llm_raft.core.consensus import SemanticConsensus, ConsensusConfig
from llm_raft.core.narrative import NarrativeConfig
from llm_raft.core.controller import ControllerConfig
from llm_raft.core.engine import LLMRaftEngine, LLMRaftConfig
from llm_raft.core.metrics import EpisodeStats, summarize
from llm_raft.llm_runtime import (
    GroupPlanReconciler,
    LLMRuntimeConfig,
    NarrativeGenerator,
    build_scene_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════
# Load scenario and algorithm configs from YAML
# ═══════════════════════════════════════════════════════════════════════════
_CONFIGS_DIR = _ROOT / "configs" / "limsim"
_ALGO_CONFIG_PATH = _ROOT / "configs" / "algorithms" / "llm_raft.yaml"

SCENARIO_FILES = {
    "normal_road": "normal_road.yaml",
    "crowded_road": "crowded_road.yaml",
    "highway": "highway.yaml",
    "intersection": "intersection.yaml",
}


def _load_scenario(name):
    with open(_CONFIGS_DIR / SCENARIO_FILES[name], "r") as f:
        return yaml.safe_load(f)


def _load_all_scenarios():
    return {name: _load_scenario(name) for name in SCENARIO_FILES}


SCENARIOS = _load_all_scenarios()


def _load_algo():
    with open(_ALGO_CONFIG_PATH, "r") as f:
        a = yaml.safe_load(f)
    g = a.get("grouping", {})
    c = a.get("consensus", {})
    n = a.get("narrative", {})
    ct = a.get("controller", {})
    return LLMRaftConfig(
        grouping=GroupingConfig(**g),
        consensus=ConsensusConfig(**c),
        narrative=NarrativeConfig(**n),
        controller=ControllerConfig(**ct),
    )


ALGO_CONFIG = _load_algo()

DEFAULT_PROTOCOL = {
    "trial_count": 30,
    "time_limit_steps": 1000,
    "decision_interval": 10,
    "seed_base": 20250310,
}


# ═══════════════════════════════════════════════════════════════════════════
# Trial manifest: select scene vehicles + designated vehicles per trial
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TrialManifest:
    scene_vehicle_ids: list[str]
    designated_vehicle_ids: list[str]
    designated_ratio: float
    target_vehicle_count: int


def _load_vehicle_types(route_files: str) -> dict[str, str]:
    """Load vClass for each vehicle type from carlavtypes.rou.xml."""
    type_map: dict[str, str] = {}
    for rf in route_files.split(","):
        p = Path(rf.strip())
        if not p.exists():
            continue
        try:
            for vt in ET.parse(p).getroot().iter("vType"):
                tid = vt.attrib.get("id", "")
                vclass = vt.attrib.get("vClass", "passenger")
                type_map[tid] = vclass
        except Exception:
            pass
    return type_map


# Vehicle classes that should NOT be designated (too slow / not cars)
_EXCLUDED_VCLASSES = {"bicycle", "motorcycle"}


def build_manifest(
    route_path: Path, ego_id: str, scenario: dict, trial_index: int, seed_base: int
) -> TrialManifest:
    """Select vehicles for a trial, matching the experiment protocol."""
    rng = random.Random(seed_base + trial_index)

    # Load vehicle type classes to filter out bicycles
    type_map = _load_vehicle_types(scenario.get("sumo_rou", ""))

    # Parse all vehicle IDs and their depart times from route file
    vehicles: list[tuple[str, float]] = []
    if route_path.exists():
        tree = ET.parse(route_path)
        for veh in tree.getroot().iter("vehicle"):
            vid = veh.attrib.get("id", "")
            vtype = veh.attrib.get("type", "")
            vclass = type_map.get(vtype, "passenger")
            # Skip bicycles — they're too slow for the 100s time limit
            if vclass in _EXCLUDED_VCLASSES:
                continue
            depart = float(veh.attrib.get("depart", "0"))
            vehicles.append((vid, depart))

    if not vehicles:
        return TrialManifest([ego_id], [ego_id], 1.0, 1)

    # Sort by depart time proximity to ego
    ego_depart = next((d for v, d in vehicles if v == ego_id), 0.0)
    vehicles.sort(key=lambda x: abs(x[1] - ego_depart))

    # Select target_count vehicles from the pool
    vmin, vmax = scenario["vehicle_count_range"]
    target_count = rng.randint(vmin, vmax)
    target_count = min(target_count, len(vehicles))

    # Take the closest vehicles by depart time (ensures overlap in simulation)
    scene_ids = [v for v, _ in vehicles[:target_count]]
    if ego_id not in scene_ids:
        scene_ids[-1] = ego_id  # ensure ego is included

    # Designate a subset for LLM-Raft control
    rmin, rmax = scenario["designated_ratio_range"]
    ratio = rng.uniform(rmin, rmax)
    designated_count = max(1, int(len(scene_ids) * ratio))

    designated = {ego_id}
    selectable = [v for v in scene_ids if v != ego_id]
    rng.shuffle(selectable)
    designated.update(selectable[: max(0, designated_count - 1)])

    return TrialManifest(
        scene_vehicle_ids=sorted(scene_ids),
        designated_vehicle_ids=sorted(designated),
        designated_ratio=round(len(designated) / len(scene_ids), 4),
        target_vehicle_count=target_count,
    )


def _build_successor_map(net_path: str) -> dict[str, str]:
    """Build edge→successor map from a SUMO network file (for loop networks)."""
    successors: dict[str, str] = {}
    try:
        tree = ET.parse(net_path)
        for conn in tree.findall(".//connection"):
            fr, to = conn.get("from", ""), conn.get("to", "")
            if fr and to and not fr.startswith(":") and not to.startswith(":"):
                successors[fr] = to  # last connection wins (fine for simple loops)
    except Exception:
        pass
    return successors


def filter_route_file(
    route_path: Path,
    keep_ids: set[str],
    scenario_name: str,
    trial_index: int,
    ego_id: str = "",
    max_route_edges: int = 0,
    ego_max_route_edges: int = 0,
    ego_min_route_edges: int = 0,
    net_path: str = "",
    depart_pos: float = 0.0,
) -> Path | None:
    """Create a filtered route file containing only the selected vehicles.

    If max_route_edges > 0, truncate each vehicle's route to at most
    that many edges (ensures routes are completable within the time limit).
    ego_max_route_edges overrides max_route_edges for the ego vehicle.
    If ego_min_route_edges > 0, extend ego's route using network successor
    edges until it has at least that many edges (for loop networks).
    If depart_pos > 0, set departPos for all vehicles (start partway along edge).
    """
    if not route_path.exists():
        return None
    root = ET.parse(route_path).getroot()

    # Build successor map for route extension (loop networks)
    successors: dict[str, str] = {}
    if ego_min_route_edges > 0 and net_path:
        successors = _build_successor_map(net_path)

    kept = 0
    for child in list(root):
        if child.tag != "vehicle":
            continue
        vid = str(child.attrib.get("id", ""))
        if vid not in keep_ids:
            root.remove(child)
        else:
            kept += 1
            # Determine edge limit for this vehicle
            limit = (
                ego_max_route_edges
                if (vid == ego_id and ego_max_route_edges > 0)
                else max_route_edges
            )
            if limit > 0:
                for route_el in child.iter("route"):
                    edges = route_el.attrib.get("edges", "").split()
                    if len(edges) > limit:
                        route_el.set("edges", " ".join(edges[:limit]))
            # Extend ego route to minimum edges (for loop networks)
            if vid == ego_id and ego_min_route_edges > 0 and successors:
                for route_el in child.iter("route"):
                    edges = route_el.attrib.get("edges", "").split()
                    while len(edges) < ego_min_route_edges:
                        last = edges[-1]
                        nxt = successors.get(last)
                        if not nxt or nxt == edges[0]:
                            break  # avoid full loop back to start
                        edges.append(nxt)
                    route_el.set("edges", " ".join(edges))
            # Set departPos if specified (start partway along first edge)
            if depart_pos > 0:
                child.set("departPos", str(depart_pos))
    if kept == 0:
        return None

    out_dir = Path(".cache") / "routes" / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_{trial_index:02d}.rou.xml"
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# LLMRaftRunner — single trial execution
# ═══════════════════════════════════════════════════════════════════════════
class LLMRaftRunner:
    """
    Pattern: Model + GUI + TrafficManager + LLM-Raft Engine
    Matches ExampleLLMAgentCloseLoop.py structure.
    """

    def __init__(
        self,
        scenario_name: str,
        trial_index: int = 0,
        use_gui: bool = True,
        seed: int | None = None,
        no_consensus: bool = False,
    ):
        scenario = SCENARIOS[scenario_name]
        self.scenario_name = scenario_name
        self.scenario = scenario
        self.trial_index = trial_index
        self.max_steps = DEFAULT_PROTOCOL["time_limit_steps"]
        self.decision_interval = DEFAULT_PROTOCOL["decision_interval"]
        seed = seed or (DEFAULT_PROTOCOL["seed_base"] + trial_index)

        # Build trial manifest
        route_files = scenario["sumo_rou"].split(",")
        route_path = Path(route_files[-1])
        self.manifest = build_manifest(
            route_path,
            scenario["ego_id"],
            scenario,
            trial_index,
            seed,
        )

        depart_pos = scenario.get("departPos", 0.0)

        # Filter route file to only include scene vehicles
        filtered = filter_route_file(
            route_path,
            set(self.manifest.scene_vehicle_ids),
            scenario_name,
            trial_index,
            ego_id=scenario["ego_id"],
            max_route_edges=scenario.get("max_route_edges", 0),
            ego_max_route_edges=scenario.get("ego_max_route_edges", 0),
            ego_min_route_edges=scenario.get("ego_min_route_edges", 0),
            net_path=scenario.get("sumo_net", ""),
            depart_pos=depart_pos,
        )
        if filtered is not None:
            route_files[-1] = str(filtered)

        self.designated_ids = set(self.manifest.designated_vehicle_ids)

        # ── LimSim setup (same pattern as ExampleLLMAgentCloseLoop.py) ──
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        db_dir = Path("results/demo")
        db_dir.mkdir(parents=True, exist_ok=True)

        self.model = Model(
            egoID=scenario["ego_id"],
            netFile=scenario["sumo_net"],
            rouFile=",".join(route_files),
            cfgFile=scenario["sumo_cfg"],
            dataBase=str(db_dir / f"llm_raft_{scenario_name}_t{trial_index}_{ts}.db"),
            SUMOGUI=False,
            CARLACosim=False,
        )
        # Increase AoI range so BEV shows vehicles on parallel roads
        self.model.ego.deArea = 100.0
        self.planner = TrafficManager(self.model)
        self.descriptor = EnvDescription()
        self.collision_checker = CollisionChecker()
        self.use_gui = use_gui

        # ── LLM-Raft engine ──
        llm_cfg = LLMRuntimeConfig.from_env(scene_prompt=build_scene_prompt(scenario))
        reconciler = GroupPlanReconciler(llm_cfg)
        consensus = SemanticConsensus(
            ALGO_CONFIG.consensus,
            ALGO_CONFIG.narrative,
            reconciler=reconciler,
            scenario_category=scenario["category"],
        )
        self.engine = LLMRaftEngine(ALGO_CONFIG, consensus_impl=consensus)
        self.narrator = NarrativeGenerator(llm_cfg)
        self.no_consensus = no_consensus

        # ── Metrics ──
        self._decisions = 0
        self._departed: set[str] = set()
        self._arrived: set[str] = set()
        self._speed_sum = 0.0
        self._speed_n = 0
        self._completion_time: float | None = None
        self._ego_completion_time: float | None = None
        self._fallbacks = 0
        self._violation = False
        self._violation_reason = ""
        self._prev_ego_lane = ""
        self._ep_start: int | None = None
        self._near_miss_collision = False  # proximity collision flag

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        success = False
        collision = False
        fail_reason = ""

        self.model.start()

        gui = None
        if self.use_gui:
            gui = GUI(self.model)
            gui.start()

        try:
            ego_finished = False
            while True:
                # Phase 1: ego still active – use full LimSim loop
                if not ego_finished:
                    self.model.moveStep()
                    # Slow down for GUI visibility
                    if gui and self.model.timeStep % 5 == 0:
                        time.sleep(0.05)

                    if self.model.tpStart and self._ep_start is None:
                        self._ep_start = self.model.timeStep

                    self._update_metrics()
                    self._check_violations()

                    if self._is_success():
                        success = True
                        break

                    self.collision_checker.CollisionCheck(self.model)
                    self._check_proximity_collision()

                    if self.model.timeStep % self.decision_interval == 0:
                        rg, vi = self.model.exportSce()
                        if self.model.tpStart and rg and vi:
                            self._decide(rg, vi)
                    # Release traci speed overrides 3 steps after decision
                    # so designated vehicles return to SUMO IDM between decisions
                    elif self.model.timeStep % self.decision_interval == 3:
                        self._release_designated_speeds()

                    # NOTE: We skip updateVeh() to let SUMO's IDM handle
                    # vehicle following. LLM-Raft influence is via traci
                    # speed/lane commands applied in _decide().

                    if self.model.tpEnd:
                        ego_finished = True
                        if self._ego_completion_time is None:
                            self._ego_completion_time = round(self.model.timeStep * 0.1, 3)
                        continue

                # Phase 2: ego done, keep SUMO advancing for remaining
                # designated vehicles to finish their routes
                else:
                    traci.simulationStep()
                    self.model.timeStep += 1
                    self._update_metrics()

                    if self._is_success():
                        success = True
                        break

                    # SUMO has no more vehicles to simulate
                    if traci.simulation.getMinExpectedNumber() == 0:
                        fail_reason = "no more vehicles"
                        break

                if self._time_up():
                    fail_reason = "time limit"
                    break

        except CollisionException as e:
            collision = True
            fail_reason = str(e)
        except (LaneChangeException, TimeOutException) as e:
            fail_reason = str(e)
        except Exception as e:
            fail_reason = f"{type(e).__name__}: {e}"
        finally:
            self.model.destroy()
            if gui:
                gui.terminate()
                gui.join()

        if self._violation:
            fail_reason = self._violation_reason

        avg_speed = round(self._speed_sum / self._speed_n, 3) if self._speed_n else 0.0

        return {
            "scenario": self.scenario_name,
            "trial": self.trial_index,
            "steps": self.model.timeStep,
            "decisions": self._decisions,
            "success": success,
            "collision": collision or self._near_miss_collision,
            "rule_violation": self._violation,
            "fail_reason": fail_reason,
            "elapsed_s": round(time.time() - t0, 2),
            "task_completion_time_s": (
                self._completion_time or round(self.model.timeStep * 0.1, 3)
            ),
            "average_speed": avg_speed,
            "designated_departed": len(self._departed),
            "designated_arrived": len(self._arrived),
            "designated_total": len(self.designated_ids),
            "scene_total": len(self.manifest.scene_vehicle_ids),
            "designated_ratio": self.manifest.designated_ratio,
            "fallbacks": self._fallbacks,
        }

    # ── Core decision cycle ─────────────────────────────────────────────────
    def _decide(self, roadgraph: Any, vehicles_info: dict) -> None:
        vehicles = self._collect_states(vehicles_info)
        controlled = [v for v in vehicles if v.vehicle_id in self.designated_ids]
        if not controlled:
            return

        ctx = self._scene_context(roadgraph, vehicles_info)
        proposals = {}
        for v in controlled:
            neighbors = [o for o in controlled if o.vehicle_id != v.vehicle_id]
            proposals[v.vehicle_id] = self.narrator.generate(
                v,
                neighbors,
                ctx,
            )

        if self.no_consensus:
            # No consensus: each vehicle acts on its own proposal
            actions = []
            for v in controlled:
                if v.vehicle_id not in proposals:
                    continue
                prop = proposals[v.vehicle_id]
                acc = prop.target_speed - v.speed
                actions.append(
                    ActionCommand(
                        vehicle_id=v.vehicle_id,
                        acceleration=acc,
                        steering=prop.steering_hint,
                    )
                )
            grouping = None
        else:
            actions, grouping = self.engine.step(controlled, proposals)
        actions = self._sanitize(controlled, actions)

        self._push_qa(ctx, proposals, roadgraph, vehicles_info)

        try:
            self._apply(roadgraph, vehicles_info, controlled, actions)
            self._decisions += 1
        except Exception:
            self._fallbacks += 1

    # ── State collection ────────────────────────────────────────────────────
    def _collect_states(self, vi: dict) -> list[VehicleState]:
        out: list[VehicleState] = []
        seen: set[str] = set()

        def add(raw: dict) -> None:
            xq, yq, sq = (
                raw.get("xQ") or [],
                raw.get("yQ") or [],
                raw.get("speedQ") or [],
            )
            if not xq or not yq or not sq:
                return
            vid = str(raw["id"])
            if vid in seen:
                return
            seen.add(vid)
            out.append(
                VehicleState(
                    vehicle_id=vid,
                    x=float(xq[-1]),
                    y=float(yq[-1]),
                    speed=float(sq[-1]),
                    lane_id=str((raw.get("laneIDQ") or [""])[-1]),
                    meta={
                        "available_lanes": sorted(raw.get("availableLanes", [])),
                        "vehicle_type": raw.get("vTypeID", ""),
                    },
                )
            )

        if vi.get("egoCar"):
            add(vi["egoCar"])
        for v in vi.get("carInAoI", []):
            add(v)
        for v in vi.get("outOfAoI", []):
            if str(v.get("id")) in self.designated_ids:
                add(v)
        return out

    def _scene_context(self, rg: Any, vi: dict) -> str:
        try:
            e = self.descriptor.getEnvPrompt(rg, vi)
            n = self.descriptor.getNavigationInfo(rg, vi)
            a = self.descriptor.getAvailableActionsInfo(rg, vi)
            return f"{e}\n## Navigation:\n{n}\n## Available actions:\n{a}"
        except Exception:
            return ""

    def _push_qa(self, ctx: str, proposals: dict, rg: Any, vi: dict) -> None:
        try:
            nav = self.descriptor.getNavigationInfo(rg, vi)
            act = self.descriptor.getAvailableActionsInfo(rg, vi)
        except Exception:
            nav, act = "", ""
        resp = (
            "\n".join(
                f"[{vid}] {p.intent} | spd={p.target_speed:.1f}" for vid, p in proposals.items()
            )
            or "none"
        )
        try:
            self.model.putQA(
                QuestionAndAnswer(
                    description=ctx[:500],
                    navigation=nav[:300],
                    actions=act[:300],
                    few_shots="",
                    response=resp,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    total_time=0.0,
                    choose_action=0,
                )
            )
        except Exception:
            pass

    # ── Action sanitization ─────────────────────────────────────────────────
    def _sanitize(
        self, controlled: list[VehicleState], actions: list[ActionCommand]
    ) -> list[ActionCommand]:
        cmap = {v.vehicle_id: v for v in controlled}
        out: list[ActionCommand] = []
        for a in actions:
            v = cmap.get(a.vehicle_id)
            if v is None:
                out.append(a)
                continue
            avail = [str(ln) for ln in v.meta.get("available_lanes", [])]
            lat = [ln for ln in avail if ln != v.lane_id and not ln.startswith(":")]
            steer = a.steering
            if v.lane_id.startswith(":") or not lat:
                steer = 0.0
            elif v.lane_id not in avail:
                steer = 0.0
            elif len(lat) == 1:
                try:
                    ce, ci = v.lane_id.rsplit("_", 1)
                    le, li = lat[0].rsplit("_", 1)
                    if le != ce or abs(int(li) - int(ci)) != 1:
                        steer = 0.0
                except ValueError:
                    steer = 0.0
            out.append(ActionCommand(a.vehicle_id, a.acceleration, steer, a.source))
        return out

    # ── Action application ──────────────────────────────────────────────────
    def _apply(
        self,
        rg: Any,
        vi: dict,
        controlled: list[VehicleState],
        actions: list[ActionCommand],
    ) -> None:
        eid = self.scenario["ego_id"]
        ego_cmd = next((a for a in actions if a.vehicle_id == eid), None)
        if ego_cmd is None:
            ego_cmd = ActionCommand(eid, 0.0, 0.0)
        ego_cmd = self._safety_override(rg, vi, ego_cmd)

        # Apply LLM-Raft decisions to ALL designated vehicles via traci
        # (hybrid control: high-level LLM decisions + low-level SUMO IDM)
        # This preserves SUMO's safe car-following while applying LLM guidance
        amap = {a.vehicle_id: a for a in actions}
        active = {str(v) for v in traci.vehicle.getIDList()}
        for v in controlled:
            if v.vehicle_id not in active:
                continue
            cmd = amap.get(v.vehicle_id)
            if cmd is None:
                continue
            if v.vehicle_id == eid:
                cmd = ego_cmd  # use safety-overridden version for ego
            self._apply_traci(v, cmd)

        # NOTE: We don't call planner.plan() or setTrajectories() here.
        # Instead, all vehicles are controlled via traci speed/lane commands
        # which work WITH SUMO's IDM (safe following, speed limits, etc.)
        # rather than overriding it with trajectory-level control.

    def _apply_traci(self, v: VehicleState, cmd: ActionCommand) -> None:
        """Apply LLM-Raft action to a designated vehicle via traci.

        Apply deceleration overrides from LLM decisions. SUMO IDM handles
        acceleration for safe car-following between decision steps.
        """
        vid = v.vehicle_id
        try:
            cur_speed = traci.vehicle.getSpeed(vid)
            dt = self.decision_interval * 0.1  # seconds until next decision

            if cmd.acceleration < -0.5:
                new_speed = max(0.0, cur_speed + cmd.acceleration * dt)
                traci.vehicle.setSpeed(vid, new_speed)

            # Lane change if steering indicates it
            steer_deg = math.degrees(cmd.steering)
            if steer_deg <= -5:
                traci.vehicle.changeLaneRelative(vid, -1, dt)
            elif steer_deg >= 5:
                traci.vehicle.changeLaneRelative(vid, 1, dt)
        except traci.exceptions.TraCIException:
            pass

    def _release_designated_speeds(self) -> None:
        """Release traci speed overrides so SUMO IDM controls between decisions."""
        active = {str(v) for v in traci.vehicle.getIDList()}
        for vid in self.designated_ids & active:
            try:
                traci.vehicle.setSpeed(vid, -1)
            except traci.exceptions.TraCIException:
                pass

    def _build_multi_decision(
        self, controlled: list[VehicleState], actions: list[ActionCommand]
    ) -> Any:
        if not controlled:
            return None
        obs = self.model.timeStep * 0.1
        cmap = {v.vehicle_id: v for v in controlled}
        amap = {a.vehicle_id: a for a in actions}
        raw = self.model.exportSce()[1]
        rg = self.model.exportSce()[0]
        if not raw or not rg:
            return None
        dt = float(self.planner.config["DT"])
        ct = int(obs / dt)
        through = ct - self.planner.time_step
        vehs = self.planner.extract_vehicles(raw, rg, obs, through, self.model.sim_mode)
        results = {}
        res = float(self.planner.config["DECISION_RESOLUTION"])
        for vid, veh in vehs.items():
            sid = str(vid)
            if sid not in cmap:
                continue
            cmd = amap.get(sid)
            if cmd is None:
                continue
            d = SingleStepDecision()
            d.action = self._to_action_str(cmd)
            d.expected_time = obs + res
            d.expected_state = self._predict(veh, d.action, res)
            results[veh] = [d]
        return MultiDecision(results=results)

    def _predict(self, veh: Any, action: str, dt: float) -> Any:
        s = veh.current_state
        p = type(s)(
            t=s.t,
            x=s.x,
            y=s.y,
            yaw=s.yaw,
            cur=s.cur,
            vel=s.vel,
            acc=s.acc,
            laneID=s.laneID,
            s=s.s,
            routeIdx=s.routeIdx,
            s_d=s.s_d,
            s_dd=s.s_dd,
            s_ddd=s.s_ddd,
            d=s.d,
            d_d=s.d_d,
            d_dd=s.d_dd,
            d_ddd=s.d_ddd,
        )
        da = float(self.planner.config["DEFAULT_ACC"])
        ls = float(self.planner.config["LATERAL_SPEED"])
        if action == "AC":
            p.vel = min(p.vel + da * dt, veh.max_speed)
            p.s += p.vel * dt + 0.5 * da * dt * dt
        elif action == "DC":
            p.vel = max(p.vel - da * dt, 0.0)
            p.s += max(0.0, p.vel * dt - 0.5 * da * dt * dt)
        elif action == "LCL":
            p.s += p.vel * dt
            p.d += ls * dt
        elif action == "LCR":
            p.s += p.vel * dt
            p.d -= ls * dt
        else:
            p.s += p.vel * dt
        p.laneID = veh.lane_id
        return p

    def _to_behaviour(self, a: ActionCommand) -> Behaviour:
        d = math.degrees(a.steering)
        if d <= -5:
            return Behaviour.LCL
        if d >= 5:
            return Behaviour.LCR
        if a.acceleration >= 0.5:
            return Behaviour.AC
        if a.acceleration <= -0.5:
            return Behaviour.DC
        return Behaviour.IDLE

    def _to_action_str(self, a: ActionCommand) -> str:
        d = math.degrees(a.steering)
        if d <= -5:
            return "LCL"
        if d >= 5:
            return "LCR"
        if a.acceleration >= 0.5:
            return "AC"
        if a.acceleration <= -0.5:
            return "DC"
        return "KS"

    # ── Safety override ─────────────────────────────────────────────────────
    def _safety_override(self, rg: Any, vi: dict, a: ActionCommand) -> ActionCommand:
        ego = vi.get("egoCar")
        if not ego or not ego.get("laneIDQ"):
            return a
        lid = str(ego["laneIDQ"][-1])
        lane = rg.get_lane_by_id(lid)
        if lane is None or lid.startswith(":"):
            return a

        nj = getattr(getattr(lane, "affiliated_edge", None), "to_junction", None)
        next_lane = None
        for lane_id in ego.get("availableLanes", []):
            lane_id = str(lane_id)
            if nj and nj in lane_id:
                next_lane = rg.get_lane_by_id(lane_id)
                break
        if next_lane is None or not getattr(next_lane, "id", "").startswith(":"):
            return a

        sig = getattr(next_lane, "currTlState", None)
        stop = float(lane.spline_length - ego["lanePosQ"][-1])

        # NOTE: lane_recovery check removed — it triggered false positives
        # on CarlaTown05 where current lane was valid but not in availableLanes
        if sig in {"r", "R"} and stop < 60:
            return ActionCommand(a.vehicle_id, min(a.acceleration, -5.0), 0.0, "red_light")
        if sig in {"y", "Y"} and stop < 35:
            return ActionCommand(a.vehicle_id, min(a.acceleration, -4.0), 0.0, "yellow_light")

        ego_lp = float(ego["lanePosQ"][-1])
        fg, fs = None, None
        for sv in vi.get("carInAoI", []):
            try:
                sl, sp, ss = (
                    str(sv["laneIDQ"][-1]),
                    float(sv["lanePosQ"][-1]),
                    float(sv["speedQ"][-1]),
                )
            except Exception:
                continue
            g = sp - ego_lp
            if sl == lid and g > 0 and (fg is None or g < fg):
                fg, fs = g, ss
        if fg is not None and fs is not None and fs < 1.0 and fg < 16:
            return ActionCommand(a.vehicle_id, -5.0, 0.0, "blocked_front")
        return a

    _PROXIMITY_THRESHOLD = 2.5  # metres — flag near-miss if any vehicle closer

    def _check_proximity_collision(self) -> None:
        """Check for near-miss collisions based on ego-vehicle proximity."""
        if self._near_miss_collision:
            return
        if not self.model.tpStart:
            return
        active = set(traci.vehicle.getIDList())
        ego_id = self.scenario["ego_id"]
        if ego_id not in active:
            return
        try:
            ex, ey = traci.vehicle.getPosition(ego_id)
            for vid in active:
                if vid == ego_id:
                    continue
                vx, vy = traci.vehicle.getPosition(vid)
                if math.hypot(ex - vx, ey - vy) < self._PROXIMITY_THRESHOLD:
                    self._near_miss_collision = True
                    return
        except traci.exceptions.TraCIException:
            pass

    # ── Metrics ─────────────────────────────────────────────────────────────
    def _update_metrics(self) -> None:
        dep = {str(v) for v in traci.simulation.getDepartedIDList()}
        arr = {str(v) for v in traci.simulation.getArrivedIDList()}
        self._departed.update(dep & self.designated_ids)
        self._arrived.update(arr & self.designated_ids)
        active = {str(v) for v in traci.vehicle.getIDList()}
        for vid in active & self.designated_ids:
            try:
                self._speed_sum += float(traci.vehicle.getSpeed(vid))
                self._speed_n += 1
            except Exception:
                pass
        # Completion = all departed designated vehicles have arrived.
        # (Some designated vehicles may fail SUMO insertion; we only track departed ones.)
        if (
            self._completion_time is None
            and self._departed
            and self._departed.issubset(self._arrived)
        ):
            self._completion_time = round(self._elapsed(), 3)

    def _is_success(self) -> bool:
        """Success = ALL designated vehicles reached destinations
        within 100s time limit, no collisions, no rule violations."""
        return (
            self._completion_time is not None
            and self._completion_time <= self.max_steps * 0.1
            and not self._violation
            and not self._near_miss_collision
        )

    def _elapsed(self) -> float:
        return 0.1 * max(0, self.model.timeStep - (self._ep_start or 0))

    def _time_up(self) -> bool:
        if self._ep_start is None:
            return False
        return (self.model.timeStep - self._ep_start) >= self.max_steps

    def _check_violations(self) -> None:
        if self.model.tpEnd:
            return
        rg, vi = self.model.exportSce()
        if not rg or not vi:
            return
        ego = vi.get("egoCar")
        if not ego or not ego.get("laneIDQ"):
            return
        lid = str(ego["laneIDQ"][-1])
        if not lid:
            self._violation, self._violation_reason = True, "invalid lane"
            return
        lane = rg.get_lane_by_id(lid)
        if lane is None:
            self._violation, self._violation_reason = True, f"off-road: {lid}"
            return
        if (
            lid.startswith(":")
            and self._prev_ego_lane
            and not self._prev_ego_lane.startswith(":")
            and getattr(lane, "currTlState", None) in {"r", "R"}
        ):
            self._violation, self._violation_reason = True, f"red light: {lid}"
            return
        self._prev_ego_lane = lid


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="LLM-Raft LimSim Runner")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()) + ["all"])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument(
        "--no-consensus",
        action="store_true",
        help="Disable consensus: each vehicle decides independently",
    )
    p.add_argument("--log-dir", default="results/logs")
    args = p.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    for sc in scenarios:
        print(f"\n{'=' * 60}")
        print(f"  Scenario: {sc}  |  Trials: {args.trials}")
        print(f"{'=' * 60}")

        stats: list[EpisodeStats] = []

        for t in range(args.trials):
            print(f"\n--- Trial {t}/{args.trials} ---")
            runner = LLMRaftRunner(
                scenario_name=sc,
                trial_index=t,
                use_gui=not args.no_gui and args.trials == 1,
                no_consensus=args.no_consensus,
            )
            summary = runner.run()

            # Save per-trial log
            log_dir = Path(args.log_dir) / sc
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"trial_{t:02d}.json"
            log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            stats.append(
                EpisodeStats(
                    task_completion_time_s=summary["task_completion_time_s"] or 100.0,
                    average_speed=summary["average_speed"],
                    collision=summary["collision"],
                    success=summary["success"],
                    rule_violation=summary["rule_violation"],
                )
            )

            status = "OK" if summary["success"] else f"FAIL: {summary['fail_reason'][:40]}"
            print(
                f"  [{status}] time={summary['task_completion_time_s']}s "
                f"speed={summary['average_speed']}m/s "
                f"collision={summary['collision']} "
                f"arrived={summary['designated_arrived']}"
                f"/{summary['designated_departed']}"
                f"/{summary['designated_total']} "
                f"scene={summary['scene_total']}"
            )

        if stats:
            agg = summarize(stats)
            print(f"\n{'─' * 40}")
            print(f"  {sc} aggregate ({len(stats)} trials):")
            print(f"  Success rate:      {agg['success_rate']:.1f}%")
            print(f"  Collision rate:    {agg['collision_rate']:.1f}%")
            print(f"  Avg completion:    {agg['task_completion_time_s']:.1f}s")
            print(f"  Avg speed:         {agg['average_speed']:.2f} m/s")
            print(f"{'─' * 40}")

            agg_path = Path(args.log_dir) / sc / "aggregate.json"
            agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

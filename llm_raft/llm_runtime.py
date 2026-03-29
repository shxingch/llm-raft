from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import http.client
from dataclasses import dataclass

from .data_types import GroupPlan, NarrativeProposal, VehicleState
from .helpers import clamp


@dataclass
class LLMRuntimeConfig:
    model: str = "gpt-4.1"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout_s: float = 30.0
    temperature: float = 0.1
    max_output_tokens: int = 1024
    scene_prompt: str = ""  # Pre-built scene prompt injected into system message

    @classmethod
    def from_env(cls, scene_prompt: str = "") -> LLMRuntimeConfig:
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")),
            scene_prompt=scene_prompt,
        )


def build_scene_prompt(cfg: dict) -> str:
    """Build the scene portion of the system prompt from config values.

    Numbers (speed limit, following distance, etc.) come from the YAML config
    so different scenarios get different driving parameters automatically.
    """
    speed_kmh = cfg.get("speed_limit_kmh", 50)
    speed_ms = speed_kmh / 3.6
    following_m = cfg.get("following_distance_m", 10)
    noise = cfg.get("action_noise", 0.02)
    rules = cfg.get("scene_rules", "")
    yellow_m = cfg.get("yellow_light_stop_m", 20)
    red_m = cfg.get("red_light_stop_m", 35)

    parts = [
        f"Speed limit: {speed_kmh} km/h ({speed_ms:.1f} m/s).",
        f"Minimum following distance: {following_m}m.",
        f"Begin braking for red lights within {red_m}m of stop line.",
        f"Begin slowing for yellow lights within {yellow_m}m.",
        f"Control noise: ±{noise:.2f} (small random perturbation on execution).",
    ]
    if rules:
        parts.append(f"\n{rules.strip()}")
    return "\n".join(parts)


# ── JSON helpers ──────────────────────────────────────────────────────────


def _extract_json_block(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _relative_summary(origin: VehicleState, other: VehicleState) -> str:
    dx = other.x - origin.x
    dy = other.y - origin.y
    return (
        f"id={other.vehicle_id}, dx={dx:.1f}, dy={dy:.1f}, "
        f"speed={other.speed:.1f}, lane={other.lane_id or 'unknown'}"
    )


def _coerce_range(
    data: object, default: tuple[float, float], lo: float, hi: float
) -> tuple[float, float]:
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        return default
    try:
        a = clamp(float(data[0]), lo, hi)
        b = clamp(float(data[1]), lo, hi)
    except (TypeError, ValueError):
        return default
    return (min(a, b), max(a, b))


def _compact_context(text: str, max_lines: int = 10, max_chars: int = 1200) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])[:max_chars]


# ── OpenAI-compatible HTTP client ─────────────────────────────────────────


class OpenAICompatibleChatClient:
    def __init__(self, cfg: LLMRuntimeConfig) -> None:
        self.cfg = cfg

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict | None:
        if not self.cfg.api_key:
            return None

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
            ):
                if attempt == 2:
                    return None
                time.sleep(0.6 * (attempt + 1))
        else:
            return None

        choices = data.get("choices") or []
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "")
        return _extract_json_block(content)


# ── Narrative Proposal Generator ──────────────────────────────────────────

_PROPOSAL_SYSTEM = (
    "You are an autonomous driving strategist. "
    "Return JSON with keys: intent, justification, "
    "acceleration_range, steering_range, target_speed, steering_hint.\n"
    "Rules:\n"
    "- Safety first, then efficient progress.\n"
    "- target_speed in m/s, steering_hint in [-0.35, 0.35].\n"
    "- acceleration_range as [min, max] within [-5, 10].\n"
    "- steering_range as [min, max] within [-0.35, 0.35].\n"
    "- Keep justification under 10 words.\n"
)

_CONSENSUS_SYSTEM = (
    "You coordinate a small group of autonomous vehicles. "
    "Return JSON with keys: plan_text, target_speed, "
    "steering_hint, acceleration_range, steering_range.\n"
    "Rules:\n"
    "- Primary: avoid collisions. Secondary: maximize progress.\n"
    "- If no imminent conflict, prefer fastest safe plan.\n"
    "- target_speed in m/s, ranges same as proposals.\n"
    "- Keep plan_text under 12 words.\n"
)


class NarrativeGenerator:
    """Generates per-vehicle narrative proposals via LLM API."""

    def __init__(self, cfg: LLMRuntimeConfig) -> None:
        self.cfg = cfg
        self.client = OpenAICompatibleChatClient(cfg)
        # System prompt = base role + scene rules (cached across calls)
        self._sys = _PROPOSAL_SYSTEM
        if cfg.scene_prompt:
            self._sys += f"\nScene context:\n{cfg.scene_prompt}\n"

    def generate(
        self,
        vehicle: VehicleState,
        neighbors: list[VehicleState],
        env_context: str = "",
    ) -> NarrativeProposal:
        proposal = self._call_api(vehicle, neighbors, env_context)
        if proposal is not None:
            return proposal
        # Minimal safe fallback when API is unreachable
        return NarrativeProposal(
            vehicle_id=vehicle.vehicle_id,
            intent="maintain current trajectory",
            justification="safe default",
            target_speed=max(vehicle.speed, 3.0),
            steering_hint=0.0,
            acceleration_range=(0.0, 2.0),
            steering_range=(0.0, 0.0),
        )

    def _call_api(
        self,
        vehicle: VehicleState,
        neighbors: list[VehicleState],
        env_context: str,
    ) -> NarrativeProposal | None:
        neighbor_text = "\n".join(_relative_summary(vehicle, n) for n in neighbors[:6]) or "none"
        dynamic = _compact_context(env_context)
        user_prompt = (
            f"self: id={vehicle.vehicle_id}, x={vehicle.x:.1f}, "
            f"y={vehicle.y:.1f}, speed={vehicle.speed:.1f}, "
            f"lane={vehicle.lane_id or 'unknown'}\n"
            f"neighbors:\n{neighbor_text}\n"
        )
        if dynamic:
            user_prompt += f"live scene:\n{dynamic}\n"
        user_prompt += "Decide the next-step action."

        data = self.client.complete_json(self._sys, user_prompt)
        if not data:
            return None
        try:
            return NarrativeProposal(
                vehicle_id=vehicle.vehicle_id,
                intent=str(data["intent"]),
                justification=str(data["justification"]),
                target_speed=max(0.0, float(data["target_speed"])),
                steering_hint=clamp(float(data["steering_hint"]), -0.35, 0.35),
                acceleration_range=_coerce_range(
                    data.get("acceleration_range"), (0.0, 3.0), -5.0, 10.0
                ),
                steering_range=_coerce_range(data.get("steering_range"), (0.0, 0.0), -0.35, 0.35),
            )
        except (KeyError, TypeError, ValueError):
            return None


# ── Group Plan Reconciler (Consensus) ─────────────────────────────────────


class GroupPlanReconciler:
    """Reconciles per-vehicle proposals into a group plan via LLM API."""

    def __init__(self, cfg: LLMRuntimeConfig) -> None:
        self.cfg = cfg
        self.client = OpenAICompatibleChatClient(cfg)
        self._sys = _CONSENSUS_SYSTEM
        if cfg.scene_prompt:
            self._sys += f"\nScene context:\n{cfg.scene_prompt}\n"

    def reconcile(
        self,
        group_id: int,
        members: list[VehicleState],
        proposals: list[NarrativeProposal],
        scenario_category: str,
    ) -> GroupPlan | None:
        plan = self._call_api(group_id, members, proposals)
        if plan is not None:
            return plan
        # Minimal fallback: average proposals
        return self._average_fallback(group_id, proposals)

    def _call_api(
        self,
        group_id: int,
        members: list[VehicleState],
        proposals: list[NarrativeProposal],
    ) -> GroupPlan | None:
        proposal_text = "\n".join(
            f"id={p.vehicle_id}, intent={p.intent}, "
            f"target_speed={p.target_speed:.1f}, "
            f"accel_range={p.acceleration_range}"
            for p in proposals[:6]
        )
        member_text = "\n".join(
            f"id={m.vehicle_id}, x={m.x:.1f}, y={m.y:.1f}, "
            f"speed={m.speed:.1f}, lane={m.lane_id or 'unknown'}"
            for m in members[:6]
        )
        user_prompt = (
            f"group={group_id}\n"
            f"members:\n{member_text}\n"
            f"proposals:\n{proposal_text}\n"
            "Choose one cooperative plan for this group."
        )
        data = self.client.complete_json(self._sys, user_prompt)
        if not data:
            return None
        try:
            return GroupPlan(
                group_id=group_id,
                plan_text=str(data["plan_text"]),
                target_speed=max(0.0, float(data["target_speed"])),
                steering_hint=clamp(float(data["steering_hint"]), -0.35, 0.35),
                proposer_ids=[p.vehicle_id for p in proposals],
                acceleration_range=_coerce_range(
                    data.get("acceleration_range"), (0.0, 3.0), -5.0, 10.0
                ),
                steering_range=_coerce_range(data.get("steering_range"), (0.0, 0.0), -0.35, 0.35),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _average_fallback(group_id: int, proposals: list[NarrativeProposal]) -> GroupPlan:
        if not proposals:
            return GroupPlan(
                group_id=group_id,
                plan_text="No proposals, hold position.",
                target_speed=0.0,
                steering_hint=0.0,
                proposer_ids=[],
            )
        n = len(proposals)
        return GroupPlan(
            group_id=group_id,
            plan_text="; ".join(p.intent for p in proposals),
            target_speed=sum(p.target_speed for p in proposals) / n,
            steering_hint=sum(p.steering_hint for p in proposals) / n,
            proposer_ids=[p.vehicle_id for p in proposals],
            acceleration_range=(
                min((p.acceleration_range or (0, 0))[0] for p in proposals),
                max((p.acceleration_range or (0, 0))[1] for p in proposals),
            ),
            steering_range=(
                min((p.steering_range or (0, 0))[0] for p in proposals),
                max((p.steering_range or (0, 0))[1] for p in proposals),
            ),
        )

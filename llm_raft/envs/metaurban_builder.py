from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetaUrbanScenarioConfig:
    scenario_name: str
    env_type: str
    num_scenarios: int
    start_seed: int
    map_id: str
    traffic_density: float
    object_density: float
    crswalk_density: float
    spawn_human_num: int
    spawn_wheelchairman_num: int = 0
    spawn_edog_num: int = 0
    spawn_erobot_num: int = 0
    spawn_drobot_num: int = 0
    max_actor_num: int = 20
    use_render: bool = False
    horizon: int = 1000
    window_size: tuple[int, int] = (1400, 900)
    min_success_steps: int = 20


def build_vehicle_metaurban_env(env_type: str):
    from metaurban import SidewalkDynamicMetaUrbanEnv, SidewalkStaticMetaUrbanEnv

    base_cls = SidewalkDynamicMetaUrbanEnv if env_type == "dynamic" else SidewalkStaticMetaUrbanEnv

    class VehicleMetaUrbanEnv(base_cls):
        @staticmethod
        def _is_arrive_destination(vehicle):
            min_success_steps = int(vehicle.engine.global_config.get("min_success_steps", 0))
            if vehicle.engine.episode_step < min_success_steps:
                return False
            if (
                hasattr(vehicle.navigation, "final_lane")
                and vehicle.navigation.final_lane is not None
            ):
                long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
                return (
                    vehicle.navigation.final_lane.length - 5
                    < long
                    < vehicle.navigation.final_lane.length + 5
                ) and (
                    vehicle.navigation.get_current_lane_width() / 2
                    >= lat
                    >= (0.5 - vehicle.navigation.get_current_lane_num())
                    * vehicle.navigation.get_current_lane_width()
                )
            route_completion = getattr(vehicle.navigation, "route_completion", 0.0)
            return route_completion > 0.95

        def _is_out_of_road(self, vehicle):
            ret = not vehicle.on_lane
            if self.config["out_of_route_done"]:
                ret = ret or vehicle.out_of_route
            elif self.config["on_continuous_line_done"]:
                ret = ret or (
                    vehicle.on_yellow_continuous_line
                    or vehicle.on_white_continuous_line
                    or vehicle.crash_sidewalk
                )
            if self.config["on_broken_line_done"]:
                ret = ret or vehicle.on_broken_line
            if self.config.get("relax_out_of_road_done", False) and vehicle.navigation is not None:
                current_lateral = abs(getattr(vehicle.navigation, "current_lateral", 0.0))
                ret = ret or current_lateral > self.config["max_lateral_dist"]
            return ret

        def reward_function(self, vehicle_id: str):
            vehicle = self.agents[vehicle_id]
            step_info: dict[str, Any] = {}

            if vehicle.lane in vehicle.navigation.current_ref_lanes:
                current_lane = vehicle.lane
                positive_road = 1
            else:
                current_lane = vehicle.navigation.current_ref_lanes[0]
                current_road = vehicle.navigation.current_road
                positive_road = 1 if not current_road.is_negative_road() else -1
            long_last, _ = current_lane.local_coordinates(vehicle.last_position)
            long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

            from metaurban.utils import clip

            if self.config["use_lateral_reward"]:
                lateral_factor = clip(
                    1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(),
                    0.0,
                    1.0,
                )
            else:
                lateral_factor = 1.0

            reward = 0.0
            reward += (
                self.config["driving_reward"]
                * (long_now - long_last)
                * lateral_factor
                * positive_road
            )
            reward += (
                self.config["speed_reward"]
                * (vehicle.speed_km_h / vehicle.max_speed_km_h)
                * positive_road
            )
            step_info["step_reward"] = reward

            if self._is_arrive_destination(vehicle):
                reward = +self.config["success_reward"]
            elif self._is_out_of_road(vehicle):
                reward = -self.config["out_of_road_penalty"]
            elif vehicle.crash_vehicle:
                reward = -self.config["crash_vehicle_penalty"]
            elif vehicle.crash_object:
                reward = -self.config["crash_object_penalty"]
            elif vehicle.crash_sidewalk:
                reward = -self.config.get(
                    "crash_sidewalk_penalty", self.config["crash_object_penalty"]
                )
            step_info["route_completion"] = getattr(vehicle.navigation, "route_completion", 0.0)
            return reward, step_info

        def _get_agent_manager(self):
            from metaurban.manager.agent_manager import VehicleAgentManager

            return VehicleAgentManager(init_observations=self._get_observations())

    return VehicleMetaUrbanEnv


def build_env_config(cfg: MetaUrbanScenarioConfig) -> dict[str, Any]:
    return dict(
        use_render=cfg.use_render,
        map=cfg.map_id,
        num_scenarios=cfg.num_scenarios,
        start_seed=cfg.start_seed,
        random_spawn_lane_index=False,
        traffic_density=cfg.traffic_density,
        object_density=cfg.object_density,
        crswalk_density=cfg.crswalk_density,
        spawn_human_num=cfg.spawn_human_num,
        spawn_wheelchairman_num=cfg.spawn_wheelchairman_num,
        spawn_edog_num=cfg.spawn_edog_num,
        spawn_erobot_num=cfg.spawn_erobot_num,
        spawn_drobot_num=cfg.spawn_drobot_num,
        max_actor_num=cfg.max_actor_num,
        walk_on_all_regions=False,
        show_sidewalk=True,
        show_crosswalk=True,
        show_ego_navigation=False,
        show_mid_block_map=False,
        use_lateral_reward=True,
        manual_control=False,
        horizon=cfg.horizon,
        drivable_area_extension=55,
        height_scale=1,
        accident_prob=0.0,
        relax_out_of_road_done=True,
        out_of_route_done=True,
        on_continuous_line_done=False,
        on_broken_line_done=False,
        max_lateral_dist=5.0,
        speed_reward=0.1,
        debug=False,
        min_success_steps=cfg.min_success_steps,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
        ),
        traffic_vehicle_config=dict(
            show_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=False,
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,
        ),
        window_size=cfg.window_size,
    )

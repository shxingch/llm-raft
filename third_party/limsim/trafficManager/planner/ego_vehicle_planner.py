import time,copy

from trafficManager.common.observation import Observation
from trafficManager.common.vehicle import Behaviour, Vehicle
from trafficManager.planner.abstract_planner import AbstractEgoPlanner
from trafficManager.predictor.abstract_predictor import Prediction

import logger, logging
from utils.obstacles import DynamicObstacle, ObsType, Rectangle
from utils.roadgraph import JunctionLane, NormalLane, RoadGraph, AbstractLane
from utils.trajectory import State, Trajectory
from simInfo.CustomExceptions import NoPathFoundError

import numpy as np
import math

from trafficManager.planner.frenet_optimal_planner import frenet_optimal_planner
from trafficManager.common import cost
from trafficManager.common.vehicle import Action as action

decision_logger = logger.setup_app_level_logger(logger_name="ego_planner", file_name="llm_decision.log")
logging = logging.getLogger("ego_planner").getChild(__name__)

vehicle_debuglog = 'vehicle_debug.txt'
def log_vehicle(state, path: str = vehicle_debuglog):
    with open(path, "a",encoding='utf-8') as file:  # 使用'a'模式，以便在文件末尾追加内容
        file.write(f"t: {state.t}, s: {state.s}, d: {state.d}, yaw: {state.yaw}, vel: {state.vel}, acc: {state.acc}\n")
        file.write("="*50 + "\n")

class LLMEgoPlanner(AbstractEgoPlanner):
    def plan(self,
             ego_veh: Vehicle,
             observation: Observation,
             roadgraph: RoadGraph,
             prediction: Prediction,
             T,
             ego_action:action,
             config) -> Trajectory:
        logging.info("\n============= planning at {} s =============\n".format(T))
        vehicle_id = ego_veh.id
        start = time.time()
        current_lane = roadgraph.get_lane_by_id(ego_veh.lane_id)
        obs_list = []
        # Process static obstacle
        for obs in observation.obstacles:
            obs_list.append(obs)

        # Process dynamic_obstacles
        for predict_veh, prediction in prediction.results.items():
            if predict_veh.id == vehicle_id:
                continue

            shape = Rectangle(predict_veh.length, predict_veh.width)
            current_state = State(x=prediction[0].x,
                                  y=prediction[0].y,
                                  s=prediction[0].s,
                                  d=prediction[0].d,
                                  yaw=prediction[0].yaw,
                                  vel=prediction[0].vel)
            dynamic_obs = DynamicObstacle(obstacle_id=predict_veh.id,
                                          shape=shape,
                                          obstacle_type=ObsType.CAR,
                                          current_state=current_state,
                                          lane_id=predict_veh.lane_id)
            for i in range(1, len(prediction)):
                state = State(x=prediction[i].x,
                              y=prediction[i].y,
                              s=prediction[i].s,
                              d=prediction[i].d,
                              yaw=prediction[i].yaw,
                              vel=prediction[i].vel)
                dynamic_obs.future_trajectory.states.append(state)

            obs_list.append(dynamic_obs)

        """
        Predict for current vehicle
        """
        next_lane = roadgraph.get_available_next_lane(
            current_lane.id, ego_veh.available_lanes)
        lanes = [current_lane, next_lane] if next_lane != None else [
            current_lane]

        
        
        
        if ego_veh.behaviour in [Behaviour.AC, Behaviour.DC, Behaviour.IDLE]:
            # 将 ego_action 传递给 acdc_trajectory_generator 方法
            path = self.acdc_trajectory_generator(ego_veh, lanes, roadgraph, config, ego_veh.behaviour, ego_action)
        # if ego_veh.behaviour == Behaviour.IDLE:
        #     path = self.acdc_trajectory_generator(ego_veh, lanes,roadgraph,config, ego_veh.behaviour)
        # elif ego_veh.behaviour == Behaviour.AC:
        #     # Accelerate
        #     path = self.acdc_trajectory_generator(ego_veh, lanes,roadgraph,config, ego_veh.behaviour)
        # elif ego_veh.behaviour == Behaviour.DC:
        #     # Decelerate
        #     path = self.acdc_trajectory_generator(ego_veh, lanes,roadgraph,config, ego_veh.behaviour)
        elif ego_veh.behaviour == Behaviour.LCL:
            # Turn Left
            left_lane = roadgraph.get_lane_by_id(current_lane.left_lane())
            path = self.lanechange_trajectory_generator(
                ego_veh,
                left_lane,
                obs_list,
                ego_action,
                config,
                T,
            )
        elif ego_veh.behaviour == Behaviour.LCR:
            # Turn Right
            right_lane = roadgraph.get_lane_by_id(current_lane.right_lane())
            path = self.lanechange_trajectory_generator(
                ego_veh,
                right_lane,
                obs_list,
                ego_action,
                config,
                T,
            )
        else:
            logging.error(
                "Vehicle {} has unknown behaviour {}".format(
                    ego_veh.id, ego_veh.behaviour)
            )
        logging.debug(
            "Vehicle {} Total planning time: {}".format(
                ego_veh.id, time.time() - start)
        )

        return path

    def acdc_trajectory_generator(self, 
        vehicle: Vehicle, 
        lanes: AbstractLane,
        roadgraph: RoadGraph, 
        config, 
        behaviour:Behaviour,
        ego_action: action
    ) -> Trajectory:
        course_t = config["MIN_T"]  # 未来规划空间
        dt = config["DT"]  # time tick
        current_state = vehicle.current_state
        
        # if behaviour == Behaviour.AC:
        #     target_acc = ego_action.acc
        #     if target_acc > config["ACC_MAX"]:
        #         target_acc = config["ACC_MAX"]

        # elif behaviour == Behaviour.DC:
        #     target_acc = ego_action.acc
        #     if target_acc < config["ACC_MIN"]:
        #         target_acc = config["ACC_MIN"]
        #     # When the speed decreases to 0, it does not continue to decelerate and stops at the same place
        #     # 此处用来免去车辆在速度降为0后继续减速的情况，但是可以思考一下已经有IDM的情况下是否仍然需要？
        #     if current_state.vel + target_acc*course_t < 0:
        #         course_t = (0.1 - current_state.vel)/target_acc
        if behaviour == Behaviour.AC or behaviour == Behaviour.DC:
            target_acc = ego_action.acc
            if target_acc > config["ACC_MAX"]:
                target_acc = config["ACC_MAX"]
            elif target_acc < config["ACC_MIN"]:
                target_acc = config["ACC_MIN"]
        elif behaviour == Behaviour.IDLE:
            target_acc = 0

        target_v = current_state.vel + target_acc*course_t
        target_s = current_state.s + (current_state.vel + target_v)* course_t * 0.5
        if target_s - current_state.s < vehicle.length:
            target_state = State(s=target_s, s_d=target_v, d=current_state.d)
        else:
            target_state = State(s=target_s, s_d=target_v, d=0)

        # judge stopping the car
        if (round(current_state.vel, 1) <= 0.1 and
            (target_s - current_state.s) <= vehicle.length):  # already stopped, keep it
            logging.debug(f"Vehicle {vehicle.id} Already stopped")
            path = Trajectory()
            for t in np.arange(0, config["MIN_T"], dt):
                path.states.append(
                    State(t=t, s=current_state.s, d=current_state.d, yaw=None))
            path.frenet_to_cartesian(lanes, vehicle.current_state)
            path.cost = (
                cost.smoothness(path, lanes[0].course_spline, config["weights"]) *
                dt + cost.guidance(path, config["weights"]) * dt +
                cost.acc(path, config["weights"]) * dt +
                cost.jerk(path, config["weights"]) * dt)
        else:
            # 计算first point state
            next_state = copy.deepcopy(vehicle)
            log_vehicle(vehicle.current_state)
            next_state.current_state.yaw = current_state.yaw + ego_action.steering # 注意steering有正负
            next_state.current_state.x = current_state.x + (current_state.vel + ego_action.acc * dt) * math.cos(current_state.yaw) * dt
            next_state.current_state.y = current_state.y + (current_state.vel + ego_action.acc * dt) * math.sin(current_state.yaw) * dt
            next_state.current_state.vel = current_state.vel + ego_action.acc * dt
            next_state.current_state.acc = ego_action.acc
            new_state = next_state.get_state_in_lane(roadgraph.get_lane_by_id(next_state.lane_id))

            path = frenet_optimal_planner.calc_spec_path(new_state,
                                                        target_state, course_t, dt
                                                        )
            path.states.insert(0, current_state)
            
            # 计算后续点state
            if course_t != config["MIN_T"]:
                logging.debug(f"course_t is {course_t}, dt is {dt}, path length is {len(path.states)}, yaw is {path.states[-1].yaw}, current yaw is {current_state.yaw}")
                for t in np.arange(path.states[-1].t + dt, config["MIN_T"], dt):
                    path.states.append(
                        State(t=t, s=target_state.s, d=target_state.d, yaw=None))
            path.frenet_to_cartesian(lanes, current_state)
            path.cost = (
                cost.smoothness(path, lanes[0].course_spline, config["weights"]) *
                dt + cost.guidance(path, config["weights"]) * dt +
                cost.acc(path, config["weights"]) * dt +
                cost.jerk(path, config["weights"]) * dt)
        return path

    def lanechange_trajectory_generator(self,
        vehicle: Vehicle,
        target_lane: AbstractLane,
        obs_list,
        ego_action:action,
        config,
        T,
    ) -> Trajectory:
        target_vel = vehicle.current_state.vel
        dt = config["DT"]
        s_sample = config["S_SAMPLE"]
        n_s_sample = config["N_S_SAMPLE"]
        sample_t = [config["MIN_T"] / 1.5]  # Sample course time
        
        next_state = copy.deepcopy(vehicle)
        
        print("ego_action.acc", ego_action.acc)
        print("ego_action.steering", ego_action.steering)
        if ego_action.steering is None:
            ego_action.steering = 0.0
        
        log_vehicle(vehicle.current_state)
        
        next_state.current_state.yaw = next_state.current_state.yaw + ego_action.steering # 注意steering有正负
        next_state.current_state.x = next_state.current_state.x + (next_state.current_state.vel + ego_action.acc * dt) * math.cos(next_state.current_state.yaw) * dt
        next_state.current_state.y = next_state.current_state.y + (next_state.current_state.vel + ego_action.acc * dt) * math.sin(next_state.current_state.yaw) * dt
        next_state.current_state.vel = next_state.current_state.vel + ego_action.acc * dt
        next_state.current_state.acc = ego_action.acc
        state_in_target_lane = vehicle.get_state_in_lane(target_lane)
        next_state_in_target_lane = next_state.get_state_in_lane(target_lane)


        vel_min = max(state_in_target_lane.vel - 2.0, 0)
        vel_max = min(target_vel + s_sample * n_s_sample * 1.01, target_lane.speed_limit)
        vel_max = max(vel_max, 5.0)
        sample_s = np.empty(0)
        for t in sample_t:
            sample_s = np.append(
                sample_s,
                np.arange(
                    state_in_target_lane.s + t * vel_min,
                    state_in_target_lane.s + t * vel_max,
                    s_sample,
                ),
            )

        # Step 2: Calculate Paths
        best_path = None
        best_cost = math.inf
        for t in sample_t:
            for s in sample_s:
                s_d = vehicle.current_state.vel
                target_state = State(t=t, s=s, d=0, s_d=s_d)
                path = frenet_optimal_planner.calc_spec_path(
                    next_state_in_target_lane, target_state, target_state.t, dt)
                if not path.states:
                    continue
                path.states.insert(0, vehicle.current_state)
                path.frenet_to_cartesian(target_lane, vehicle.current_state)
                path.cost = (
                    cost.smoothness(path, target_lane.course_spline,
                                    config["weights"]) * dt +
                    cost.vel_diff(path, target_vel, config["weights"]) * dt +
                    cost.guidance(path, config["weights"]) * dt +
                    cost.acc(path, config["weights"]) * dt +
                    cost.jerk(path, config["weights"]) * dt +
                    cost.obs(vehicle, path, obs_list, config) +
                    cost.changelane(config["weights"]))
                if not path.is_nonholonomic():
                    continue
                if path.cost < best_cost:
                    best_cost = path.cost
                    best_path = path
        if best_path is not None:
            logging.debug(f"Vehicle {vehicle.id} found a lane change path with cost: {best_cost}")
            return best_path
        else:
            return None
            # raise NoPathFoundError("Can't generate the trajectory!")

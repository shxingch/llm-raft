"""
MetaUrban 动态环境中的LLM驾驶控制

该脚本演示了如何使用大语言模型(LLM)在MetaUrban的动态环境中进行自动驾驶。
LLM会接收周围车辆信息、环境信息和导航信息，然后输出相应的驾驶动作。

注意：需要设置OPENAI_API_KEY环境变量或在代码中配置API密钥。
"""

import argparse
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import openai
from metaurban import SidewalkDynamicMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.obs.mix_obs import ThreeSourceMixObservation


class LLMDriveController:
    """LLM驾驶控制器类"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        初始化LLM驾驶控制器
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.conversation_history = []
        
    def extract_vehicle_info(self, observation: np.ndarray, vehicle) -> Dict[str, Any]:
        """
        提取车辆状态信息
        
        Args:
            observation: 环境观察数据
            vehicle: 车辆对象
            
        Returns:
            包含车辆状态信息的字典
        """
        vehicle_info = {
            "position": {
                "x": float(vehicle.position[0]),
                "y": float(vehicle.position[1]),
                "z": float(vehicle.position[2]) if len(vehicle.position) > 2 else 0.0
            },
            "velocity": {
                "speed_kmh": float(vehicle.speed_km_h),
                "velocity_x": float(vehicle.velocity[0]),
                "velocity_y": float(vehicle.velocity[1])
            },
            "orientation": {
                "heading_degree": float(np.rad2deg(vehicle.heading_theta)),
                "steering": float(vehicle.steering)
            },
            "road_info": {
                "dist_to_left_side": float(vehicle.dist_to_left_side),
                "dist_to_right_side": float(vehicle.dist_to_right_side),
                "on_lane": bool(vehicle.on_lane)
            }
        }
        
        # 添加车道信息
        if vehicle.navigation and vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.navigation.current_ref_lanes[-1]
            lane_info = {
                "lane_width": float(current_lane.width),
                "heading_diff": float(vehicle.heading_diff(current_lane))
            }
            vehicle_info["lane_info"] = lane_info
            
        return vehicle_info
    
    def extract_surrounding_vehicles_info(self, observation: np.ndarray, vehicle) -> List[Dict[str, Any]]:
        """
        提取周围车辆信息
        
        Args:
            observation: 环境观察数据
            vehicle: 自车对象
            
        Returns:
            周围车辆信息列表
        """
        surrounding_vehicles = []
        
        # 从激光雷达传感器获取周围车辆
        if hasattr(vehicle.engine, "get_sensor"):
            lidar_sensor = vehicle.engine.get_sensor("lidar")
            if lidar_sensor:
                # 获取周围物体
                detected_objects = lidar_sensor.get_surrounding_objects(vehicle, radius=50)
                vehicles = lidar_sensor.get_surrounding_vehicles(detected_objects)
                
                for other_vehicle in list(vehicles)[:5]:  # 限制最多5辆车
                    # 计算相对位置
                    relative_pos = other_vehicle.position - vehicle.position
                    distance = float(np.linalg.norm(relative_pos[:2]))
                    
                    # 转换到车辆坐标系
                    local_coords = vehicle.convert_to_local_coordinates(
                        other_vehicle.position, vehicle.position
                    )
                    
                    vehicle_info = {
                        "distance": distance,
                        "relative_position": {
                            "front_back": float(local_coords[0]),  # 正值表示前方
                            "left_right": float(local_coords[1])   # 正值表示左侧
                        },
                        "velocity": {
                            "speed_kmh": float(other_vehicle.speed_km_h),
                            "relative_speed": float(other_vehicle.speed_km_h - vehicle.speed_km_h)
                        },
                        "heading_degree": float(np.rad2deg(other_vehicle.heading_theta))
                    }
                    surrounding_vehicles.append(vehicle_info)
        
        return surrounding_vehicles
    
    def extract_navigation_info(self, vehicle) -> Dict[str, Any]:
        """
        提取导航信息
        
        Args:
            vehicle: 车辆对象
            
        Returns:
            导航信息字典
        """
        navigation_info = {}
        
        if vehicle.navigation:
            # 获取导航信息
            navi_info = vehicle.navigation.get_navi_info()
            
            # 获取检查点
            try:
                checkpoints = vehicle.navigation.get_checkpoints()
                if checkpoints and len(checkpoints) >= 2:
                    checkpoint1, checkpoint2 = checkpoints[0], checkpoints[1]
                    
                    # 计算到检查点的距离和方向
                    dist_to_checkpoint1 = np.linalg.norm(
                        np.array(checkpoint1[:2]) - np.array(vehicle.position[:2])
                    )
                    
                    # 转换到车辆坐标系
                    relative_checkpoint1 = vehicle.convert_to_local_coordinates(
                        checkpoint1, vehicle.position
                    )
                    
                    navigation_info = {
                        "distance_to_next_checkpoint": float(dist_to_checkpoint1),
                        "checkpoint_direction": {
                            "front_back": float(relative_checkpoint1[0]),
                            "left_right": float(relative_checkpoint1[1])
                        },
                        "navigation_raw": navi_info.tolist() if hasattr(navi_info, 'tolist') else list(navi_info)
                    }
            except Exception as e:
                navigation_info["error"] = f"导航信息获取失败: {str(e)}"
        
        return navigation_info
    
    def extract_environment_info(self, observation: np.ndarray, vehicle, info: Dict) -> Dict[str, Any]:
        """
        提取环境信息
        
        Args:
            observation: 环境观察数据
            vehicle: 车辆对象
            info: step信息
            
        Returns:
            环境信息字典
        """
        environment_info = {
            "reward": float(info.get("reward", 0.0)),
            "episode_length": int(info.get("episode_length", 0)),
            "out_of_road": bool(info.get("out_of_road", False)),
            "crash": bool(info.get("crash", False)),
            "arrive_dest": bool(info.get("arrive_dest", False)),
            "max_step": bool(info.get("max_step", False))
        }
        
        # 添加道路边界信息（如果可用）
        if hasattr(vehicle, 'dist_to_left_side') and hasattr(vehicle, 'dist_to_right_side'):
            environment_info["road_boundaries"] = {
                "left_distance": float(vehicle.dist_to_left_side),
                "right_distance": float(vehicle.dist_to_right_side)
            }
        
        return environment_info
    
    def create_prompt(self, vehicle_info: Dict, surrounding_vehicles: List[Dict], 
                     navigation_info: Dict, environment_info: Dict) -> str:
        """
        创建发送给LLM的提示词
        
        Args:
            vehicle_info: 自车信息
            surrounding_vehicles: 周围车辆信息
            navigation_info: 导航信息
            environment_info: 环境信息
            
        Returns:
            格式化的提示词字符串
        """
        prompt = f"""
你是一个专业的自动驾驶AI助手。请根据以下信息，为车辆提供安全、高效的驾驶决策。

## 当前车辆状态
- 位置: ({vehicle_info['position']['x']:.2f}, {vehicle_info['position']['y']:.2f})
- 速度: {vehicle_info['velocity']['speed_kmh']:.1f} km/h
- 朝向: {vehicle_info['orientation']['heading_degree']:.1f}°
- 转向角: {vehicle_info['orientation']['steering']:.3f}
- 车道位置: 左侧距离 {vehicle_info['road_info']['dist_to_left_side']:.2f}m, 右侧距离 {vehicle_info['road_info']['dist_to_right_side']:.2f}m
- 是否在车道内: {'是' if vehicle_info['road_info']['on_lane'] else '否'}

## 周围车辆情况
"""
        
        if surrounding_vehicles:
            for i, sv in enumerate(surrounding_vehicles):
                prompt += f"""
车辆 {i+1}:
  - 距离: {sv['distance']:.2f}m
  - 相对位置: 前后 {sv['relative_position']['front_back']:.2f}m (正值=前方), 左右 {sv['relative_position']['left_right']:.2f}m (正值=左侧)
  - 速度: {sv['velocity']['speed_kmh']:.1f} km/h (相对速度: {sv['velocity']['relative_speed']:.1f} km/h)
  - 朝向: {sv['heading_degree']:.1f}°
"""
        else:
            prompt += "\n无周围车辆检测到。\n"
        
        prompt += f"""
## 导航信息
"""
        if navigation_info.get("distance_to_next_checkpoint"):
            prompt += f"""
- 到下一检查点距离: {navigation_info['distance_to_next_checkpoint']:.2f}m
- 检查点方向: 前后 {navigation_info['checkpoint_direction']['front_back']:.2f}m, 左右 {navigation_info['checkpoint_direction']['left_right']:.2f}m
"""
        else:
            prompt += "- 导航信息不可用\n"
        
        prompt += f"""
## 环境状态
- 当前奖励: {environment_info['reward']:.3f}
- 是否偏离道路: {'是' if environment_info['out_of_road'] else '否'}
- 是否发生碰撞: {'是' if environment_info['crash'] else '否'}
- 是否到达目的地: {'是' if environment_info['arrive_dest'] else '否'}

## 请根据以上信息提供驾驶决策

请返回一个JSON格式的响应，包含以下字段：
- "steering": 转向控制值 (范围: -1.0 到 1.0，负值向右转，正值向左转)
- "throttle": 油门/刹车控制值 (范围: -1.0 到 1.0，正值加速，负值刹车)
- "reasoning": 决策理由的简短说明

示例格式:
{{"steering": 0.1, "throttle": 0.5, "reasoning": "前方无障碍，轻微左转避开右侧车辆，适中加速"}}

重要安全规则:
1. 避免碰撞是第一优先级
2. 保持在车道内行驶
3. 与其他车辆保持安全距离
4. 根据交通情况调整速度
5. 平滑控制，避免急转急刹
"""
        
        return prompt
    
    def parse_llm_response(self, response_text: str) -> Tuple[float, float, str]:
        """
        解析LLM响应并提取控制动作
        
        Args:
            response_text: LLM的响应文本
            
        Returns:
            (steering, throttle, reasoning) 元组
        """
        try:
            # 尝试提取JSON部分
            response_text = response_text.strip()
            
            # 找到JSON开始和结束位置
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                data = json.loads(json_text)
                
                steering = float(data.get("steering", 0.0))
                throttle = float(data.get("throttle", 0.0))
                reasoning = str(data.get("reasoning", "无理由说明"))
                
                # 限制动作范围
                steering = np.clip(steering, -1.0, 1.0)
                throttle = np.clip(throttle, -1.0, 1.0)
                
                return steering, throttle, reasoning
            else:
                raise ValueError("未找到有效的JSON格式")
                
        except Exception as e:
            print(f"解析LLM响应时出错: {e}")
            print(f"原始响应: {response_text}")
            # 返回安全的默认动作
            return 0.0, 0.0, f"解析错误，使用默认动作: {str(e)}"
    
    def get_action(self, observation: np.ndarray, vehicle, info: Dict) -> Tuple[float, float]:
        """
        获取LLM的驾驶动作
        
        Args:
            observation: 环境观察数据
            vehicle: 车辆对象
            info: step信息
            
        Returns:
            (steering, throttle) 动作元组
        """
        try:
            # 提取各种信息
            vehicle_info = self.extract_vehicle_info(observation, vehicle)
            surrounding_vehicles = self.extract_surrounding_vehicles_info(observation, vehicle)
            navigation_info = self.extract_navigation_info(vehicle)
            environment_info = self.extract_environment_info(observation, vehicle, info)
            
            # 创建提示词
            prompt = self.create_prompt(vehicle_info, surrounding_vehicles, navigation_info, environment_info)
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的自动驾驶AI助手，需要根据环境信息提供安全的驾驶控制指令。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1  # 低温度以获得更稳定的响应
            )
            
            # 解析响应
            response_text = response.choices[0].message.content
            steering, throttle, reasoning = self.parse_llm_response(response_text)
            
            print(f"LLM决策 - 转向: {steering:.3f}, 油门: {throttle:.3f}")
            print(f"决策理由: {reasoning}")
            print("-" * 50)
            
            return steering, throttle
            
        except Exception as e:
            print(f"LLM调用失败: {e}")
            # 返回安全的默认动作
            return 0.0, 0.0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MetaUrban LLM驾驶控制演示")
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", 'all'])
    parser.add_argument("--density_obj", type=float, default=0.3, help="物体密度")
    parser.add_argument("--density_ped", type=float, default=1.0, help="行人密度") 
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API密钥")
    parser.add_argument("--model", type=str, default="gpt-4", help="使用的LLM模型")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大步数")
    args = parser.parse_args()

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 配置环境
    map_type = 'X'  # 使用交叉路口地图
    den_scale = args.density_ped
    
    config = dict(
        crswalk_density=1,
        object_density=args.density_obj,
        walk_on_all_regions=False,
        use_render=True,
        map=map_type,
        manual_control=False,  # 关闭手动控制
        drivable_area_extension=55,
        height_scale=1,
        show_mid_block_map=False,
        show_ego_navigation=True,  # 显示导航信息
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=True,  # 显示激光雷达
            show_navi_mark=True,
            show_line_to_navi_mark=True,
            show_dest_mark=True,
            enable_reverse=True,
            # 激光雷达配置
            lidar=dict(
                num_lasers=240,
                distance=50,
                num_others=4,  # 检测周围车辆数量
                gaussian_noise=0.0,
                dropout_prob=0.0,
                add_others_navi=True
            )
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # 场景设置
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        
        # 行人和机器人数量
        spawn_human_num=int(20 * den_scale),
        spawn_wheelchairman_num=int(1 * den_scale),
        spawn_edog_num=int(2 * den_scale),
        spawn_erobot_num=int(1 * den_scale),
        spawn_drobot_num=int(1 * den_scale),
        max_actor_num=20,
        
        window_size=(1200, 900),
        
        # 观察配置
        agent_observation=LidarStateObservation,  # 使用激光雷达状态观察
    )

    # 添加多传感器配置（如果需要）
    if args.observation == "all":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(
                    rgb_camera=(RGBCamera, 1920, 1080),
                    depth_camera=(DepthCamera, 640, 640),
                    semantic_camera=(SemanticCamera, 640, 640),
                ),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )

    # 创建环境
    env = SidewalkDynamicMetaUrbanEnv(config)
    
    # 创建LLM控制器
    llm_controller = LLMDriveController(api_key=args.api_key, model=args.model)
    
    # 重置环境
    observation, info = env.reset(seed=30)
    
    print("=" * 60)
    print("MetaUrban LLM驾驶控制演示")
    print("=" * 60)
    print(f"使用模型: {args.model}")
    print(f"地图类型: {map_type}")
    print(f"物体密度: {args.density_obj}")
    print(f"行人密度: {args.density_ped}")
    print("=" * 60)
    
    try:
        step_count = 0
        total_reward = 0.0
        
        while step_count < args.max_steps:
            # 获取当前车辆
            vehicle = env.vehicle
            
            # 获取LLM的驾驶动作
            steering, throttle = llm_controller.get_action(observation, vehicle, info)
            action = [steering, throttle]
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            # 打印状态信息
            if step_count % 10 == 0:
                print(f"步数: {step_count}, 累计奖励: {total_reward:.3f}, "
                      f"速度: {vehicle.speed_km_h:.1f} km/h")
            
            # 检查是否结束
            if terminated or truncated:
                print(f"\n回合结束! 原因: {info}")
                print(f"总步数: {step_count}")
                print(f"总奖励: {total_reward:.3f}")
                
                # 重置环境继续下一回合
                observation, info = env.reset(
                    ((env.current_seed + 1) % config['num_scenarios']) + env.engine.global_config['start_seed']
                )
                step_count = 0
                total_reward = 0.0
                print("\n开始新回合...")
                print("=" * 60)
                
            # 添加小延迟以便观察
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("关闭环境...")
        env.close()
        print("程序结束")


if __name__ == "__main__":
    main() 
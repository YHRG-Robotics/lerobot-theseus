# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

from lerobot.common.robot_devices.robots.configs import Theseus_S1RobotConfig
from lerobot.common.robot_devices.robots.theseus import make_s1_arm_from_configs
from S1_SDK import S1_arm


PYNPUT_AVAILABLE = True
try:
    # Only import if there's a valid X server or if we're not on a Pi
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        print("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    print(f"Could not import pynput: {e}")


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    return safe_goal_pos

def move_arm_linear_to_target(arm: S1_arm,steps: int = 30,sleep_time: float = 0.01):
    arm.refresh()
    current_joints = arm.get_pos()[:6]  # 只取前6个关节

    current_joints = np.array(current_joints, dtype=np.float32)
    target_joints = np.zeros(6, dtype=np.float32)

    trajectory = np.linspace(current_joints, target_joints, steps)

    for i, joints in enumerate(trajectory):
        arm.joint_control_mit(joints.tolist())
        time.sleep(sleep_time)


class Theseus_S1Robot:
    def __init__(self, config: Theseus_S1RobotConfig):
        self.robot_type = config.type
        self.config = config
        self.leader_arms = {"leader":make_s1_arm_from_configs(self.config.leader_arms),
                            "leader2":make_s1_arm_from_configs(self.config.leader_arms)}
        self.follower_arms = {"follow":make_s1_arm_from_configs(self.config.follower_arms),
                              "follow2":make_s1_arm_from_configs(self.config.follower_arms)}
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False
        self.last_frames = {}
        self.last_present_speed = {}

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    
    @property
    def motor_features(self) -> dict:
        follower_arm_names = [
            "waist",
            "shouldert",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
        ]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(follower_arm_names),),
                "names": follower_arm_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(follower_arm_names),),
                "names": follower_arm_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0
    
    @property
    def available_arms(self):
        available = []
        for name in self.leader_arms:
            available.append(get_arm_id(name, "leader"))
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        return available
    
    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

    
        # Check both arms can be read

        for name in self.follower_arms:
            self.follower_arms[name].refresh()
            self.follower_arms[name].get_pos()
            self.follower_arms[name].enable()
        for name in self.leader_arms:
            self.leader_arms[name].refresh()
            self.leader_arms[name].get_pos()
            self.leader_arms[name].enable()

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True
    
    def capture_observation(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot is not connected.")
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            self.follower_arms[name].refresh()
            follower_pos[name] = self.follower_arms[name].get_pos()
            follower_pos[name] = torch.from_numpy(follower_pos[name], dtype=torch.float32)
            # self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            # self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict
    

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms["follow"].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            # if self.config.max_relative_target is not None:
            #     present_pos = self.follower_arms[name].read("Present_Position")
            #     present_pos = torch.from_numpy(present_pos)
            #     goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].joint_control_mit( goal_pos)

        return torch.cat(action_sent)


    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            self.leader_arms[name].refresh()
            leader_pos[name] = self.leader_arms[name].get_pos().astype(np.float32)
            self.leader_arms[name].gravity()
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            # self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            # leader_pos["leader"].refresh()
            goal_pos = leader_pos["leader"]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            # if self.config.max_relative_target is not None:
            #     self.follower_arms[name].refresh()
            #     present_pos = self.follower_arms[name].get_pos()
            #     present_pos = torch.from_numpy(present_pos)
            #     goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].joint_control_mit( goal_pos)
            self.follower_arms[name].control_gripper(goal_pos[-1],0.1)
            # self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            self.follower_arms[name].refresh()
            follower_pos[name] = self.follower_arms[name].get_pos().astype(np.float32)
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            # self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            # self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict
    
    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            move_arm_linear_to_target(self.follower_arms[name],steps=100,sleep_time=0.02)
            self.follower_arms[name].disable()

        for name in self.leader_arms:
            move_arm_linear_to_target(self.leader_arms[name],steps=100,sleep_time=0.02)
            self.leader_arms[name].disable()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

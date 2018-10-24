from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.drivers.turtlebot import Turtlebot
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
import pybullet_data
import cv2

CALC_OBSTACLE_PENALTY = 1

tracking_camera = {
    'yaw': 0,
    'z_offset': 0.7,
    'distance': 1,
    'pitch': -20
}

class TurtlebotBaseEnv(CameraRobotEnv):

    distance_weight = 1.0

    def __init__(self, config, controller=None, step_limit=None, gpu_count=0):
        self.config = self.parse_config(config)

        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Turtlebot(self.config, env=self, 
                                       use_controller=controller))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0
        self.step_limit = step_limit
        self.steps = 0

        self._last_base_position = None

    def get_odom(self):
        return np.array(self.robot.body_xyz) - np.array(self.config["initial_pos"]), np.array(self.robot.body_rpy)

    def reset(self):
        obs = CameraRobotEnv._reset(self)
        self.total_frame = 0
        self.total_reward = 0
        self.steps = 0
        return obs

    def _termination(self, debugmode=False):
        if self.step_limit:
            return self.steps >= self.step_limit
        return False

    def calc_rewards_and_done(self, action, state):
        done = self._termination(state)
        rewards = self._rewards(a)
        return rewards, done

class TurtlebotForwardWalkEnv(TurtlebotBaseEnv):
    def __init__(self, config, gpu_count=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "Test Env")
        TurtlebotBaseEnv.__init__(self, config, gpu_count=gpu_count)

    def _rewards(self, action=None, debugmode=False):
        return [0,]

"""
class TurtlebotNavigateSpeedControlEnv(TurtlebotNavigateEnv):
    def __init__(self, config, gpu_count=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        TurtlebotNavigateEnv.__init__(self, config, gpu_count)
        self.robot.keys_to_action = {
            (ord('s'), ): [-0.1,0], ## backward
            (ord('w'), ): [0.1,0], ## forward
            (ord('d'), ): [0,0.1], ## turn right
            (ord('a'), ): [0,-0.1], ## turn left
            (): [0,0]
        }

        self.base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])
        self.base_action_v = np.array([0.001, 0.001, 0.001, 0.001])
        self.action_space = gym.spaces.Discrete(5)
        #control_signal = -0.5
        #control_signal_omega = 0.5
        self.v = 0
        self.omega = 0
        self.kp = 100
        self.ki = 0.1
        self.kd = 25
        self.ie = 0
        self.de = 0
        self.olde = 0
        self.ie_omega = 0
        self.de_omega = 0
        self.olde_omage = 0

    def step(self, action):
        real_action = [action[0]+action[1], action[0]-action[1]]

        obs, rew, env_done, info = TurtlebotNavigateEnv.step(self, real_action)

        self.v = obs["nonviz_sensor"][3]
        self.omega = obs["nonviz_sensor"][-1]

        return obs,rew,env_done,info
"""

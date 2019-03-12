from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.envs.env_real import RealEnv
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

class TurtlebotRealNavigateEnv(RealEnv):
    def __init__(self, config, controller=None, step_limit=None, gpu_count=0):
        self.config = self.parse_config(config)
        RealEnv.__init__(self, self.config)
        self.robot_introduce(Turtlebot(self.config, env=self, 
                             use_controller=controller))

    def _step(self, action):
        return RealEnv._step(self, action)

    def _reset(self):
        obs, rew, done, info = RealEnv._reset(self)
        return obs

    def calc_rewards_and_done(self, action, state):
        rew = 0
        done = False
        return rew, done

    def _rewards(self, action=None, debugmode=False):
        return [0,]

class TurtlebotRealPlanningEnv(RealEnv):
    def __init__(self, config, controller=None, step_limit=None, gpu_count=0):
        self.config = self.parse_config(config)
        RealEnv.__init__(self, self.config)
        self.robot_introduce(Turtlebot(self.config, env=self, 
                             use_controller=controller))
        self.goal_location = np.array([self.config["target_pos"][0], self.config["target_pos"][1]])
        self.target_dim = self.config["target_dim"]

    def _step(self, action):
        self.obs, rew, done, info = RealEnv._step(self, action)
        self.odom = info["odom"]
        self.obs["target"] = self.get_target_observation()
        return self.obs, rew, done, info

    def _reset(self):
        self.obs, rew, done, info = RealEnv._reset(self)
        self.odom = info["odom"]
        self.obs["target"] = self.get_target_observation()
        print(self.obs["target"].shape)
        return self.obs

    def calc_rewards_and_done(self, action, state):
        rew = 0
        done = False
        return rew, done

    def _rewards(self, action=None, debugmode=False):
        return [0,]

    def get_target_observation(self):
        self.agent_pos = np.array([self.odom["x"], self.odom["y"]])
        self.target_vector = self.goal_location - self.agent_pos
        r = np.linalg.norm(self.target_vector)
        angle_to_target = np.arctan2(self.target_vector[1], self.target_vector[0])
        theta = angle_to_target - self.odom["heading"]
        #target_3vector = np.array([np.cos(angle_to_target), np.sin(angle_to_target), r])
        target_3vector = np.array([np.cos(theta), np.sin(theta), r])
        #target_stack =  np.moveaxis(np.tile(target_3vector, (self.target_dim,self.target_dim,1)), -1, 0)
        target_stack = target_3vector
        return target_stack
    

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

    def step(self, action):
        state = CameraRobotEnv._step(self, action)
        self.steps += 1
        return state

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

class TurtlebotMotorServoingEnv(TurtlebotBaseEnv):

    _init_base_position = None

    def __init__(self, config, step_limit=None, gpu_count=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "Test Env")
        TurtlebotBaseEnv.__init__(self, config, step_limit=step_limit, gpu_count=gpu_count)

    def _rewards(self, action=None, debugmode=False):
        if self._init_base_position is None:
            self._init_base_position = np.array(self.robot.get_position())
        current_base_position = np.array(self.robot.get_position())
        progress_reward = np.linalg.norm(current_base_position - self._init_base_position)
        reward = []
        reward.append(self.distance_weight*progress_reward)
        reward = sum(reward)
        return [reward,]

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

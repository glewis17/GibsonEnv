import numpy as np
import zmq
import gym, gym.spaces
import yaml

from gibson.envs.goggle import Goggle
from gibson.envs.env_bases import *

class RealEnv(BaseEnv):

    def __init__(self, config):
        BaseEnv.__init__(self, config, "building", {})

        self.goggles = Goggle()
        self.zmq_context = zmq.Context()

        self.zmq_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_sub_socket.connect("tcp://171.64.70.204:5556")
        self.zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "image")

        self.zmq_pub_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_socket.bind("tcp://*:5556")

        self.robot = None
        self._robot_introduced = False

    def __del__(self):
        self.zmq_context.destroy()

    def robot_introduce(self, robot):
        self.robot = robot
        self.robot.env = self
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        self._robot_introduced = True

    def _get_data(self):
        res = self.zmq_sub_socket.recv_multipart()
        data = np.frombuffer(res[1], dtype=np.uint8)
        data = np.resize(data, (240, 320, 3))
        goggle_img = self.goggles.rgb_callback(data)
        #goggle_img = np.moveaxis(goggle_img, -1, 0) # swap for pytorch
        return goggle_img

    def _step(self, action):
        self.zmq_pub_socket.send_string("action %s" % str(action))
        obs = self._get_data()
        rew = 0
        env_done = False
        info = {}
        return obs, rew, env_done, info

    def _reset(self):
        return self._get_data()

    def render(self, mode='human', close=False):
        return None

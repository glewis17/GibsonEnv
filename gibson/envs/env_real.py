import numpy as np
import zmq
import gym, gym.spaces
import yaml
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt
import threading
import time

from gibson.envs.goggle import Goggle
from gibson.envs.env_bases import *
from gibson.envs.env_ui import *

TURTLEBOT_IP = '171.64.70.150'
PORT = 5559

class RealEnv(BaseEnv):

    def __init__(self, config):
        BaseEnv.__init__(self, config, "building", {})
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(TURTLEBOT_IP, PORT))

        if self.config["display_ui"]:
            self.port_ui = 5552
            if self.config["display_ui"]:
                self.UI = OneViewUI(self.config["resolution"], self, self.port_ui)

    def __del__(self):
        self.context.destroy()

    def robot_introduce(self, robot):
        self.robot = robot
        self.robot.env = self
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        self._robot_introduced = True

    def _step(self, action):
        self.socket.send_string("action %s" % str(action))
        data = self.socket.recv_multipart()
        timestep = data[2].decode("utf-8")
        data = np.frombuffer(data[1], dtype=np.uint8)
        data = np.resize(data, (240, 320, 3))
        self.obs = {}
        self.obs["rgb_filled"] = data[:,:,::-1]
        self.obs["nonviz_sensor"] = np.zeros(3)
        if self.config["display_ui"]:
            self.UI.refresh()
            self.UI.update_view(self.render(), View.RGB_FILLED)
        return self.obs, 0, False, {'timestep': timestep}

    def _reset(self):
        print("Sent action to robot")
        self.socket.send_string("action 3")
        print("Waiting for reply from robot")
        data = self.socket.recv_multipart()
        data = np.frombuffer(data[1], dtype=np.uint8)
        data = np.resize(data, (240, 320, 3))
        self.obs = {}
        self.obs["rgb_filled"] = data[:,:,::-1]
        self.obs["nonviz_sensor"] = np.zeros(3)
        if self.config["display_ui"]:
            self.UI.refresh()
            self.UI.update_view(self.render(), View.RGB_FILLED)
        return self.obs

    def render(self, mode='human', close=False):
        img = imresize(self.obs["rgb_filled"][:,40:280], (256, 256, 3))
        return img

    def get_keys_to_action(self):
        return self.robot.keys_to_action

    def get_key_pressed(self, relevant=None):
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

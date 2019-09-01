import numpy as np
import zmq
import gym, gym.spaces
import yaml
import cv2
from scipy.misc import imresize
import threading
import time
import json
import math
import zlib
from numba import cuda
import fisheye_to_rectilinear as f2r
import os

from gibson.envs.goggle import Goggle
from gibson.envs.env_bases import *
from gibson.envs.env_ui import *
import matplotlib.pyplot as plt

#TURTLEBOT_IP = '171.64.70.187'
#TURTLEBOT_IP = '10.42.0.124'
TURTLEBOT_IP = '192.168.1.3'
PORT = 5559

class RealEnv(BaseEnv):

    def __init__(self, config):
        BaseEnv.__init__(self, config, "building", {})
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(TURTLEBOT_IP, PORT))
        #self.goggles = Goggle()
        self.camera = "kinect"
        #self.camera = "theta"
        if self.camera == "kinect":
            self.in_img_shape = (480, 640, 3)
            #self.in_img_shape = (240, 320, 3)
            self.out_img_shape = self.in_img_shape
        elif self.camera == "theta": 
            self.fisheye_img_shape = (960, 960, 3)
            #self.in_img_shape = (960, 960, 3)
            self.in_img_shape = (490, 490, 3)
            #max_radius_in = self.in_img_shape[0]//2
            max_radius_in = self.fisheye_img_shape[0]//2
            max_FOV_in = math.pi
            focal_length = max_radius_in / (max_FOV_in/2.)
            max_FOV_out = np.deg2rad(90.)
            max_radius_out = int(focal_length*math.tan(max_FOV_out/2.))
            #max_radius_out = int(focal_length*(max_FOV_out/2.))
            self.out_img_shape = (2*max_radius_out, 2*max_radius_out, 3)
            self.out_img = np.empty(self.out_img_shape, dtype='float32')
            self.out_img_device = cuda.to_device(self.out_img)
            self.defisheye = f2r.build_grid(self.out_img, f2r.fisheye_to_rectilinear_kernel)
        self.obs = {}
        self.step = 0

        """
        if self.config["display_ui"]:
            self.port_ui = 5552
            if self.config["display_ui"]:
                self.UI = OneViewUI(self.config["resolution"], self, self.port_ui)
        """

    def __del__(self):
        self.context.destroy()

    def robot_introduce(self, robot):
        self.robot = robot
        self.robot.env = self
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.out_img_shape)
        self.sensor_space = self.robot.sensor_space
        self._robot_introduced = True

    def _receive(self):
        data = self.socket.recv_multipart()
        rec = zlib.decompress(data[1])
        img = np.frombuffer(rec, dtype=np.uint8)
        img = np.resize(img, self.in_img_shape)
        img = img[:, :, ::-1]
        res = None
        if self.camera == "kinect":
            res = img
        elif self.camera == "theta":
            img_max_val = np.amax(img)
            img_min_val = np.amin(img)
            img = f2r.normalize_image(img)
            img_device = cuda.to_device(img)
            self.defisheye(self.out_img_device, img_device, np.array(self.fisheye_img_shape))
            res = self.out_img_device.copy_to_host()
            #res = (res * img_max_val) + img_min_val
            res = f2r.unnormalize_image(res, img_max_val, img_min_val)
            res = res.astype('uint8')
        self.obs["rgb_filled"] = res
        self.obs["nonviz_sensor"] = np.zeros(3)
        timestep = None
        if len(data) > 2:
            timestep = data[2].decode("utf-8")
        odom = None
        if len(data) > 3:
            odom = json.loads(data[3].decode("utf-8"))
        return {'timestep': timestep, 'odom': odom, 'image': res}

    def _step(self, action):
        self.step += 1
        self.socket.send_string("action %s" % str(action))
        """
        data = self.socket.recv_multipart()
        img = np.frombuffer(data[1], dtype=np.uint8)
        img = np.resize(img, (240, 320, 3))
        img = img[:,:,::-1]-np.zeros_like(img)
        timestep = data[2].decode("utf-8")
        odom = None
        if len(data) > 3:
            odom = json.loads(data[3].decode("utf-8"))
        depth = None
        if len(data) > 4:
            depth = np.frombuffer(data[4], dtype=np.int16)
            depth = np.resize(depth, (240, 320, 1))
        self.obs = {}
        #self.obs["rgb_filled"] = self.goggles.rgb_callback(img, depth)
        self.obs["rgb_filled"] = img
        self.obs["nonviz_sensor"] = np.zeros(3)
        """
        out_dict = self._receive()
        #if self.config["display_ui"]:
        #    self.UI.refresh()
        #    self.UI.update_view(self.render(), View.RGB_FILLED)
        #return self.obs, 0, False, {'timestep': timestep, 'odom': odom, 'depth': depth}
        return self.obs, 0, False, out_dict

    def _reset(self):
        print("Sent action to robot")
        self.socket.send_string("action 3")
        print("Waiting for reply from robot")
        """
        data = self.socket.recv_multipart()
        img = np.frombuffer(data[1], dtype=np.uint8)
        img = np.resize(img, (240, 320, 3))
        img = img[:, :, ::-1]-np.zeros_like(img)
        img = np.ones_like(img)
        timestep = data[2].decode("utf-8")
        odom = None
        if len(data) > 3:
            odom = json.loads(data[3].decode("utf-8"))
        depth = None
        if len(data) > 4:
            depth = np.frombuffer(data[4], dtype=np.uint16)
            depth = np.resize(depth, (240, 320, 1))
        self.obs = {}
        #self.obs["rgb_filled"] = self.goggles.rgb_callback(img, depth)
        self.obs["rgb_filled"] = img
        self.obs["nonviz_sensor"] = np.zeros(3)
        """
        out_dict = self._receive()
        #if self.config["display_ui"]:
        #    self.UI.refresh()
        #    self.UI.update_view(self.render(), View.RGB_FILLED)
        #return self.obs, 0, False, {'timestep': timestep, 'odom': odom}
        return self.obs, 0, False, out_dict

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

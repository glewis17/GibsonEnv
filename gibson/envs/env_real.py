import numpy as np
import zmq
import gym

from gibson.envs.goggle import Goggle

class RealEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        self.goggles = Goggle()
        self.zmq_context = zmq.Context()

        self.zmq_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_sub_socket.connect("tcp://171.64.70.204:5556")
        self.zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "image")

        self.zmq_pub_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_socket.bind("tcp://*:5556")

    def _get_data(self):
        res = self.zmq_sub_socket.recv_multipart()
        data = np.frombuffer(res[1], dtype=np.uint8)
        data = np.resize(data, (240, 320, 3))
        goggle_img = self.goggles.rgb_callback(data)
        return goggle_img

    def step(self, action):
        self.zmq_pub_socket.send_string("action %s" % str(action))
        return self._get_data()

    def reset(self):
        return self._get_data()

    def render(self, mode='human', close=False):
        pass

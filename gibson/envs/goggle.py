#!/usr/bin/python
import argparse
import os
import numpy as np
from gibson import assets
from torchvision import datasets, transforms

import gym
import sys
import time
import matplotlib
import time
import pygame
import pybullet as p
from gibson.core.render.profiler import Profiler
from gibson.learn.completion import CompletionNet
import torch.nn as nn
import torch
from torch.autograd import Variable
assets_file_dir = os.path.dirname(assets.__file__)

class Goggle:
    def __init__(self):
        #self.rgb = None
        self.depth = None

        self.model = self.load_model()
        self.imgv = Variable(torch.zeros(1, 3, 240, 320), volatile=True).cuda()
        self.maskv = Variable(torch.zeros(1, 2, 240, 320), volatile=True).cuda()
        self.mean = torch.from_numpy(np.array([0.57441127, 0.54226291, 0.50356019]).astype(np.float32))
        self.mean = self.mean.view(3, 1, 1).repeat(1, 240, 320)

    def load_model(self):
        comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        comp = nn.DataParallel(comp).cuda()
        comp.load_state_dict(
            torch.load(os.path.join(assets_file_dir, "unfiller_256.pth")))

        model = comp.module
        model.eval()
        print(model)
        return model

    def rgb_callback(self, img):
        rows, cols, _ = img.shape
        print(img.shape)

        tf = transforms.ToTensor()
        #img = img[:, :, np.newaxis]
        source = tf(img)
        mask = (torch.sum(source[:3, :, :], 0) > 0).float().unsqueeze(0)

        self.imgv.data.copy_(source)
        self.maskv.data.copy_(mask)
        recon = self.model(self.imgv, self.maskv)
        goggle_img = (recon.data.clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        return goggle_img

#goggle = Goggle()
#goggle.run()

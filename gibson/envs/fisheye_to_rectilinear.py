import os
import math
import numpy as np
from numba import cuda

def normalize_image(img):
    print("IN GIBSON F2R")
    img = img.astype('float32')
    img = img - np.amin(img)
    img = img / np.amax(img)
    return np.ascontiguousarray(img)

@cuda.jit(device=True)
def fisheye_projection_radius(focal_length, theta, proj):
    radius = -1.
    if proj == 0: # equidistant
        radius = focal_length*theta
    elif proj == 1: # equisolid
        radius = focal_length*2.*math.sin(theta/2.)
    elif proj == 2: # orthographic
        radius = focal_length*math.sin(theta)
    elif proj == 3: # stereographic
        radius = math.tan(theta/2.)*2.
    return radius

@cuda.jit(device=True)
def fisheye_projection_focal_length(radius, theta, proj):
    focal_length = -1.
    if proj == 0: # equidistant
        focal_length = radius / theta
    elif proj == 1: # equisolid
        focal_length = radius / (2.*math.sin(theta/2.))
    elif proj == 2: # orthographic
        focal_length = radius / math.sin(theta)
    elif proj == 3: # stereographic
        focal_length = radius / (2.*math.tan(theta/2.))
    return focal_length

@cuda.jit
# Assumes input and output images are squares
def fisheye_to_rectilinear_kernel(out_img, in_img, fisheye_img_shape):
    x_out, y_out = cuda.grid(2)
    x_out_valid = x_out >= 0 and x_out < out_img.shape[0]
    y_out_valid = y_out >= 0 and y_out < out_img.shape[1]
    if not (x_out_valid and y_out_valid):
        return
    in_center = (in_img.shape[0]//2, in_img.shape[1]//2)
    out_center = (out_img.shape[0]//2, out_img.shape[1]//2)
    #max_radius_in = in_img.shape[0]//2
    max_radius_in = fisheye_img_shape[0]//2
    max_FOV_in = math.pi
    max_theta_in = max_FOV_in/2.
    proj = 0 # equidistant
    focal_length = fisheye_projection_focal_length(max_radius_in, max_theta_in, proj)
    x_rel_out = x_out - out_center[0]
    y_rel_out = y_out - out_center[1]
    radius_out = math.sqrt(float(x_rel_out**2 + y_rel_out**2))
    theta = math.atan(radius_out / focal_length)
    radius_in = fisheye_projection_radius(focal_length, theta, proj)
    ratio = None
    if abs(radius_out) < 1e-7:
        ratio = 1
    else:
        ratio = (radius_in / radius_out)
    x_rel_in = ratio*x_rel_out
    y_rel_in = ratio*y_rel_out
    x_in = int(x_rel_in + in_center[0])
    y_in = int(y_rel_in + in_center[1])
    x_in_valid = x_in >= 0 and x_in < in_img.shape[0]
    y_in_valid = y_in >= 0 and y_in < in_img.shape[1]
    if (x_in_valid and y_in_valid):
        out_img[x_out, y_out, 0] = in_img[x_in, y_in, 0]
        out_img[x_out, y_out, 1] = in_img[x_in, y_in, 1]
        out_img[x_out, y_out, 2] = in_img[x_in, y_in, 2]

def build_grid(out_img, func):
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(out_img.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(out_img.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return func[blockspergrid, threadsperblock]

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import zmq

class ROSHandler:

    def __init__(self):
        rospy.init_node('ros2zmq')

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5556")

        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/gibson_ros/camera_goggle/rgb/image", Image, self.rgb_callback)

    def rgb_callback(self, data):
        print("In rgb callback!")
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        img = cv2.resize(img, (320,240))
        self.socket.send(img)

rh = ROSHandler()
rospy.spin()

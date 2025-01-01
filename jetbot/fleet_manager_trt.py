#!/usr/bin/env python
# coding: utf-8

# # Object Following - Live Demo
# 
# In this notebook we'll show how you can follow an object with JetBot!  We'll use a pre-trained neural network
# that was trained on the [COCO dataset](http://cocodataset.org) to detect 90 different common objects.  These include
# 
# * Person (index 0)
# * Cup (index 47)
# 
# and many others (you can check [this file](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_complete_label_map.pbtxt) for a full list of class indices).  The model is sourced from the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection),
# which provides utilities for training object detectors for custom tasks also!  Once the model is trained, we optimize it using NVIDIA TensorRT on the Jetson Nano.
# 
# This makes the network very fast, capable of real-time execution on Jetson Nano!  We won't run through all of the training and optimization steps in this notebook though.
# 
# Anyways, let's get started.  First, we'll want to import the ``ObjectDetector`` class which takes our pre-trained SSD engine.

# ### Compute detections on single camera image

# In[ ]:

# from queue import Empty
import cv2
import numpy as np
# import traitlets
from traitlets import Float, Bool, Any

from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
from jetbot import ObjectFollower
from jetbot import RoadCruiserTRT
# from jetbot.utils import get_cls_dict_yolo, get_cls_dict_ssd

import time


def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def object_center_detection(det):
    """Computes the center x, y coordinates of the object"""
    # print(self.matching_detections)
    bbox = det['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    object_center = (center_x, center_y)
    return object_center


class FleeterTRT(ObjectFollower, RoadCruiserTRT):
    cap_image = Any()
    # model parameters
    # follower_model = Unicode(default_value='').tag(config=True)
    # type_follower_model = Unicode(default_value='').tag(config=True)
    # cruiser_model = Unicode(default_value='').tag(config=True)
    # type_cruiser_model = Unicode(default_value='').tag(config=True)
    conf_th = Float(default_value=0.5).tag(config=True)
    # label = Integer(default_value=1).tag(config=True)
    # label_text = Unicode(default_value='').tag(config=True)
    # control parameters
    # speed_rc = Float(default_value=0).tag(config=True)
    speed_fm = Float(default_value=0.10).tag(config=True)
    speed_gain_fm = Float(default_value=0.01).tag(config=True)
    speed_dev_fm = Float(default_value=0.5).tag(config=True)
    turn_gain_fm = Float(default_value=0.3).tag(config=True)
    steering_bias_fm = Float(default_value=0.0).tag(config=True)
    # blocked = Float(default_value=0).tag(config=True)
    target_view = Float(default_value=0.6).tag(config=True)
    mean_view = Float(default_value=0).tag(config=True)
    e_view = Float(default_value=0).tag(config=True)
    # is_detecting = Bool(default_value=True).tag(config=True)
    is_detected = Bool(default_value=False).tag(config=True)

    def __init__(self, init_sensor_fm=False):

        # the parent classes (ObjectFollower, RoadCruiserTRT) may revisit during initialization,
        # causing the  re-instantiation error of camera and robot motor,
        # which should be avoided when design the parent classes
        ObjectFollower.__init__(self, init_sensor_of=False)
        RoadCruiserTRT.__init__(self, init_sensor_rc=False)

        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_object = None
        self.is_detecting = True
        self.is_detected = False
        self.is_loaded = False

        self.robot = None
        self.capturer = None
        if init_sensor_fm:
            self.robot = Robot.instance()
            self.capturer = Camera()
            self.img_width = self.capturer.width
            self.img_height = self.capturer.height
            self.cap_image = np.empty(shape=(self.img_height, self.img_width, 3), dtype=np.uint8).tobytes()
            self.current_image = np.empty((self.img_height, self.img_width, 3))

        self.default_speed = self.speed_fm
        self.detect_duration_max = 10
        self.no_detect = 0
        self.target_view = 0.5
        self.mean_view = 0
        self.mean_view_prev = 0
        self.e_view = 0
        self.e_view_prev = 0

        self.execution_time_fm = []
        # self.fps = []

    def execute_fm(self, change):

        # do object following
        start_time = time.time()
        self.execute(change)
        end_time = time.time()
        # self.execution_time.append(end_time - start_time + self.capturer.cap_time)
        self.execution_time_fm.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

        # if the closest object is not detected and followed, perform the road cruising
        if not self.is_detected:
            self.execute_rc(change)
            self.speed_fm = self.speed_rc  # set fleet mge speed to road cruising speed (self.speed)

    def start_fm(self, change):
        self.capturer.unobserve_all()
        self.load_object_detector(change)  # load object detector function in object follower module
        self.load_road_cruiser(change)  # load_road_cruiser function in road_cruiser_trt module

        print("start running!")
        self.capturer.observe(self.execute_fm, names='value')

    def execute(self, change):
        # print("start execution !")
        self.current_image = change['new']

        # compute all detected objects
        self.run_objects_detection()
        self.closest_object_detection()
        # detections = self.object_detector(image)
        # print(self.detections)

        # draw all detections on image
        for det in self.detections[0]:
            bbox = det['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                          (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (255, 0, 0), 2)

        # select detections that match selected class label
        # get detection closest to the center of view field and draw it
        cls_obj = self.closest_object
        if cls_obj is not None:
            self.is_detected = True
            self.no_detect = self.detect_duration_max  # set max detection no to prevent temporary loss of object detection
            bbox = cls_obj['bbox']
            cv2.rectangle(self.current_image, (int(self.img_width * bbox[0]), int(self.img_height * bbox[1])),
                          (int(self.img_width * bbox[2]), int(self.img_height * bbox[3])), (0, 255, 0), 5)

            self.mean_view = 0.4 * (bbox[2] - bbox[0]) + 0.6 * self.mean_view_prev
            self.e_view = self.target_view - self.mean_view
            if np.abs(self.e_view / self.target_view) > 0.1:
                self.speed_fm = self.speed_fm + self.speed_gain_fm * self.e_view + self.speed_dev_fm * (
                        self.e_view - self.e_view_prev)
            # self.speed = self.speed_fm

            self.mean_view_prev = self.mean_view
            self.e_view_prev = self.e_view

        # otherwise go forward if no target detected
        if cls_obj is None:
            if self.no_detect <= 0:  # if object is not detected for a duration, road cruising
                self.mean_view = 0.0
                self.mean_view_prev = 0.0
                self.is_detected = False
                self.cap_image = bgr8_to_jpeg(self.current_image)
                return
            else:
                self.no_detect -= 1  # observe for a duration for the miss of object detection
            # self.robot.forward(float(self.speed))

        # otherwise, steer towards target
        else:
            # move the robot forward and steer proportional target's x-distance from center
            center = object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed_fm + self.turn_gain_fm * center[0] + self.steering_bias_fm),
                float(self.speed_fm - self.turn_gain_fm * center[0] + self.steering_bias_fm)
            )

        # update image widget
        self.cap_image = bgr8_to_jpeg(self.current_image)
        # print("ok!")
        # return self.cap_image

    def stop_fm(self, change):
        from jetbot.utils import plot_exec_time
        print("start stopping!")

        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # self.road_cruiser.stop_cruising(change)
        # plot execution time of road cruiser model processing
        cruiser_model_name = "road cruiser model"
        cruiser_model_str = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], cruiser_model_name, cruiser_model_str)

        # plot execution time of fleet controller model processing
        follower_model_name = "fleet controller model"
        follower_model_str = self.follower_model.split("/")[-1].split(".")[0]
        plot_exec_time(self.execution_time_fm[1:], follower_model_name, follower_model_str)
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.follower_model.split(".")[0])
        # plt.show()

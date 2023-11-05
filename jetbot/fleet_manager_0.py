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

from queue import Empty
import torch.nn.functional as F
import cv2
import numpy as np
import traitlets
import os
import time

# from jetbot import ObjectDetector
# from jetbot.object_detection_yolo import ObjectDetector_YOLO
from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
from jetbot import ObjectDetector
from jetbot import RoadCruiser
from jetbot.utils import get_cls_dict_yolo, get_cls_dict_ssd

import time

class Fleeter(traitlets.HasTraits):
    
    cap_image = traitlets.Any()
    label = traitlets.Integer(default_value=1).tag(config=True)
    label_text = traitlets.Unicode(default_value='').tag(config=True)
    speed = traitlets.Float(default_value=0.15).tag(config=True)
    turn_gain = traitlets.Float(default_value=0.3).tag(config=True)
    steering_bias = traitlets.Float(default_value=0.0).tag(config=True)
    blocked = traitlets.Float(default_value=0).tag(config=True)
    target_view= traitlets.Float(default_value=0.6).tag(config=True)
    mean_view = traitlets.Float(default_value=0).tag(config=True)
    e_view = traitlets.Float(default_value=0).tag(config=True)
    is_dectecting = traitlets.Bool(default_value=True).tag(config=True)
    is_dectected = traitlets.Bool(default_value=False).tag(config=True)
    
    def __init__(self, follower_model='ssd_mobilenet_v2_coco_onnx.engine', type_follower_model="SSD", cruiser_model='resnet18', type_cruiser_model='resnet'):

        self.follower_model = follower_model
        self.type_follower_model = type_follower_model

        # self.obstacle_detector = Avoider(model_params=self.avoider_model)
        if self.type_follower_model == "SSD" or self.type_follower_model == "YOLO":
            # from jetbot import ObjectDetector
            self.object_detector = ObjectDetector(self.follower_model, self.type_follower_model)
        # elif type_model == "YOLO":
        #    from jetbot.object_detection_yolo import ObjectDetector_YOLO
        #    self.object_detector = ObjectDetector_YOLO(self.follower_model)

        self.cruiser_model = cruiser_model
        self.type_cruiser_model = type_cruiser_model
        self.road_cruiser = RoadCruiser(cruiser_model = self.cruiser_model, type_cruiser_model = self.type_cruiser_model)
        
        # self.robot = self.road_cruiser.robot
        self.robot = Robot.instance()
        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_objec = None
        self.is_dectecting = True
        self.is_dectected = False

        # Camera instance would be better to put after all models instantiation
        # self.capturer = Camera()
        self.capturer = self.road_cruiser.camera
        self.img_width = self.capturer.width
        self.img_height = self.capturer.height
        self.cap_image = np.empty((self.img_height, self.img_width, 3), dtype=np.uint8).tobytes()
        self.current_image = np.empty((self.img_height, self.img_width, 3))
        
        self.default_speed = self.speed
        self.detect_duration_max = 10
        self.no_detect = 0
        self.target_view = 0.5
        self.mean_view = 0
        self.mean_view_prev = 0
        self.e_view = 0
        self.e_view_prev = 0

        self.execution_time = []
        self.fps = []

    def run_objects_detection(self):
        # self.image = self.capturer.value
        # print(self.image[1][1], np.shape(self.image))
        self.detections = self.object_detector(self.current_image)
        self.matching_detections = [d for d in self.detections[0] if d['label'] == int(self.label)]
        
        if self.type_follower_model == "SSD":
            self.label_text = get_cls_dict_ssd('coco')[int(self.label)]
        elif self.type_follower_model == "YOLO":
            self.label_text = get_cls_dict_yolo('coco')[int(self.label)]
        # print(int(self.label), "\n", self.matching_detections)
        
    def object_center_detection(self, det):
        """Computes the center x, y coordinates of the object"""
        # print(self.matching_detections)
        bbox = det['bbox']
        center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
        center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
        object_center = (center_x, center_y)
        return object_center
    
    def norm(self, vec):
        """Computes the length of the 2D vector"""
        return np.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def closest_object_detection(self):
        """Finds the detection closest to the image center"""
        closest_detection = None
        if len(self.matching_detections) != 0:
            for det in self.matching_detections:
                if closest_detection is None:
                    closest_detection = det
                elif self.norm(self.object_center_detection(det)) < self.norm(self.object_center_detection(closest_detection)):
                    closest_detection = det
        
        self.closest_object =  closest_detection
        
    def execute_fleeting(self, change):
        
        # do object following
        start_time = time.process_time()
        self.execute(change)
        end_time = time.process_time()
        # self.execution_time.append(end_time - start_time + self.capturer.cap_time)
        self.execution_time.append(end_time - start_time)
        self.fps.append(1/(end_time - start_time))

        # if closest object is not detected and followed, do road cruising
        if not self.is_dectected:
            self.road_cruiser.execute(change)

    def start_run(self, change):
        self.capturer.unobserve_all()
        print("start running")
        self.capturer.observe(self.execute_fleeting, names='value')
 
    def execute(self, change):
        # print("start excution !")
        self.current_image = change['new']
        width = self.img_width
        height = self.img_height

        # compute all detected objects
        self.run_objects_detection()
        self.closest_object_detection()
        # detections = self.object_detector(image)
        # print(self.detections)
        
        # draw all detections on image
        for det in self.detections[0]:
            
            bbox = det['bbox']
            cv2.rectangle(self.current_image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
            
        # select detections that match selected class label
        # get detection closest to center of field of view and draw it
        cls_obj = self.closest_object
        if cls_obj is not None:
            self.is_dectected = True
            self.no_detect = self.detect_duration_max           # set max detection no to prevent temperary loss of object detection
            bbox = cls_obj['bbox']
            cv2.rectangle(self.current_image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
            
            self.mean_view = 0.8 * (bbox[2] - bbox[0]) + 0.2 * self.mean_view_prev
            self.e_view = self.target_view - self.mean_view
            
            self.speed = self.speed +  0.01 * self.e_view + 0.5 * (self.e_view - self.e_view_prev)
            
            self.mean_view_prev =  self.mean_view
            self.e_view_prev = self.e_view
            
        # otherwise go forward if no target detected
        if cls_obj is None:
            if self.no_detect <= 0:         # if object is not detected for a duration, road cruising
                self.mean_view = 0.0
                self.mean_view_prev = 0.0
                self.is_dectected = False
                self.cap_image = bgr8_to_jpeg(self.current_image)
                # self.speed = self.default_speed
                return
            else:
                self.no_detect -= 1         # observe for a duration for the miss of object detection
            # self.robot.forward(float(self.speed))
            
        # otherwise steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center =self.object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed + self.turn_gain * center[0] + self.steering_bias),
                float(self.speed - self.turn_gain * center[0] + self.steering_bias)
            )

        # update image widget
        self.cap_image = bgr8_to_jpeg(self.current_image)
        # print("ok!")
        # return self.cap_image
        

    def stop_run(self, change):
        from jetbot.utils import plot_exec_time
        print("start stopping!")
        self.road_cruiser.stop_cruising(change)

        # plot exection time of fleet controller model processing
        model_name = "fleet controller model"
        plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.follower_model.split(".")[0])


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
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import traitlets
import os

# from jetbot import ObjectDetector
# from jetbot.object_detection_yolo import ObjectDetector_YOLO
from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
import time
import threading


#
# model = ObjectDetector('ssd_mobilenet_v2_coco_onnx.engine')
# model = ObjectDetector_YOLO('yolov4-288.engine')


class ObjectFollower(traitlets.HasTraits):
    cap_image = traitlets.Any()
    label = traitlets.Integer(default_value=1).tag(config=True)
    speed = traitlets.Float(default_value=0.15).tag(config=True)
    turn_gain = traitlets.Float(default_value=0.3).tag(config=True)
    steering_bias = traitlets.Float(default_value=0.0).tag(config=True)
    blocked = traitlets.Float(default_value=0).tag(config=True)
    is_dectecting = traitlets.Bool(default_value=True).tag(config=True)

    def __init__(self, follower_model='ssd_mobilenet_v2_coco_onnx.engine',
                 avoider_model='../collision_avoidance/best_model.pth', type_follower_model="SSD"):
        self.follower_model = follower_model
        self.avoider_model = avoider_model

        # self.obstacle_detector = Avoider(model_params=self.avoider_model)
        if type_follower_model == "SSD" or type_follower_model == "YOLO":
            from jetbot import ObjectDetector
            self.object_detector = ObjectDetector(self.follower_model, type_follower_model)
        # elif type_model == "YOLO":
        #    from jetbot.object_detection_yolo import ObjectDetector_YOLO
        #    self.object_detector = ObjectDetector_YOLO(self.follower_model)

        self.robot = Robot()
        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_object = None
        self.is_dectecting = True

        # Camera instance would be better to put after all models instantiation
        self.capturer = Camera()
        self.img_width = self.capturer.width
        self.img_height = self.capturer.height
        self.cap_image = np.empty((self.img_height, self.img_width, 3), dtype=np.uint8).tobytes()

    def run_follower_detection(self):
        # self.image = self.capturer.value
        # print(self.image[1][1], np.shape(self.image))
        self.detections = self.object_detector(self.current_image)
        self.matching_detections = [d for d in self.detections[0] if d['label'] == int(self.label)]
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
                elif self.norm(self.object_center_detection(det)) < self.norm(
                        self.object_center_detection(closest_detection)):
                    closest_detection = det

        self.closest_object = closest_detection

    def start_run(self):
        self.capturer.unobserve_all()
        print("start running")
        self.capturer.observe(self.execute, names='value')

    def execute(self, change):
        # print("start excution !")
        self.current_image = change['new']
        width = self.img_width
        height = self.img_height

        # print(image)
        # ** execute collision model to determine if blocked
        # self.obstacle_detector.detect(self.current_image)
        # self.blocked = self.obstacle_detector.prob_blocked
        # turn left if blocked
        if self.blocked > 0.5:
            #      # robot.left(0.3)
            self.robot.left(0.05)
            self.cap_image = bgr8_to_jpeg(self.current_image)
            return

        # compute all detected objects
        self.run_follower_detection()
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
            bbox = cls_obj['bbox']
            cv2.rectangle(self.current_image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

        # otherwise go forward if no target detected
        if cls_obj is None:
            self.robot.forward(float(self.speed))

        # otherwise steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center = self.object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed + self.turn_gain * center[0] + self.steering_bias),
                float(self.speed - self.turn_gain * center[0] + self.steering_bias)
            )

        # update image widget
        # image_widget.value = bgr8_to_jpeg(image)
        self.cap_image = bgr8_to_jpeg(self.current_image)
        # print("ok!")
        # return self.cap_image

    def stop_run(self):
        # with out:
        print("start stopping!")
        self.capturer.unobserve_all()
        self.robot.stop()
        self.capturer.stop()


class Avoider(object):

    def __init__(self, model_params='../collision_avoidance/best_model.pth'):
        self.model_params = model_params
        self.collision_model = torchvision.models.alexnet(pretrained=False)
        self.collision_model.classifier[6] = torch.nn.Linear(self.collision_model.classifier[6].in_features, 2)
        self.collision_model.load_state_dict(torch.load(self.model_params))
        # collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
        self.device = torch.device('cuda')
        self.collision_model = self.collision_model.to(self.device)
        self.prob_blocked = 0

    def detect(self, image):
        collision_output = self.collision_model(self.preprocess(image)).detach().cpu()
        self.prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
        # blocked_widget.value = prob_blocked

    def preprocess(self, camera_value):
        # global device
        mean = 255.0 * np.array([0.485, 0.456, 0.406])
        stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        normalize = torchvision.transforms.Normalize(mean, stdev)
        x = camera_value
        x = cv2.resize(x, (224, 224))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = normalize(x)
        x = x.to(self.device)
        x = x[None, ...]
        return x

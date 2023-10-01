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
import threading

# from jetbot import ObjectDetector
# from jetbot.object_detection_yolo import ObjectDetector_YOLO
from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg
import time

    # 
    # model = ObjectDetector('ssd_mobilenet_v2_coco_onnx.engine')
    # model = ObjectDetector_YOLO('yolov4-288.engine')


class Follower_Detector(traitlets.HasTraits):
    
    cap_image = traitlets.Any()
    label = traitlets.Integer(default_value=1).tag(config=True)
    speed = traitlets.Float(default_value=0.15).tag(config=True)
    turn_gain = traitlets.Float(default_value=0.3).tag(config=True)
    blocked = traitlets.Float(default_value=0).tag(config=True)
      
    def __init__(self, cam, follower_model='ssd_mobilenet_v2_coco_onnx.engine', type_model="SSD"):
        self.follower_model = follower_model
        self.avoider_model = avoider_model
        self.capturer = cam

        if type_model == "SSD":
            from jetbot import ObjectDetector
            self.object_detector = ObjectDetector(self.follower_model)
        elif type_model == "YOLO":
            from jetbot.object_detection_yolo import ObjectDetector_YOLO
            self.object_detector = ObjectDetector_YOLO(self.follower_model)
        
        # self.obstacle_detector = Avoider(model_params=self.avoider_model)
        self.robot = Robot()
        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_objec = None
        self.img_width = 300
        self.img_height = 300
        self.cap_image = np.empty((self.img_width, self.img_height, 3), dtype=np.uint8).tobytes()
        
        # self.execute({'new': self.capturer.value})

        
    def run_follower_detection(self):
        self.image = self.capturer.value
        # print(self.image[1][1], np.shape(self.image))
        self.detections = self.object_detector(self.image)
        self.matching_detections = [d for d in self.detections[0] if d['label'] == int(self.label)]        
          
    def object_center_detection(self, det):
        """Computes the center x, y coordinates of the object"""
        # self.object_center = None
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
                # center = self.object_center_detection(det)
                if closest_detection is None:
                    closest_detection = det
                elif self.norm(self.object_center_detection(det)) < self.norm(self.object_center_detection(closest_detection)):
                    closest_detection = det
        
        self.closest_object =  closest_detection
        # self.closest_object_center = self.object_center_detection(closest_detection)
        
    def start_run(self):
        # self.cap_image =  bgr8_to_jpeg(self.capturer.value)
        # self.cap_image = self.execute({'new': self.capturer.value})
        self.execute({'new': self.capturer.value})
        time.sleep(2)
        self.capturer.unobserve_all()
        print("start running")
        self.capturer.observe(self.execute, names='value')
        # cap_image = self.cap_image
       
        # return  cap_image
    
    def execute(self, change):
        print("start excution !")
        image = change['new']
        width = self.img_width
        height = self.img_height

        # print(image)
        # ** execute collision model to determine if blocked
        # collision_output = collision_model(preprocess(image)).detach().cpu()
        # prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
        # blocked_widget.value = prob_blocked

        # self.obstacle_detector.detect(image)
        # self.blocked = self.obstacle_detector.prob_blocked
        # turn left if blocked
        # if self.prob_blocked > 0.5:
        #      # robot.left(0.3)
        #    self.robot.left(0.05)
        #    self.cap_image.value = bgr8_to_jpeg(image)
        #    return

        # compute all detected objects
        self.run_follower_detection()
        self.closest_object_detection()
        # detections = self.object_detector(image)
        # print(self.detections)

        # draw all detections on image
        for det in self.detections[0]:
            
            bbox = det['bbox']
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

        # select detections that match selected class label

        # get detection closest to center of field of view and draw it
        cls_obj = self.closest_object
        if cls_obj is not None:
            bbox = cls_obj['bbox']
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

        # otherwise go forward if no target detected
        if cls_obj is None:
            self.robot.forward(float(self.speed))

        # otherwise steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center =self.object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed + self.turn_gain * center[0]),
                float(self.speed - self.turn_gain * center[0])
            )

        # update image widget
        # image_widget.value = bgr8_to_jpeg(image)
        self.cap_image = bgr8_to_jpeg(image)
        # print("ok!")
        # return self.cap_image
        

    def stop_run(self):
        # with out:
        print("start stopping!")
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()
        # clear_output(wait=True)
        # get_ipython().run_line_magic('reset', '-f')


class Obstacle_Avoider(traitlets.HasTraits):
    blocked = traitlets.Float(default_value=0).tag(config=True)
    def __init__(self, cam, model_params='../collision_avoidance/best_model.pth'):
        self.model_params = model_params
        self.collision_model = torchvision.models.alexnet(pretrained=False)
        self.collision_model.classifier[6] = torch.nn.Linear(self.collision_model.classifier[6].in_features, 2)
        self.collision_model.load_state_dict(torch.load(self.model_params))
        self.device = torch.device('cuda')
        self. collision_model = self.collision_model.to(self.device)
        self.prob_blocked = 0
        self.blocked = 0

        self.capturer = cam
        self.img_width = 300
        self.img_height = 300
        
        self.robot = Robot()
               
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

    def start_run(self):
        # self.execute({'new': self.capturer.value})
        # time.sleep(2)
        self.capturer.unobserve_all()
        self.capturer.observe(self.execute, names='value')
        print("start obstacle detection ! ")
       
        # return  cap_image

    def execute(self, change):
        print("start excution !")
        image = change['new']

        self.detect(image)
        self.prob_blocked
        self.blocked = self.prob_blocked
        
        # turn left if blocked
        if self.prob_blocked > 0.5:
            # self.robot.left(0.3)
            self.robot.left(0.05)
        #    return

        # return

def object_follower(fd_model, oa_model, type_model):
    cam = Camera()
    FO = Follower_Detector(cam, fd_model, type_model)
    OA = Obstacle_Avoider(cam, oa_model)
    
    FO.start_run()
    OA.start_run()


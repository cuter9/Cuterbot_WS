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
# from jetbot import RoadCruiser
from jetbot.utils import get_cls_dict_yolo, get_cls_dict_ssd

import torch
import torchvision
import torchvision.transforms as transforms

import time

class Fleeter(traitlets.HasTraits):
    
    cap_image = traitlets.Any()
    label = traitlets.Integer(default_value=1).tag(config=True)
    label_text = traitlets.Unicode(default_value='').tag(config=True)
    speed = traitlets.Float(default_value=0.15).tag(config=True)
    speed_gain = traitlets.Float(default_value=0.01).tag(config=True)
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
        
        self.detections = None
        self.matching_detections = None
        self.object_center = None
        self.closest_objec = None
        self.is_dectecting = True
        self.is_dectected = False

        # Camera instance would be better to put after all models instantiation
        self.capturer = Camera()
        # self.capturer = self.road_cruiser.camera
        self.img_width = self.capturer.width
        self.img_height = self.capturer.height
        self.cap_image = np.empty((self.img_height, self.img_width, 3), dtype=np.uint8).tobytes()
        self.current_image = np.empty((self.img_height, self.img_width, 3))
        
        self.default_speed = self.speed
        self.detect_duration_max = 5
        self.no_detect = self.detect_duration_max
        self.target_view = 0.5
        self.mean_view = 0
        self.mean_view_prev = 0
        self.e_view = 0
        self.e_view_prev = 0

        self.execution_time = []
        self.fps = []
        
        # self.robot = self.road_cruiser.robot
        self.robot = Robot.instance()
        
        self.cruiser_model = cruiser_model
        self.type_cruiser_model = type_cruiser_model
        self.road_cruiser = RoadCruiser_4_ft(cruiser_model = self.cruiser_model, type_cruiser_model = self.type_cruiser_model)
        # self.road_cruiser.robot = self.robot
        # self.robot = self.road_cruiser.robot


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
        
 
    def execute_of(self):
        # print("start execute_of !")
        
        start_time = time.process_time()

        # self.current_image = change['new']
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
            self.speed = self.speed +  self.speed_gain * self.e_view + 0.5 * (self.e_view - self.e_view_prev)
            
            self.mean_view_prev =  self.mean_view
            self.e_view_prev = self.e_view
            
        # otherwise go forward if no target detected
        if cls_obj is None:
            end_time = time.process_time()
            self.execution_time.append(end_time - start_time)
            
            if self.no_detect <= 0:         # if object is not detected for a duration, road cruising
                self.mean_view = 0.0
                self.mean_view_prev = 0.0
                self.is_dectected = False
                # self.no_detect = self.detect_duration_max
                
                # self.road_cruiser.execute_rc(self.current_image)
                # print('no_objects detected !')
                self.cap_image = bgr8_to_jpeg(self.current_image)
                # self.speed = self.default_speed
                return
            
            else:
                self.no_detect -= 1         # observe no objects for a duration for the miss of object detection
            # self.robot.forward(float(self.speed))
            
        # otherwise steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center =self.object_center_detection(cls_obj)
            self.robot.set_motors(
                float(self.speed + self.turn_gain * center[0] + self.steering_bias),
                float(self.speed - self.turn_gain * center[0] + self.steering_bias)
            )
        
            end_time = time.process_time()
            self.execution_time.append(end_time - start_time)

        # update image widget
        self.cap_image = bgr8_to_jpeg(self.current_image)
        # print("ok!")
        # return self.cap_image
        
    def execute_fleeting(self, change):
        # print("start running execute_fleeting")
        # do object following
        # start_time = time.process_time()
        self.current_image = change['new']
        self.execute_of()
        # end_time = time.process_time()
        # self.execution_time.append(end_time - start_time + self.capturer.cap_time)
        # self.execution_time.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

        # if closest object is not detected and followed, do road cruising
        # print('check objects detectd !', self.is_dectected)
        if not self.is_dectected:
            # self.road_cruiser.speed = self.speed_gain
            # self.road_cruiser.speed = self.speed_gain
        #    print('no_objects detected !')
            self.road_cruiser.execute_rc(self.current_image)

    def start_run(self, change):
        self.capturer.unobserve_all()
        print("start running")
        self.capturer.observe(self.execute_fleeting, names='value')
        # self.capturer.observe(self.execute_of, names='value')

    def stop_run(self, change):
        from jetbot.utils import plot_exec_time
        print("start stopping!")
        
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # self.road_cruiser.stop_cruising()
        # plot exection time of road cruiser model processing
        cruiser_model_name = "road cruiser model"
        plot_exec_time(self.road_cruiser.execution_time[1:], cruiser_model_name, self.road_cruiser.cruiser_model_str)
        
        # plot exection time of fleet controller model processing
        fleet_model_name = "fleet controller model"
        plot_exec_time(self.execution_time[1:], fleet_model_name, self.follower_model.split(".")[0])


class RoadCruiser_4_ft(traitlets.HasTraits):
    speed_gain = traitlets.Float(default_value=0.15).tag(config=True)
    steering_gain = traitlets.Float(default_value=0.08).tag(config=True)
    steering_dgain = traitlets.Float(default_value=1.5).tag(config=True)
    steering_bias = traitlets.Float(default_value=0.0).tag(config=True)
    steering = traitlets.Float(default_value=0.0).tag(config=True)
    x_slider = traitlets.Float(default_value=0).tag(config=True)
    y_slider = traitlets.Float(default_value=0).tag(config=True)
    speed = traitlets.Float(default_value=0).tag(config=True)

    def __init__(self, cruiser_model='resnet18', type_cruiser_model='resnet'):
        super().__init__()
        self.cruiser_model_str = cruiser_model
        self.cruiser_model = getattr(torchvision.models, cruiser_model)(pretrained=False)
        self.type_cruiser_model = type_cruiser_model
        if type_cruiser_model == "mobilenet":
            self.cruiser_model.classifier[3] = torch.nn.Linear(self.cruiser_model.classifier[3].in_features, 2)
            self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_' + cruiser_model + '.pth'))

        elif type_cruiser_model == "resnet":
            self.cruiser_model.fc = torch.nn.Linear(self.cruiser_model.fc.in_features, 2)
            self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_' + cruiser_model + '.pth'))
            # self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_resnet34.pth'))
            # model.load_state_dict(torch.load('best_steering_model_xy_resnet50.pth'))

        # self.camera = Camera()
        # self.robot = Robot.instance()
        self.robot = Robot.instance()
        self.angle = 0.0
        self.angle_last = 0.0
        self.execution_time = []
        self.fps = []
        self.x_slider = 0
        self.y_slider = 0

        # model = torchvision.models.mobilenet_v3_large(pretrained=False)
        # model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

        # model = torchvision.models.resnet18(pretrained=False)
        # model = torchvision.models.resnet34(pretrained=False)
        # model = torchvision.models.resnet50(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, 2)
        # model.load_state_dict(torch.load('best_steering_model_xy_mobilenet_v3_large.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet18.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet34.pth'))
        # model.load_state_dict(torch.load('best_steering_model_xy_resnet50.pth'))

        self.device = torch.device('cuda')
        self.cruiser_model = self.cruiser_model.to(self.device)
        self.cruiser_model = self.cruiser_model.eval().half()
        # self.cruiser_model = self.cruiser_model.float()
        # self.cruiser_model = self.cruiser_model.to(self.device, dtype=torch.float)
        # self.cruiser_model = self.cruiser_model.eval()

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension

    def preprocess(self, image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        # mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        # std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = PIL.Image.fromarray(image)
        # resize the cam captured image to (224, 224) for optimal resnet model inference
        image = image.resize((224, 224))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        # image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def execute_rc(self, current_image):
        # print('enter road cruising !')
        start_time = time.process_time()
        # global angle, angle_last
        # current_image = change['new']
        xy = self.cruiser_model(self.preprocess(current_image)).detach().float().cpu().numpy().flatten()
        x = xy[0]
        # y = (0.5 - xy[1]) / 2.0
        y = (1 + xy[1])

        self.x_slider = x.item()
        self.y_slider = y.item()

        self.speed = self.speed_gain

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)
        pid = self.angle * self.steering_gain + (self.angle - self.angle_last) * self.steering_dgain
        self.angle_last = self.angle

        self.steering = pid + self.steering_bias
        
        # print('steering : ', self.steering)
        # self.robot.left_motor.value = max(min(self.speed_gain + self.steering, 1.0), 0.0)
        # self.robot.right_motor.value = max(min(self.speed_gain - self.steering, 1.0), 0.0)
        # print('left motor value :', self.robot.left_motor.value)
        
        end_time = time.process_time()
        # self.execution_time.append(end_time - start_time + self.camera.cap_time)
        self.execution_time.append(end_time - start_time)

    # We accomplish that with the observe function.
    # def start_cruising(self):
        # self.execute({'new': self.camera.value})
        # self.camera.observe(self.execute, names='value')

    # def stop_cruising(self):
        # from jetbot.utils import plot_exec_time
        # self.camera.unobserve(self.execute, names='value')
        # self.camera.unobserve_all()
        # time.sleep(1.0)
        # self.robot.stop()
        # self.camera.stop()

        # plot exection time of road cruiser model processing
        # model_name = "road cruiser model"
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.cruiser_model_str)

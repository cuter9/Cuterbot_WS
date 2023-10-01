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


from jetbot import ObjectDetector

# from jetbot.object_detection_yolo import ObjectDetector_YOLO
model = ObjectDetector('ssd_mobilenet_v2_coco_onnx.engine')
# model = ObjectDetector_YOLO('yolov4-288.engine')


# Internally, the ``ObjectDetector`` class uses the TensorRT Python API to execute the engine that we provide.  It also takes care of preprocessing the input to the neural network, as
# well as parsing the detected objects.  Right now it will only work for engines created using the ``jetbot.ssd_tensorrt`` package. That package has the utilities for converting
# the model from the TensorFlow object detection API to an optimized TensorRT engine.
# 
# Next, let's initialize our camera.  Our detector takes 300x300 pixel input, so we'll set this when creating the camera.
# 
# > Internally, the Camera class uses GStreamer to take advantage of Jetson Nano's Image Signal Processor (ISP).  This is super fast and offloads
# > a lot of the resizing computation from the CPU. 

# In[ ]:


from jetbot import Camera

# camera = Camera.instance(width=300, height=300)
camera = Camera()

# Now, let's execute our network using some camera input.  By default the ``ObjectDetector`` class expects ``bgr8`` format that the camera produces.  However,
# you could override the default pre-processing function if your input is in a different format.

# In[ ]:


detections = model(camera.value)

print(detections)

# If there are any COCO objects in the camera's field of view, they should now be stored in the ``detections`` variable.

# ### Display detections in text area
# 
# We'll use the code below to print out the detected objects.

# In[ ]:


from IPython.display import display
import ipywidgets.widgets as widgets

detections_widget = widgets.Textarea()

detections_widget.value = str(detections)

display(detections_widget)

# You should see the label, confidence, and bounding box of each object detected in each image.  There's only one image (our camera) in this example.
# 
# 
# To print just the first object detected in the first image, we could call the following
# 
# > This may throw an error if no objects are detected

# In[ ]:


image_number = 0
object_number = 0

print(detections[image_number][object_number])

# ### Control robot to follow central object
# 
# Now we want our robot to follow an object of the specified class.  To do this we'll do the following
# 
# 1.  Detect objects matching the specified class
# 2.  Select object closest to center of camera's field of vision, this is the 'target' object
# 3.  Steer robot towards target object, otherwise wander
# 4.  If we're blocked by an obstacle, turn left
# 
# We'll also create some widgets that we'll use to control the target object label, the robot speed, and
# a "turn gain", that will control how fast the robot turns based off the distance between the target object
# and the center of the robot's field of view. 
# 
# 
# First, let's load our collision detection model.  The pre-trained model is stored in this directory as a convenience, but if you followed
# the collision avoidance example you may want to use that model if it's better tuned for your robot's environment.

# In[ ]:


import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np

collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# Great, now let's initialize our robot so we can control the motors.

# In[ ]:


from jetbot import Robot

robot = Robot()

# Finally, let's display all the control widgets and connect the network execution function to the camera updates.

# In[ ]:


from jetbot import bgr8_to_jpeg

blocked_widget = widgets.FloatSlider(min=0.0, max=1.0, value=0.0, description='blocked')
image_widget = widgets.Image(format='jpeg', width=300, height=300)
label_widget = widgets.IntText(value=6, description='tracked label')  # target to be tracked
speed_widget = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.01, description='speed')
turn_gain_widget = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.01, description='turn gain')

display(widgets.VBox([
    widgets.HBox([image_widget, blocked_widget]),
    label_widget,
    speed_widget,
    turn_gain_widget
]))

width = int(image_widget.width)
height = int(image_widget.height)


def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)


def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection


def execute(change):
    image = change['new']

    # print(image)
    # ** execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    blocked_widget.value = prob_blocked

    # turn left if blocked
    # if prob_blocked > 0.5:
    # robot.left(0.3)
    #    robot.left(0.05)
    #    image_widget.value = bgr8_to_jpeg(image)
    #    return

    # compute all detected objects
    detections = model(image)
    # print(detections)

    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == int(label_widget.value)]

    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    if det is not None:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

    # otherwise go forward if no target detected
    if det is None:
        robot.forward(float(speed_widget.value))

    # otherwise steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        robot.set_motors(
            float(speed_widget.value + turn_gain_widget.value * center[0]),
            float(speed_widget.value - turn_gain_widget.value * center[0])
        )

    # update image widget
    image_widget.value = bgr8_to_jpeg(image)


execute({'new': camera.value})

# Call the block below to connect the execute function to each camera frame update.

# In[ ]:


camera.unobserve_all()
camera.observe(execute, names='value')

# Awesome!  If the robot is not blocked you should see boxes drawn around the detected objects in blue.  The target object (which the robot follows) will be displayed in green.
# 
# The robot should steer towards the target when it is detected.  If it is blocked by an object it will simply turn left.
# 
# You can call the code block below to manually disconnect the processing from the camera and stop the robot.
# And then close the camera conneciton properly so that we can use the camera in other notebooks.

# In[ ]:


import time
from IPython.display import clear_output

out = widgets.Output()
button_stop = widgets.Button(description='Stop', tooltip='Click to stop running', icon='fa-stop')
display(button_stop, out)


def stop_run(b):
    with out:
        camera.unobserve_all()
        time.sleep(1.0)
        robot.stop()
        camera.stop()
        clear_output(wait=True)
        # get_ipython().run_line_magic('reset', '-f')


# button_stop.on_click(stop_run)

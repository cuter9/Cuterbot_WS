#!/usr/bin/env python
# coding: utf-8

# # Road Following - Data Collection (using Gamepad)
# 
# If you've run through the collision avoidance sample, your should be familiar following three steps
# 
# 1.  Data collection
# 2.  Training
# 3.  Deployment
# 
# In this notebook, we'll do the same exact thing!  Except, instead of classification, you'll learn a different fundamental technique, **regression**, that we'll use to
# enable JetBot to follow a road (or really, any path or target point).  
# 
# 1. Place the JetBot in different positions on a path (offset from center, different angles, etc)
# 
# >  Remember from collision avoidance, data variation is key!
# 
# 2. Display the live camera feed from the robot
# 3. Using a gamepad controller, place a 'green dot', which corresponds to the target direction we want the robot to travel, on the image.
# 4. Store the X, Y values of this green dot along with the image from the robot's camera
# 
# Then, in the training notebook, we'll train a neural network to predict the X, Y values of our label.  In the live demo, we'll use
# the predicted X, Y values to compute an approximate steering value (it's not 'exactly' an angle, as
# that would require image calibration, but it's roughly proportional to the angle so our controller will work fine).
# 
# So how do you decide exactly where to place the target for this example?  Here is a guide we think may help
# 
# 1.  Look at the live video feed from the camera
# 2.  Imagine the path that the robot should follow (try to approximate the distance it needs to avoid running off road etc.)
# 3.  Place the target as far along this path as it can go so that the robot could head straight to the target without 'running off' the road.
# 
# > For example, if we're on a very straight road, we could place it at the horizon.  If we're on a sharp turn, it may need to be placed closer to the robot so it doesn't run out of boundaries.
# 
# Assuming our deep learning model works as intended, these labeling guidelines should ensure the following:
# 
# 1.  The robot can safely travel directly towards the target (without going out of bounds etc.)
# 2.  The target will continuously progress along our imagined path
# 
# What we get, is a 'carrot on a stick' that moves along our desired trajectory.  Deep learning decides where to place the carrot, and JetBot just follows it :)

# ### Labeling example video
# 
# Execute the block of code to see an example of how to we labeled the images.  This model worked after only 123 images :)

# In[ ]:


from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/FW4En6LejhI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# ### Import Libraries

# So lets get started by importing all the required libraries for "data collection" purpose. We will mainly use OpenCV to visualize and save image with labels. Libraries such as uuid, datetime are used for image naming. 

# In[ ]:


# IPython Libraries for display and widgets
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display

# Camera and Motor Interface for JetBot
from jetbot import Robot, Camera, bgr8_to_jpeg

# Basic Python packages for image annotation
from uuid import uuid1
import os
import json
import glob
import datetime
import numpy as np
import cv2
import time


# ### Display Live Camera Feed

# First, let's initialize and display our camera like we did in the teleoperation notebook. 
# 
# We use Camera Class from JetBot to enable CSI MIPI camera. Our neural network takes a 224x224 pixel image as input. We'll set our camera to that size to minimize the filesize of our dataset (we've tested that it works for this task). In some scenarios it may be better to collect data in a larger image size and downscale to the desired size later.

# In[ ]:


camera = Camera()

widget_width = camera.width
widget_height = camera.height

image_widget = widgets.Image(format='jpeg', width=widget_width, height=widget_height)
target_widget = widgets.Image(format='jpeg', width=widget_width, height=widget_height)

x_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='x')
y_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='y')

def display_xy(camera_image):
    image = np.copy(camera_image)
    x = x_slider.value
    y = y_slider.value
    x = int(x * widget_width / 2 + widget_width / 2)
    y = int(y * widget_height / 2 + widget_height / 2)
    image = cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
    image = cv2.circle(image, (widget_width / 2, widget_height), 8, (0, 0,255), 3)
    image = cv2.line(image, (x,y), (widget_width / 2, widget_height), (255,0,0), 3)
    jpeg_image = bgr8_to_jpeg(image)
    return jpeg_image

time.sleep(1)
traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
traitlets.dlink((camera, 'value'), (target_widget, 'value'), transform=display_xy)

display(widgets.HBox([image_widget, target_widget]), x_slider, y_slider)


# ### Create Gamepad Controller
# 
# This step is similar to "Teleoperation" task. In this task, we will use gamepad controller to label images.
# 
# The first thing we want to do is create an instance of the Controller widget, which we'll use to label images with "x" and "y" values as mentioned in introduction. The Controller widget takes a index parameter, which specifies the number of the controller. This is useful in case you have multiple controllers attached, or some gamepads appear as multiple controllers. To determine the index of the controller you're using,
# 
# Visit http://html5gamepad.com.
# Press buttons on the gamepad you're using
# Remember the index of the gamepad that is responding to the button presses
# Next, we'll create and display our controller using that index.

# In[ ]:


controller = widgets.Controller(index=0)

display(controller)


# ### Connect Gamepad Controller to Label Images
# 
# Now, even though we've connected our gamepad, we haven't yet attached the controller to label images! We'll connect that to the left and right vertical axes using the dlink function. The dlink function, unlike the link function, allows us to attach a transform between the source and target. 

# In[ ]:


widgets.jsdlink((controller.axes[2], 'value'), (x_slider, 'value'))
widgets.jsdlink((controller.axes[3], 'value'), (y_slider, 'value'))


# ### Collect data
# 
# The following block of code will display the live image feed, as well as the number of images we've saved.  We store
# the target X, Y values by
# 
# 1. Place the green dot on the target
# 2. Press 'down' on the DPAD to save
# 
# This will store a file in the ``dataset_xy_old`` folder with files named
# 
# ``xy_<x value>_<y value>_<uuid>.jpg``
# 
# where `<x value>` and `<y value>` are the coordinates **in pixel (not in percentage)** (count from the top left corner).
# 
# When we train, we load the images and parse the x, y values from the filename

# In[ ]:


DATASET_DIR = 'dataset_xy_old'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directories not created because they already exist')

for b in controller.buttons:
    b.unobserve_all()

count_widget = widgets.IntText(description='count', value=len(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))

def xy_uuid(x, y):
    return 'xy_%03d_%03d_%s' % (x * widget_width / 2 + widget_width / 2, y * widget_height / 2 + widget_height / 2, uuid1())

def save_snapshot(change):
    if change['new']:
        uuid = xy_uuid(x_slider.value, y_slider.value)
        image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_widget.value)
        count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

controller.buttons[13].observe(save_snapshot, names='value')

display(widgets.VBox([
    target_widget,
    count_widget
]))


# Again, let's close the camera conneciton properly so that we can use the camera in other notebooks.

# In[ ]:


camera.stop()


# ### Next

# Once you've collected enough data, we'll need to copy that data to our GPU desktop or cloud machine for training. First, we can call the following terminal command to compress our dataset folder into a single zip file.  
# 
# > If you're training on the JetBot itself, you can skip this step!

# The ! prefix indicates that we want to run the cell as a shell (or terminal) command.
# 
# The -r flag in the zip command below indicates recursive so that we include all nested files, the -q flag indicates quiet so that the zip command doesn't print any output

# In[ ]:


def timestr():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

get_ipython().system('zip -r -q road_following_{DATASET_DIR}_{timestr()}.zip {DATASET_DIR}')


# You should see a file named road_following_<Date&Time>.zip in the Jupyter Lab file browser. You should download the zip file using the Jupyter Lab file browser by right clicking and selecting Download.

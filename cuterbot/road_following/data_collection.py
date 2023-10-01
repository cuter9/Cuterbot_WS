#!/usr/bin/env python
# coding: utf-8

# # Road Following 
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
import ipywidgets
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


# ### Data Collection

# Let's display our camera like we did in the teleoperation notebook, however this time with using a special ipywidget called `jupyter_clickable_image_widget` that lets you click on the image and take the coordinates for data annotation.
# This eliminates the needs of using the gamepad for data annotation.
# 
# We use Camera Class from JetBot to enable CSI MIPI camera. Our neural network takes a 224x224 pixel image as input. We'll set our camera to that size to minimize the filesize of our dataset (we've tested that it works for this task). In some scenarios it may be better to collect data in a larger image size and downscale to the desired size later.
# 
# The following block of code will display the live image feed for you to click on for annotation on the left, as well as the snapshot of last annotated image (with a green circle showing where you clicked) on the right.
# Below it shows the number of images we've saved.  
# 
# When you click on the left live image, it stores a file in the ``dataset_xy_old`` folder with files named
# 
# ``xy_<x value>_<y value>_<uuid>.jpg``
# 
# When we train, we load the images and parse the x, y values from the filename.
# Here `<x value>` and `<y value>` are the coordinates **in pixels** (count from the top left corner).
# 
# 

# In[ ]:


from jupyter_clickable_image_widget import ClickableImageWidget

DATASET_DIR = 'dataset_xy_old'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directories not created because they already exist')

camera = Camera()

# create image preview
camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

# create widgets
count_widget = ipywidgets.IntText(description='count')
# manually update counts at initialization
count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

def save_snapshot(_, content, msg):
    if content['event'] == 'click':
        data = content['eventData']
        x = data['offsetX']
        y = data['offsetY']
        
        # save to disk
        #dataset.save_entry(category_widget.value, camera.value, x, y)
        uuid = 'xy_%03d_%03d_%s' % (x, y, uuid1())
        image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(camera_widget.value)
        
        # display saved snapshot
        snapshot = camera.value.copy()
        snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
        snapshot_widget.value = bgr8_to_jpeg(snapshot)
        count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))
        
camera_widget.on_msg(save_snapshot)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget, snapshot_widget]),
    count_widget
])

display(data_collection_widget)


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

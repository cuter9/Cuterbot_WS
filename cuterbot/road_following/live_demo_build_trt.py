#!/usr/bin/env python
# coding: utf-8
# # Road Following - Build TensorRT model for live demo
# In this notebook, we will optimize the model we trained using TensorRT.

# ## Load the trained model We will assume that you have already downloaded ``best_steering_model_xy_resnet34.pth`` to work
# station as instructed in "train_model.ipynb" notebook. Now, you should upload model file to JetBot in to this
# notebook's directory. Once that's finished there should be a file named ``best_steering_model_xy_resnet34.pth`` in this
# notebook's directory. > Please make sure the file has uploaded fully before calling the next cell

# Execute the code below to initialize the PyTorch model. This should look very familiar from the training notebook.


import torchvision
import torch
from torch2trt import torch2trt

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()

# Next, load the trained weights from the ``best_steering_model_xy_resnet34.pth`` file that you uploaded.
model.load_state_dict(torch.load('best_steering_model_xy_resnet34.pth'))

# Currently, the model weights are located on the CPU memory execute the code below to transfer to the GPU device.

device = torch.device('cuda')

# ## TensorRT

# > If your setup does not have `torch2trt` installed, you need to first install `torch2trt` by executing the
# following in the console.
# ```
# bash cd $HOME
# git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# cd torch2trt
# sudo
# python3 setup.py install
# ```
# Convert and optimize the model using torch2trt for faster inference with TensorRT. Please see the [torch2trt](
# https://github.com/NVIDIA-AI-IOT/torch2trt) readme for more details.
# 
# > This optimization process can take a couple minutes to complete.


data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

# Save the optimized model using the cell below
torch.save(model_trt.state_dict(), 'best_steering_model_xy_trt.pth')

# ## Next
# Open live_demo_trt.ipynb to move JetBot with the TensorRT optimized model.

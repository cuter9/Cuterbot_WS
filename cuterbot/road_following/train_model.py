#!/usr/bin/env python
# coding: utf-8
# # Road Follower - Train Model
# 
# In this notebook we will train a neural network to take an input image, and output a set of x,
# y values corresponding to a target.
# We will be using PyTorch deep learning framework to train ResNet18 neural network architecture model for road
# follower application.


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
from zipfile import ZipFile
import time

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

WORK = os.getcwd()

DIR_DATA_REPO_PROJECT = "D:\\AI_Lecture_Demos\\Data_Repo\\Cuterbot\\"
os.makedirs(DIR_DATA_REPO_PROJECT, exist_ok=True)

DIR_DATA_REPO_THIS = os.path.join(DIR_DATA_REPO_PROJECT, "road_following")
os.makedirs(DIR_DATA_REPO_THIS, exist_ok=True)

TRAIN_MODEL = "resnet50"  # resnet18, resnet34, resnet50, resnet101, mobilenet_v2, vgg11, mobilenet_v3_large
# *** refererence : https://pytorch.org/docs/stable/optim.html#algorithms
# use the following learning algorithms for evaluation
TRAIN_MATHOD = "Adam"  # "Adam", "SGD", "ASGD", "Adadelta", "RAdam"; the parameters lr=0.01, momentum=0.92  may be needed

# ### Download and extract data
# 
# Before you start, you should upload the ``road_following_<Date&Time>.zip`` file that you created in the
# ``data_collection.ipynb`` notebook on the robot.
# 
# > If you're training on the JetBot you collected data on, you can skip this!
# 
# You should then extract this dataset by calling the command below:

DATA_FILE = "dataset_xy_0916_1"
ZipFile(os.path.join(DIR_DATA_REPO_THIS, DATA_FILE + '.zip')).extractall(path=DIR_DATA_REPO_THIS)


# You should see a folder named ``dataset_all`` appear in the file browser.

# ### Create Dataset Instance
# 
# Here we create a custom ``torch.utils.data.Dataset`` implementation, which implements the ``__len__`` and
# ``__getitem__`` functions.  This class is responsible for loading images and parsing the x, y values from the image
# filenames.  Because we implement the ``torch.utils.data.Dataset`` class, we can use all of the torch data utilities :)
# 
# We hard coded some transformations (like color jitter) into our dataset.  We made random horizontal flips optional
# (in case you want to follow a non-symmetric path, like a road where we need to 'stay right').  If it doesn't matter
# whether your robot follows some convention, you could enable flips to augment the dataset.


def get_x(path, width):
    """Gets the x value from the image filename"""
    return (float(int(path.split("_")[1])) - width / 2) / (width / 2)


def get_y(path, height):
    """Gets the y value from the image filename"""
    return (float(int(path.split("_")[2])) - height / 2) / (height / 2)
    # return height - float(int(path.split("_")[2]))


class XYDataset(torch.utils.data.Dataset):

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        width, height = image.size
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))

        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()


# dataset = XYDataset('dataset_xy', random_hflips=False)
dataset = XYDataset(os.path.join(DIR_DATA_REPO_THIS, 'dataset_xy'), random_hflips=False)

# ### Split dataset into train and test sets Once we read dataset, we will split data set in train and test sets. In
# this example we split train and test a 90%-10%. The test set will be used to verify the accuracy of the model we
# train.


test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

# ### Create data loaders to load data in batches
# 
# We use ``DataLoader`` class to load data in batches, shuffle data and allow using multi-subprocesses. In this
# example we use batch size of 64. Batch size will be based on memory available with your GPU and it can impact
# accuracy of the model.


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

# ### Define Neural Network Model
# 
# We use ResNet-18 model and other variants available on PyTorch TorchVision.
# 
# In a process called transfer learning, we can repurpose a pre-trained model (trained on millions of images) for a
# new task that has possibly much less data available.
#
# More details on ResNet-18 and other variant :
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# 
# More Details on Transfer Learning: https://www.youtube.com/watch?v=yofjFQddwHE 

model = getattr(models, TRAIN_MODEL)()
# model_attr = getattr(models, TRAIN_MODEL)
# model = model_attr()

optimizer = getattr(optim, TRAIN_MATHOD)(model.parameters(), weight_decay=0)
# optimizer = getattr(optim, TRAIN_MATHOD)(model.parameters(), lr=0.01, momentum=0.95)
# optimizer_attr = getattr(optim, TRAIN_METHOD)
# optimizer = optimizer_attr(model.parameters())


BEST_MODEL_PATH = os.path.join(DIR_DATA_REPO_THIS, "best_steering_model_xy_" + TRAIN_MODEL + ".pth")
# BEST_MODEL_PATH = os.path.join(DIR_DATA_REPO_THIS, "best_steering_model_xy_tmp.pth")


# ResNet model has fully connected (fc) final layer with 512 as ``in_features`` and we will be training for regression
# thus ``out_features`` as 2
# 
# Finally, we transfer our model for execution on the GPU

print("torch cuda version : ", torch.version.cuda)
print("cuda is available for pytorch: ", torch.cuda.is_available())

# ----- modify last layer for classification, and the model used in notebook should be modified too.
# model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
#                                      2)  # for mobilenet_v3 model. must add the block expansion factor 4
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # for mobilenet_v2 model. must add the block expansion factor 4
# model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)  # for VGG model. must add the block expansion factor 4
# model.fc = torch.nn.Linear(512, 2)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # for resnet model must add the block expansion factor 4

# ** you may use cpu for training
device = torch.device('cuda')
# device = torch.device('cpu')
model = model.float()
model = model.to(device, dtype=torch.float)

# ### Train Regression:
# We train for 70 epochs and save best model if the loss is reduced.

fig_1, ax_1 = plt.subplots(figsize=(8, 4))
font = {'weight': 'normal', 'size': 10}


def plot_loss(loss_data, best_loss):
    plt.cla()
    epochs = range(1, len(loss_data) + 1)
    ld_0 = [ld[0] for ld in loss_data]
    ld_1 = [ld[1] for ld in loss_data]
    ax_1.semilogy(epochs, ld_0, "r-", linewidth=1.0, label="Training Loss:{:.4f}".format(ld_0[-1]))
    ax_1.semilogy(epochs, ld_1, 'bs--', linewidth=1.0, label="Test Loss:{:.4f}".format(ld_1[-1]))
    xlim = epochs[-1] + 2
    ax_1.set_xlim(0, xlim)
    plt.legend()
    plt.title("Training convergence plot (Model : {:s}, Training Method : {:s}) \n current best test loss : {:.4f}". \
              format(TRAIN_MODEL, TRAIN_MATHOD, best_loss))
    plt.xlabel('epoch', fontdict=font)
    plt.ylabel('loss', fontdict=font)
    fig_1.canvas.draw()
    fig_1.canvas.flush_events()


NUM_EPOCHS = 200
best_loss = 1e9

loss_data = []
lt_epoch = []  # learning time per epoch
lt_sample = []  # learning time per epoch
for epoch in range(NUM_EPOCHS):
    start_epoch = time.process_time()
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        start_sample = time.process_time()
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
        end_sample = time.process_time()
        lt_sample.append(end_sample - start_sample)
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)

    end_epoch = time.process_time()
    lt_epoch.append(end_epoch - start_epoch)

    if test_loss <= best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss

    loss_data.append([train_loss, test_loss])
    plot_loss(loss_data, best_loss)
    print('Training performance at %ith epoch --- training lose: %f, test loss: %f'
          % (epoch + 1, train_loss, test_loss))

profile_plot = os.path.join(DIR_DATA_REPO_THIS, "Training_convergence_plot_Model_{:s}_Training_Method_{:s})".
                            format(TRAIN_MODEL, TRAIN_MATHOD))
fig_1.savefig(profile_plot)

# ----- training time statistics in terms of epoch
learning_time_epoch = np.array(lt_epoch)
mean_lt_epoch = np.mean(learning_time_epoch)
max_lt_epoch = np.amax(learning_time_epoch)
min_lt_epoch = np.amax(learning_time_epoch)
print(
    "mean learning time per epoch: {:.3f} s, maximum epoch learning time: {:.3f} s, minimum epoch learning time: {:.3f} s".
    format(mean_lt_epoch, max_lt_epoch, min_lt_epoch))

# ----- training time statistics in terms of sample
learning_time_sample = np.array(lt_sample)
mean_lt_sample = np.mean(learning_time_sample)
max_lt_sample = np.amax(learning_time_sample)
min_lt_sample = np.amax(learning_time_sample)
print(
    "mean learning time per sample: {:.3f} s, maximum sample learning time: {:.3f} s, minimum sample learning time: {:.3f} s".
    format(mean_lt_sample, max_lt_sample, min_lt_sample))

fig_2, axh = plt.subplots(1, 2, figsize=(10, 4))
fig_2.suptitle("Training Time Statistics")
axh[0].set_ylabel('no. of epoch')
axh[0].set_xlabel('time of a epoch training, sec.')
axh[0].hist(learning_time_epoch)
axh[1].set_ylabel('no. of sample')
axh[1].set_xlabel('time for a sample training, sec.')
axh[1].hist(learning_time_sample, bins=(0.01 * np.array(list(range(101)))).tolist())
training_time_file = os.path.join(DIR_DATA_REPO_THIS, "Training_time_Model_{:s}_Training_Method_{:s})".
                                  format(TRAIN_MODEL, TRAIN_MATHOD))
fig_2.savefig(training_time_file)

# Once the model is trained, it will generate ``best_steering_model_xy_resnet34.pth`` file which you can use for
# inference in the live demo notebook.
# 
# If you trained on a different machine other than JetBot, you'll need to upload this to the JetBot to the
# ``road_following`` example folder.

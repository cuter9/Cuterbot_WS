import os
import time

import PIL.Image
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import traitlets
# import torchvision.models as models

from jetbot import Camera
from jetbot import Robot


class RoadCruiser(traitlets.HasTraits):
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

        self.camera = Camera()
        self.robot = Robot()
        self.angle = 0.0
        self.angle_last = 0.0
        self.execution_time = []
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
        # model = model.to(device)
        # model = model.eval().half()
        self.cruiser_model = self.cruiser_model.float()
        self.cruiser_model = self.cruiser_model.to(self.device, dtype=torch.float)
        self.cruiser_model = self.cruiser_model.eval()

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension

    def preprocess(self, image):
        # mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        # std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = PIL.Image.fromarray(image)
        # image = transforms.functional.to_tensor(image).to(device).half()
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def execute(self, change):
        start_time = time.process_time()
        # global angle, angle_last
        image = change['new']
        xy = self.cruiser_model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
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

        self.robot.left_motor.value = max(min(self.speed_gain + self.steering, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain - self.steering, 1.0), 0.0)
        end_time = time.process_time()
        self.execution_time.append(end_time - start_time + self.camera.cap_time)

    # We accomplish that with the observe function.
    def start_cruising(self):
        # self.execute({'new': self.camera.value})
        self.camera.observe(self.execute, names='value')

    def stop_cruising(self, b):
        # os.environ['DISPLAY'] = ':10.0'
        # self.camera.unobserve(self.execute, names='value')
        self.camera.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.camera.stop()

        execute_time = np.array(self.execution_time[1:])
        mean_execute_time = np.mean(execute_time)
        max_execute_time = np.amax(execute_time)
        min_execute_time = np.amin(execute_time)

        print(
            "Mean execution time of model : %f \nMax execution time of model : %f \nMin execution time of model : %f " \
            % (mean_execute_time, max_execute_time, min_execute_time))
        plt.hist(execute_time, bins=(0.005 * np.array(list(range(101)))).tolist())
        plt.show()

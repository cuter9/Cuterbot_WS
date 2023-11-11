import time

import PIL.Image

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import traitlets
# import torchvision.models as models
from torch2trt import TRTModule

from jetbot import Camera
from jetbot import Robot


class RoadCruiserTRT(traitlets.HasTraits):
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
        # self.cruiser_model = getattr(torchvision.models, cruiser_model)(pretrained=False)
        self.type_cruiser_model = type_cruiser_model
        self.cruiser_model = TRTModule()
        self.cruiser_model.load_state_dict(torch.load(''.join(['best_steering_model_xy_trt_', cruiser_model, '.pth'])))
        self.camera = Camera()
        self.robot = Robot.instance()
        self.angle = 0.0
        self.angle_last = 0.0
        self.execution_time = []
        # self.fps = []
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
        # self.cruiser_model = self.cruiser_model.to(self.device)
        # self.cruiser_model = self.cruiser_model.eval().half()
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
        # self.execution_time.append(end_time - start_time + self.camera.cap_time)
        self.execution_time.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

    # We accomplish that with the observe function.
    def start_cruising(self):
        # self.execute({'new': self.camera.value})
        self.camera.observe(self.execute, names='value')

    def stop_cruising(self, b):
        import matplotlib.pyplot as plt
        from jetbot.utils import plot_exec_time
        # self.camera.unobserve(self.execute, names='value')
        self.camera.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.camera.stop()

        # plot exection time of road cruiser model processing
        model_name = 'road cruiser model'
        plot_exec_time(self.execution_time[1:], model_name, self.cruiser_model_str)
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.cruiser_model_str)
        plt.show()
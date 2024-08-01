import time

import PIL.Image

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import traitlets
from traitlets import HasTraits, Float, Unicode
import torchvision.models as pth_models

from jetbot import Camera
from jetbot import Robot


def load_tune_pth_model(pth_model_name="resnet18", pretrained=True):
    if pretrained:
        model = getattr(pth_models, pth_model_name)()       # for fine tuning
    else:
        model = getattr(pth_models, pth_model_name)(pretrained=False)   # for inferencig
    # ----- modify last layer for classification, and the model used in notebook should be modified too.

    if pth_model_name == 'mobilenet_v3_large':  # MobileNet
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
                                              2)  # for mobilenet_v3 model. must add the block expansion factor 4

    elif pth_model_name == 'mobilenet_v2':
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features,
                                              2)  # for mobilenet_v2 model. must add the block expansion factor 4

    elif pth_model_name == 'vgg11':  # VGGNet
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,
                                              2)  # for VGG model. must add the block expansion factor 4

    elif 'resnet' in pth_model_name:  # ResNet
        model.fc = torch.nn.Linear(model.fc.in_features,
                                   2)  # for resnet model must add the block expansion factor 4
        # model.fc = torch.nn.Linear(512, 2)

    elif 'efficientnet' in pth_model_name:  # ResNet
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # for efficientnet model

    elif pth_model_name == 'inception_v3':  # Inception_v3
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        if model.aux_logits:
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 2)

    return model


class RoadCruiser(HasTraits):
    cruiser_model = Unicode(default_value='').tag(config=True)
    type_cruiser_model = Unicode(default_value='').tag(config=True)
    speed_rc = Float(default_value=0).tag(config=True)
    speed_gain_rc = Float(default_value=0.15).tag(config=True)
    steering_gain_rc = Float(default_value=0.08).tag(config=True)
    steering_dgain_rc = Float(default_value=1.5).tag(config=True)
    steering_bias_rc = Float(default_value=0.0).tag(config=True)
    steering_rc = Float(default_value=0.0).tag(config=True)
    x_slider = Float(default_value=0).tag(config=True)
    y_slider = Float(default_value=0).tag(config=True)
    use_gpu = Unicode(default_value='gpu').tag(config=True)

    def __init__(self, init_sensor_rc=False):
        super().__init__()

        self.cruiser_model_pth = None

        if init_sensor_rc:
            self.capturer = Camera()
            self.robot = Robot.instance()
        # self.robot = Robot()
        self.angle = 0.0
        self.angle_last = 0.0
        # self.fps = []
        self.x_slider = 0
        self.y_slider = 0

        self.execution_time_rc = []
        self.observe(self.select_gpu, names=['use_gpu'])
        self.device = None

    def load_road_cruiser(self, change):
        # The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
        self.cruiser_model_pth = None
        pth_model_name = self.cruiser_model.split('/')[-1].split('.')[0].split('_', 4)[-1].split('-')[0]
        print('pytorch model name: %s' % pth_model_name)
        self.cruiser_model_pth = load_tune_pth_model(pth_model_name=pth_model_name, pretrained=False)

        print('path of cruiser model: %s' % self.cruiser_model)
        print('use %s' % self.use_gpu)
        # self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_' + cruiser_model + '.pth'))
        self.cruiser_model_pth.load_state_dict(torch.load(self.cruiser_model))

        if self.use_gpu == 'gpu':
            print("torch cuda version : ", torch.version.cuda)
            print("cuda is available for pytorch: ", torch.cuda.is_available())
            self.device = torch.device('cuda')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval().half()

        elif self.use_gpu == 'cpu':
            self.device = torch.device('cpu')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval()

        # self.cruiser_model = self.cruiser_model.float()
        # self.cruiser_model = self.cruiser_model.to(self.device, dtype=torch.float)
        # self.cruiser_model = self.cruiser_model.eval()

    def select_gpu(self, change):
        self.use_gpu = change['new']

    # ---- Creating the Pre-Processing Function
    # 1. Convert from HWC layout to CHW layout
    # 2. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
    # 3. Transfer the data from CPU memory to GPU memory
    # 4. Add a batch dimension

    def preprocess_rc(self, image):
        mean = None
        std = None
        if self.use_gpu == 'gpu':
            mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
            std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        elif self.use_gpu == 'cpu':
            mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        # mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        # std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = PIL.Image.fromarray(image)
        # resize the cam captured image to (224, 224) for optimal resnet model inference
        if self.type_cruiser_model == 'inception':
            image = image.resize((299, 299))
        else:
            image = image.resize((224, 224))

        if self.use_gpu == 'gpu':
            image = transforms.functional.to_tensor(image).to(self.device).half()
        elif self.use_gpu == 'cpu':
            image = transforms.functional.to_tensor(image).to(self.device)

        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def execute_rc(self, change):
        start_time = time.time()
        # global angle, angle_last
        image = change['new']
        xy = self.cruiser_model_pth(self.preprocess_rc(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]
        # y = (0.5 - xy[1]) / 2.0
        y = (1 + xy[1])

        self.x_slider = x.item()
        self.y_slider = y.item()

        self.speed_rc = self.speed_gain_rc

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)
        pid = self.angle * self.steering_gain_rc + (self.angle - self.angle_last) * self.steering_dgain_rc
        self.angle_last = self.angle

        self.steering = pid + self.steering_bias_rc

        self.robot.left_motor.value = max(min(self.speed_gain_rc + self.steering, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain_rc - self.steering, 1.0), 0.0)

        end_time = time.time()
        # self.execution_time.append(end_time - start_time + self.camera.cap_time)
        self.execution_time_rc.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

    # We accomplish that with the observe function.
    def start_rc(self, change):
        # self.execute({'new': self.camera.value})
        self.load_road_cruiser(change)
        self.capturer.observe(self.execute_rc, names='value')

    def stop_rc(self, change):
        from jetbot.utils import plot_exec_time
        # self.camera.unobserve(self.execute, names='value')
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # plot execution time of road cruiser model processing
        model_name = "road cruiser model"
        cruiser_model_str = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], model_name, cruiser_model_str)
        # plot_exec_time(self.execution_time[1:], self.fps[1:], model_name, self.cruiser_model_str)
        # plt.show()

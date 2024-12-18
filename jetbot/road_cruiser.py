import time

import PIL.Image

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from traitlets import HasTraits, Float, Unicode

from jetbot import Camera
from jetbot import Robot
from jetbot.utils.model_selection import load_tune_pth_model, tv_classifier_preprocess


class RoadCruiser(HasTraits):
    cruiser_model = Unicode(default_value='').tag(config=True)
    type_cruiser_model = Unicode(default_value='').tag(config=True)
    cruiser_model_preprocess = Unicode(default_value='').tag(config=True)
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

        self.cruiser_model_type_pth = None
        self.cruiser_model_pth = None
        self.preprocess = None
        self.cruiser_model_preprocess_pth = None

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
        self.cruiser_model_type_pth = None

        pth_model_name = self.cruiser_model.split('/')[-1].split('.')[0].split('_', 4)[-1].split('-')[0]
        print('pytorch model name: %s' % pth_model_name)
        self.cruiser_model_pth, self.cruiser_model_type_pth, self.cruiser_model_preprocess_pth = load_tune_pth_model(
            pth_model_name=pth_model_name,
            pretrained=False)

        print('path of cruiser model: %s' % self.cruiser_model)
        print('use %s for inference.' % self.use_gpu)
        # self.cruiser_model.load_state_dict(torch.load('best_steering_model_xy_' + cruiser_model + '.pth'))
        self.cruiser_model_pth.load_state_dict(torch.load(self.cruiser_model))

        # load preprocess for loaded cruiser model
        if self.cruiser_model_preprocess_pth is None:  # load pre-stored preprocess module of the trained model
            self.preprocess = tv_classifier_preprocess()
            # use weights_only=True, ref: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
            self.preprocess.load_state_dict(torch.load(self.cruiser_model_preprocess))
        else:  # used the preprocess from load_tune_pth_model
            self.preprocess = self.cruiser_model_preprocess_pth[0]

        if self.use_gpu == 'gpu':
            print("torch cuda version : ", torch.version.cuda)
            print("cuda is available for pytorch: ", torch.cuda.is_available())
            self.device = torch.device('cuda')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval().half()
            self.preprocess.to(self.device)
            self.preprocess.eval().half()

        elif self.use_gpu == 'cpu':
            self.device = torch.device('cpu')
            self.cruiser_model_pth.to(self.device)
            self.cruiser_model_pth.eval()
            self.preprocess.to(self.device)
            self.preprocess.eval()

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
        # tv = int(torchvision.__version__.split(".")[1])  # torchvision version
        image = PIL.Image.fromarray(image)

        '''
        if tv >= 13:
            preprocess = self.cruiser_model_preprocess[0]
        else:
            # load preprocess for loaded cruiser model
            preprocess = tv_classifier_preprocess()
            preprocess.load_state_dict(torch.load(self.cruiser_model_preprocess))
        '''

        if self.use_gpu == 'gpu':
            image = self.preprocess(image).to(self.device).half()
        elif self.use_gpu == 'cpu':
            image = self.preprocess(image).to(self.device)

        '''           
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
        '''
        return image[None, ...]

    def preprocess_rc_1(self, image):
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
        # y = (0.5 - xy[1]) / 2.0  # This is suitable for the image window without referring to central line
        y = (1 - xy[1])     # This is suitable for the y data around 0, i.e. the central line is the image window

        self.x_slider = x.item()
        self.y_slider = y.item()

        self.speed_rc = self.speed_gain_rc

        # angle = np.sqrt(xy)*np.arctan2(x, y)
        angle_1 = np.arctan2(x, y)
        self.angle = 0.5 * np.pi * np.tanh(0.5 * angle_1)
        pid = self.angle * self.steering_gain_rc + (self.angle - self.angle_last) * self.steering_dgain_rc
        self.angle_last = self.angle

        self.steering_rc = pid + self.steering_bias_rc

        self.robot.left_motor.value = max(min(self.speed_gain_rc + self.steering_rc, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed_gain_rc - self.steering_rc, 1.0), 0.0)

        end_time = time.time()
        self.execution_time_rc.append(end_time - start_time)
        # self.fps.append(1/(end_time - start_time))

    # We accomplish that with the observe function.
    def start_rc(self, change):
        # self.execute({'new': self.camera.value})
        self.load_road_cruiser(change)
        print("start running!")
        self.capturer.observe(self.execute_rc, names='value')

    def stop_rc(self, change):
        from jetbot.utils import plot_exec_time
        print("start stopping!")
        # self.camera.unobserve(self.execute, names='value')
        self.capturer.unobserve_all()
        time.sleep(1.0)
        self.robot.stop()
        self.capturer.stop()

        # plot execution time of road cruiser model processing
        model_name = "road cruiser model"
        cruiser_model_str = self.cruiser_model.split("/")[-1].split('.')[0]
        plot_exec_time(self.execution_time_rc[1:], model_name, cruiser_model_str)
        # plt.show()

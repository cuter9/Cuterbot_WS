import pandas as pd
import os
from traitlets import HasTraits, Unicode, List, Bool
# import numpy as np
from typing import Optional, Tuple

import torch
from torch import Tensor
import torchvision
import torchvision.models as pth_models
from torchvision.transforms import functional as F, InterpolationMode

HEAD_LIST = ['model_function', 'model_type', 'model_path', 'preprocess_path']
MODEL_REPO_DIR = os.path.join(os.environ["HOME"], "model_repo")
MODEL_REPO_DIR_DOCKER = os.path.join("/workspace", "model_repo")
os.environ['MODEL_REPO_DIR_DOCKER'] = MODEL_REPO_DIR_DOCKER
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


class tv_classifier_preprocess(torch.nn.Module):
    # import weights transform function from torchvision v0.19
    def __init__(
            self,
            *,
            crop_size: int = 224,
            resize_size: int = 256,
            mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
            std: Tuple[float, ...] = (0.229, 0.224, 0.225),
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            antialias: Optional[bool] = True,
            tv_version=None,
            tv_weights=None,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias
        self.tv_version = tv_version
        self.tv_tv_weights = tv_weights

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img


def load_pth_model(pth_model_name, weights_cls, pretrained):
    preprocess_wrap = None

    # for fine-tuning
    if weights_cls:
        try:
            weights = getattr(pth_models, weights_cls).DEFAULT
            preprocess = weights.transforms()
            classifier_preprocess = tv_classifier_preprocess(crop_size=preprocess.crop_size,
                                                             resize_size=preprocess.resize_size,
                                                             mean=preprocess.mean,
                                                             std=preprocess.std,
                                                             interpolation=preprocess.interpolation,
                                                             antialias=preprocess.antialias,
                                                             tv_version=torchvision.__version__,
                                                             tv_weights=weights_cls
                                                             )
            preprocess_wrap = [preprocess, classifier_preprocess]
        except AttributeError as err:
            print("Attribute Error - %s \n" % err, ', Check weights class ( %s ) is correct or not!' % weights_cls)

        if pretrained:
            model = getattr(pth_models, pth_model_name)(weights=weights, aux_logits=True) \
                if pth_model_name in ['googlenet', 'inception_v3'] \
                else getattr(pth_models, pth_model_name)(weights=weights) # for fine-tuning
        else:
            model = getattr(pth_models, pth_model_name)(weights=None, aux_logits=True) \
                if pth_model_name in ['googlenet', 'inception_v3'] \
                else getattr(pth_models, pth_model_name)(weights=None) # for fine-tuning

    # for inferencing
    else:
        model = getattr(pth_models, pth_model_name)(pretrained=pretrained, aux_logits=True) \
            if pth_model_name in ['googlenet', 'inception_v3'] \
            else getattr(pth_models, pth_model_name)(pretrained=pretrained)
        print("The  model is load from torchvision with version less then 0.13. \n"
              "The preprocess for the loaded model should be re-designed if it is loaded with pretrained weights, or \n "
              "The preprocess can be loaded from the pre-stored preprocess module while training the model "
              "with torchvision version >= 0.13 (it is recommended!)")

    return model, preprocess_wrap


def load_tune_pth_model(pth_model_name="resnet18", pretrained=True):
    """
    if pretrained:
        model = getattr(pth_models, pth_model_name)()  # for fine-tuning
    else:
        model = getattr(pth_models, pth_model_name)(pretrained=False)  # for inferencing
    """
    preprocess_wrap = None
    model_type = None
    model = None
    weights_cls = None

    tv = int(torchvision.__version__.split(".")[1])  # torchvision version
    # ----- modify the last layer for classification, and the model used in notebook should be modified too.
    if 'resnet' in pth_model_name:  # resnet18, resnet34, resnet50, resnet101, ...
        model_type = "ResNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = pth_model_name.replace("resnet", "ResNet") + "_Weights"

        model, preprocess_wrap = load_pth_model(pth_model_name, weights_cls, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features,
                                   2)  # for resnet model must add block expansion factor 4

    elif 'mobilenet_v3' in pth_model_name:  # 'mobilenet_v3_large' or  'mobilenet_v3_small'
        model_type = "MobileNet"
        # enter the code to convert pytorch 'mobilenet_v3' model so that can be used in Jetbot application.

    elif pth_model_name == 'mobilenet_v2':      # mobilenet_v2
        model_type = "MobileNet"
        # enter the code to convert pytorch 'mobilenet_v2' model so that can be used in Jetbot application.

    elif pth_model_name == 'vgg11':  # VGGNet
        model_type = "VggNet"
        # enter the code to convert pytorch 'vgg11' model so that can be used in Jetbot application.

    elif 'efficientnet' in pth_model_name:  # efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, ....
        model_type = "EfficientNet"
        # enter the code to convert pytorch 'efficientnet' model so that can be used in Jetbot application.

    elif pth_model_name == 'inception_v3':  # Inception_v3
        model_type = "InceptionNet"
        # enter the code to convert pytorch 'inception_v3' model so that can be used in Jetbot application.

    elif pth_model_name == 'googlenet':  # Inception
        model_type = "GoogleNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = "GoogLeNet_Weights"

        model, preprocess_wrap = load_pth_model(pth_model_name, weights_cls, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        # model.dropout = torch.nn.Dropout(p=dropout)
        if model.aux_logits:
            model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, 2)
            model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, 2)
        #   model.aux1.dropout = torch.nn.Dropout(p=dropout)
        #   model.aux2.dropout = torch.nn.Dropout(p=dropout)

    elif "densenet" in pth_model_name:  # densenet121, densenet161, densenet169, densenet201
        model_type = "DenseNet"
        # enter the code to convert pytorch 'densenet' model so that can be used in Jetbot application.

    elif "shufflenet_v2" in pth_model_name:  # shufflenet_v2_x1_0 or shufflenet_v2_x0_5
        model_type = "ShuffleNet"
        # enter the code to convert pytorch 'shufflenet_v2' model so that can be used in Jetbot application.

    elif "mnasnet" in pth_model_name:  #  mnasnet2_0,  mnasnet1_5, mnasnet1_0, or mnasnet0_5
        model_type = "MnasNet"
        # enter the code to convert pytorch 'mnasnet' model so that can be used in Jetbot application.

    else:
        assert (
                model is not None and model_type is not None), "Check if the model name set is compatible with torchvision."

    return model, model_type, preprocess_wrap


class model_selection(HasTraits):
    model_function = Unicode(default_value='object detection').tag(config=True)
    model_function_list = List(default_value=[]).tag(config=True)
    model_type = Unicode(default_value='SSD').tag(config=True)
    model_type_list = List(default_value=[]).tag(config=True)
    model_path = Unicode(default_value='').tag(config=True)
    model_path_list = List(default_value=[]).tag(config=True)
    selected_model_path = Unicode(default_value='').tag(config=True)
    preprocess_path = Unicode(default_value='').tag(config=True)
    is_selected = Bool(default_value=False).tag(config=True)

    def __init__(self, core_library='TensorRT', dir_model_repo=MODEL_REPO_DIR_DOCKER):
        super().__init__()

        self.core_library = core_library
        if self.core_library == 'TensorRT':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "trt_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)
        elif self.core_library == 'Pytorch':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "torch_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)

        for p in self.df.values:
            p[2] = os.path.join(dir_model_repo, p[2].split("/", 1)[1])  # add "workspace" to the path of model
            p[3] = os.path.join(dir_model_repo, p[3].split("/", 1)[1])  # and model preprocess

        self.model_function_list = list(self.df["model_function"].astype("category").cat.categories)
        self.update_model_type_list()
        # d_mf = self.df[self.df.model_function == self.model_function]   # data frame of given function
        # self.model_type_list = list(d_mf["model_type"].astype("category").cat.categories)
        self.update_model_list()
        # mpl = d_mf[d_mf.model_type == self.model_type].loc[:, ['model_path']].values.tolist()
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.observe(self.update_model, names=['model_function', 'model_type', 'model_path'])
        # self.is_selected = False
        # self.observe(self.selected, names=['is_selected'])

    def update_model_type_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        # mt = mf[mf.model_type == self.model_type]
        self.model_type_list = list(
            mf["model_type"].astype("category").cat.categories)  # the model types of the given model function
        return self.model_type_list

    def update_model_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        mt = mf[mf.model_type == self.model_type]  # select the models from the given model type
        mpl = mt.loc[:, ['model_path']].values
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.model_path_list = mpl[:, 0].tolist()
        return self.model_path_list

    def update_model(self, change):
        # print(change)
        if change['name'] == 'model_function':
            self.model_function = change['new']
            self.update_model_type_list()
        if change['name'] == 'model_type':
            self.model_type = change['new']
            self.update_model_list()
        if change['name'] == 'model_path':
            self.model_path = change['new']
            mp = self.df[self.df.model_path == self.model_path]
            mpp = mp.preprocess_path.tolist()
            # print("preprocess path: ", mpp)
            self.preprocess_path = mpp[0]

            # self.selected_model_path = os.path.join(MODEL_REPO_DIR_DOCKER, self.model_path.split("/", 1)[1])
        # print(self.selected_model_path)

    # def selected(self, change):
    #     self.is_selected = change['new']


'''
ms = trt_model_selection()
# ms.model_function = 'object detection'
# ms.model_type = 'SSD_FPN'
# model_type_list = ms.update_model_type_list()
model_path_list = ms.update_model_list()
print(ms.model_function_list, ms.model_type_list)
print(model_path_list)
'''

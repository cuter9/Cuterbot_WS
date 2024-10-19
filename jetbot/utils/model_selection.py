import pandas as pd
import os
from traitlets import HasTraits, Unicode, List, Bool
# import numpy as np

import torch
import torchvision.models as pth_models
import torchvision

HEAD_LIST = ['model_function', 'model_type', 'model_path']
MODEL_REPO_DIR = os.path.join(os.environ["HOME"], "model_repo")
MODEL_REPO_DIR_DOCKER = os.path.join("/workspace", "model_repo")
os.environ['MODEL_REPO_DIR_DOCKER'] = MODEL_REPO_DIR_DOCKER
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


def load_tune_pth_model(pth_model_name="resnet18", pretrained=True):
    """
    if pretrained:
        model = getattr(pth_models, pth_model_name)()  # for fine-tuning
    else:
        model = getattr(pth_models, pth_model_name)(pretrained=False)  # for inferencing
    """
    model_type = None
    model = None

    tv = int(torchvision.__version__.split(".")[1])     # torchvision version
    # ----- modify the last layer for classification, and the model used in notebook should be modified too.
    if 'mobilenet_v3' in pth_model_name:  # 'mobilenet_v3_large' or  'mobilenet_v3_small'
        model_type = "MobileNet"
        if tv >= 13:        # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = None
            if pretrained:
                if "small" in pth_model_name:
                    weights_cls = "MobileNet_V3_Small_Weights"
                elif "large" in pth_model_name:
                    weights_cls = "MobileNet_V3_Large_Weights"
                else:
                    assert weights_cls is not None, "Check the use of the pretrained model!"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
                                              2)  # for mobilenet_v3 model. must add block expansion factor 4

    elif pth_model_name == 'mobilenet_v2':
        model_type = "MobileNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights = getattr(pth_models, "MobileNet_V2_Weights").DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                print("torchvision version: %d" % tv)
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features,
                                              2)  # for mobilenet_v2 model. must add block expansion factor 4

    elif pth_model_name == 'vgg11':  # VGGNet
        model_type = "VggNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights = getattr(pth_models, "VGG11_Weights").DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,
                                              2)  # for VGG model. must add block expansion factor 4

    elif 'resnet' in pth_model_name:  # ResNet
        model_type = "ResNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights_cls = pth_model_name.replace("resnet", "ResNet") + "_Weights"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                print("torchvision version: %d" % tv)
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features,
                                   2)  # for resnet model must add block expansion factor 4
        # model.fc = torch.nn.Linear(512, 2)

    elif 'efficientnet' in pth_model_name:  # ResNet
        model_type = "EfficientNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights_cls = pth_model_name.replace("efficientnet_b", "EfficientNet_B") + "_Weights"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                print("torchvision version: %d" % tv)
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # for efficientnet model
        # model.classifier[0].dropout = torch.nn.Dropout(p=dropout)

    elif pth_model_name == 'inception_v3':  # Inception_v3
        model_type = "InceptionNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights = getattr(pth_models, "Inception_V3_Weights").DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        # model.dropout = torch.nn.Dropout(p=dropout)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        if model.aux_logits:
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 2)

    elif pth_model_name == 'googlenet':  # Inception_v3
        model_type = "GoogleNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights = getattr(pth_models, "GoogLeNet_Weights").DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        # model.dropout = torch.nn.Dropout(p=dropout)
        if model.aux_logits:
            model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, 2)
            model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, 2)
        #   model.aux1.dropout = torch.nn.Dropout(p=dropout)
        #   model.aux2.dropout = torch.nn.Dropout(p=dropout)

    elif "densenet" in pth_model_name:  # densenet121, densenet161, densenet169, densenet201
        model_type = "DenseNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights_cls = pth_model_name.replace("densenet", "DenseNet") + "_Weights"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    elif "shufflenet_v2" in pth_model_name:  # shufflenet_v2_x1_0 or shufflenet_v2_x0_5
        model_type = "ShuffleNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights_cls = pth_model_name.replace("shufflenet_v2_x", "ShuffleNet_V2_X") + "_Weights"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

    elif "mnasnet" in pth_model_name:  # mnasnet1_0 or mnasnet0_5
        model_type = "MnasNet"
        if tv >= 13:
            print("torchvision version: %d" % tv)
            if pretrained:
                weights_cls = pth_model_name.replace("mnasnet", "MNASNet") + "_Weights"
                weights = getattr(pth_models, weights_cls).DEFAULT
                model = getattr(pth_models, pth_model_name)(weights=weights)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(weights=None)  # for inferencing
        else:
            print("torchvision version: %d" % tv)
            model = getattr(pth_models, pth_model_name)(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    else:
        assert (
                model is not None and model_type is not None), "Check if the model name set is compatible with torchvision."

    return model, model_type


class model_selection(HasTraits):
    model_function = Unicode(default_value='object detection').tag(config=True)
    model_function_list = List(default_value=[]).tag(config=True)
    model_type = Unicode(default_value='SSD').tag(config=True)
    model_type_list = List(default_value=[]).tag(config=True)
    model_path = Unicode(default_value='').tag(config=True)
    model_path_list = List(default_value=[]).tag(config=True)
    selected_model_path = Unicode(default_value='').tag(config=True)
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
            p[2] = os.path.join(dir_model_repo, p[2].split("/", 1)[1])

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

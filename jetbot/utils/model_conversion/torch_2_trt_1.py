import tensorrt as trt
from torch2trt import torch2trt
import os
from model_selection import load_tune_pth_model
import torch

model_name = "mobilenet_v3_small"
MODEL_REPO_DIR = "/home/cuterbot/model_repo/road_following"
path_torch_model = os.path.join(MODEL_REPO_DIR, "best_steering_model_xy_" + model_name + ".pth")

model, model_type = load_tune_pth_model(pth_model_name=model_name, pretrained=False)
model.load_state_dict(torch.load(path_torch_model))
model = model.cuda().eval().half()
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True, log_level=trt.Logger.VERBOSE)



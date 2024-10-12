import torch_tensorrt

import torch
from torch_tensorrt.ts import convert_method_to_trt_engine as torch2engine
import os
from model_selection import load_tune_pth_model

device = 'cpu'
img_size = [224, 224]
batch_size = 1

model_name = "resnet34"
MODEL_REPO_DIR = "/home/cuterbot/model_repo/road_following"
path_torch_model = os.path.join(MODEL_REPO_DIR, "best_steering_model_xy_" + model_name + ".pth")

model, model_type = load_tune_pth_model(pth_model_name=model_name, pretrained=False)
model.load_state_dict(torch.load(path_torch_model))

img = torch.randn(batch_size, 3, *img_size)
# ts_model = torch.jit.script(model, example_inputs=img)
# ts_model = torch.jit.script(model)
ts_model = torch.jit.trace(model, img)

trt_eng = torch2engine(ts_model, method_name='forward', inputs=[img], max_batch_size=1,
                       num_min_timing_iters=2, num_avg_timing_iters=1, workspace_size=1 << 30,
                       strict_types=False, enabled_precisions=torch_tensorrt.dtype.float16, debug=True)

with open("output_engine", 'wb') as f:
    f.write(trt_eng)
    f.close()

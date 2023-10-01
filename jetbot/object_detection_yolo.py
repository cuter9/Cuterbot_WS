import tensorrt as trt
from jetbot.yolo_tensorrt import TrtYOLO, load_plugins, get_cls_dict
# from jetbot.yolo_tensorrt import TrtYOLO_sync, load_plugins, get_cls_dict
import numpy as np
import cv2


class ObjectDetector_YOLO(object):
        def __init__(self, engine_path, category_num=80):
            load_plugins()
            # cls_dict = get_cls_dict(category_num)
            self.trt_model = TrtYOLO(engine_path)
            # self.trt_model = TrtYOLO_sync(engine_path)
            
        def execute(self, inputs):
            trt_outputs = self.trt_model.execute(inputs)
            # print(trt_outputs)
            return trt_outputs
    
        def __call__(self, inputs):
            return self.execute(inputs)

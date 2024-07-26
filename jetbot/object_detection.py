from xml.sax.xmlreader import InputSource

# import tensorrt as trt
from jetbot.ssd_tensorrt import parse_boxes, parse_boxes_fpn, TRT_INPUT_NAME, TRT_OUTPUT_NAME
from .tensorrt_model import TRTModel, parse_boxes_yolo, parse_boxes_yolo_v7
import numpy as np
import cv2
from traitlets import HasTraits, Unicode, Float

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])


def bgr8_to_ssd_input(camera_value, input_shape):
    """Preprocess an image size to meet the size of model input before TRT SSD inferencing.
    """
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # x = cv2.resize(x, (300, 300))
    x = cv2.resize(x, input_shape)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]


def bgr8_to_ssd_fpn_input(camera_value, input_shape):
    """Preprocess an image size to meet the size of model input before TRT SSD FPN model inferencing.
       the input for SSD FPN mode which converted from Tensorflow V2 does nor need normalization to +/- 1
       because the normalization function has been already included in the model
    """
    x = camera_value
    x = cv2.resize(x, input_shape)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # x = cv2.resize(x, (300, 300))
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    # return x[None, ...]
    return x[None, ...]


def preprocess_yolo(img, input_shape, letter_box=False):
    """Preprocess an image size to meet the size of model input before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, input_shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def preprocess_yolo_v7(img, input_shape, letter_box=False):
    """Preprocess an image size to meet the size of model input before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, input_shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


class ObjectDetector(HasTraits):
    engine_path = Unicode(default_value='').tag(config=True)
    conf_th = Float(default_value=0.5).tag(config=True)

    def __init__(self):
        # assert type_model is not None
        super().__init__()
        self.input_shape = None
        self.postprocess_od = None
        self.preprocess_od = None
        self.type_model_od = None
        self.trt_model_od = None

        # self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME],
        #                          output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
    def load_od_engine(self):
        self.trt_model_od = TRTModel(self.engine_path)
        if self.type_model_od == 'SSD':
            self.preprocess_od = bgr8_to_ssd_input
            self.postprocess_od = parse_boxes
        elif self.type_model_od == 'SSD_FPN':
            self.preprocess_od = bgr8_to_ssd_fpn_input
            self.postprocess_od = parse_boxes_fpn
        elif self.type_model_od == 'YOLO':
            self.preprocess_od = preprocess_yolo
            self.postprocess_od = parse_boxes_yolo
        elif self.type_model_od == 'YOLO_v7':
            self.preprocess_od = preprocess_yolo_v7
            self.postprocess_od = parse_boxes_yolo_v7
        self.input_shape = self.trt_model_od.input_shape

    def execute_od(self, *inputs, conf_th=None):

        if conf_th is None:
            conf_th = self.conf_th
        trt_outputs = self.trt_model_od(self.preprocess_od(*inputs, self.input_shape))
        if self.type_model_od == 'YOLO_v7':
            detections = self.postprocess_od(trt_outputs, self.input_shape, conf_th=conf_th)
        else:
            detections = self.postprocess_od(trt_outputs, conf_th=conf_th)

        # print(detections)
        return detections

    def __call__(self, *inputs, conf_th=None):
        if conf_th is None:
            conf_th = self.conf_th
        return self.execute_od(*inputs, conf_th=conf_th)

"""yolo_with_plugins.py

Implementation of TrtYOLO class with the yolo_layer plugins.
"""


from __future__ import print_function

import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import ctypes
import atexit


# try:
#    ctypes.cdll.LoadLibrary('./plugins/libyolo_layer.so')
# except OSError as e:
#    raise SystemExit('ERROR: failed to load ./plugins/libyolo_layer.so.  '
#                     'Did you forget to do a "make" in the "./plugins/" '
#                     'subdirectory?') from e

def load_plugins():
    library_path = os.path.join(os.path.dirname(__file__), 'libyolo_layer.so')
    ctypes.CDLL(library_path)

def _preprocess_yolo(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.

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
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


def _postprocess_yolo(trt_outputs, img_w, img_h, conf_th, nms_threshold,
                      input_shape, letter_box=False):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()

    # Returns
        boxes, scores, classes (after NMS)
    """
    # filter low-conf detections and concatenate results of all yolo layers
    detections = []
    for o in trt_outputs:
        dets = o.reshape((-1, 7))
        # print(np.shape(dets), dets[3:5])
        dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
        detections.append(dets)
    
    detections = np.concatenate(detections, axis=0)

    if len(detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        box_scores = detections[:, 4] * detections[:, 6]

        # scale x, y, w, h from [0, 1] to pixel values
        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        if letter_box:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2
        
        detections[:, 0:4] *= np.array(
            [old_w, old_h, old_w, old_h], dtype=np.float32)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        if letter_box:
            xx = xx - offset_w
            yy = yy - offset_h
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        # boxes = np.concatenate([xx,  yy,  (xx+ww), (yy+hh)], axis=1)
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
        
        # print(np.shape(classes))

    return boxes, scores, classes


def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

class TrtYOLO_sync(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, engine_path, input_names=None, output_names=None, category_num=80, letter_box=False):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.category_num = category_num
        self.letter_box = letter_box
        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # self.stream = torch.cuda.Stream()
        # self.stream = torch.cuda.default_stream()

        self.input_shape = get_input_shape(self.engine)

        
        if self.engine.has_implicit_batch_dimension:
            print("engine is built from uff model")
        else:
            print("engine is built from onnx model")

        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names
            
        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names
            
        # self.final_shapes = final_shapes
        
        # destroy at exit
        atexit.register(self.destroy)
    
    def _input_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
    
    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
    
    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]
    
    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def create_output_buffers(self, batch_size):
        outputs_cuda = [None] * len(self.output_names)
        outputs_host = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            # dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            # if self.final_shapes is not None:
            #    shape = (batch_size, ) + self.final_shapes[i]
            # else:
                # print("output binding shape : ",  idx, self.engine.get_binding_shape(idx))
            #    shape = (batch_size, ) + tuple(self.engine.get_binding_shape(idx))
            # device = torch_device_from_trt(self.engine.get_location(idx))

            binding_dims = self.engine.get_binding_shape(idx)
            if len(binding_dims) == 4:
                # explicit batch case (TensorRT 7+)
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                # implicit batch case (TensorRT 6 or older)
                size = trt.volume(binding_dims) * self.engine.max_batch_size
            else:
                raise ValueError('bad dims of binding %s: %s' % (output_name, str(binding_dims)))
            
            assert size % 7 == 0

            outputs_host[i] = cuda.pagelocked_empty(size, dtype)
            outputs_cuda[i] = cuda.mem_alloc(outputs_host[i].nbytes)
            # output = torch.empty(size=shape, dtype=dtype, device=device)
            # outputs[i] = output_host, output_cuda
        return outputs_host, outputs_cuda 

    def form_detection(self, boxes, scores, classes, img_w, img_h):
        all_detections = []
        detections = []
        for b, s, c in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            det_dict = dict(label = int(c), confidence = float(s), 
                            bbox = [b[0]/img_w, b[1]/img_h, b[2]/img_w, b[3]/img_h])
            # det_dict = dict(label = c, confidence = s, bbox = b)
            detections.append(det_dict)
        all_detections.append(detections)
        # print([d["label"] for d in all_detections[0]])
        return  all_detections

    def execute(self, img, conf_th=0.3, letter_box=None):
        letter_box = self.letter_box if letter_box is None else letter_box
        img_resized = _preprocess_yolo(img, self.input_shape, letter_box)
        # print(np.shape(img_resized), img_resized[-1])
        batch_size = img.shape[0]

        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        # map input bindings
        inputs_host = [None] * len(self.input_names)
        inputs_cuda = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            # self.inputs[0].host = np.ascontiguousarray(inputs)
            idx = self.engine.get_binding_index(name)
            # convert to appropriate format
            # inputs_torch[i] = torch.from_numpy(inputs[i])
            inputs_host[i] = np.ascontiguousarray(img_resized)
            inputs_cuda[i] = cuda.mem_alloc(inputs_host[i].nbytes)
            cuda.memcpy_htod(inputs_cuda[i], inputs_host[i])
            # inputs_cuda[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)), memory_format=torch.contiguous_format)
            # inputs_cuda[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            
            # bindings[idx] = int(inputs_cuda[i].data_ptr())
            bindings[idx] = int(inputs_cuda[i])
        assert len(inputs_host) == 1

        outputs_host, outputs_cuda = self.create_output_buffers(batch_size)
        
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            bindings[idx] = int(outputs_cuda[i])
        assert len(outputs_host) == 1
        
        # self.context.execute(batch_size, bindings)
        # self.context.execute_async(batch_size, bindings, stream_handle = self.stream.cuda_stream)
        self.context.execute_v2(bindings)
        # self.stream.synchronize()
        
        for i, out in enumerate(outputs_cuda):
            cuda.memcpy_dtoh(outputs_host[i], out)
        
        boxes, scores, classes = _postprocess_yolo(
            outputs_host, img.shape[1], img.shape[0], conf_th,
            nms_threshold=0.5, input_shape=self.input_shape,
            letter_box=letter_box)        
        
        # if self.engine.has_implicit_batch_dimension:
        #    outputs = [buffer.cpu().numpy() for buffer in output_buffers]
        # else:
        #    outputs = [np.squeeze(buffer.cpu().numpy(), axis=0) for buffer in output_buffers]
        # outputs = [buffer.cpu().numpy() for buffer in output_buffers]
        
        # self.stream.synchronize()

        # clip x1, y1, x2, y2 within original image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)
         # return boxes, scores, classes
        # print(classes)
        return self.form_detection(boxes, scores, classes, img.shape[1],  img.shape[0])
    
    # def __call__(self, inputs):
    #    return self.execute(inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()

        
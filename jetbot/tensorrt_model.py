import numpy as np
import torch
import tensorrt as trt
import atexit

def nms_boxes_yolo(detections, nms_threshold):
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

        # width1 = np.maximum(0.0, xx2 - xx1 + 1)
        # height1 = np.maximum(0.0, yy2 - yy1 + 1)
        width1 = np.maximum(0.0, xx2 - xx1)
        height1 = np.maximum(0.0, yy2 - yy1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

# def parse_boxes_yolo(trt_outputs, conf_th=0.3, nms_threshold=0.5, 
#                      input_shape, letter_box=False):
def parse_boxes_yolo(trt_outputs, conf_th=0.3, nms_threshold=0.5):
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
        # print(np.shape(dets))
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
        # old_h, old_w = img_h, img_w
        # offset_h, offset_w = 0, 0
        # if letter_box:
        #    if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
        #        old_h = int(input_shape[0] * img_w / input_shape[1])
        #        offset_h = (old_h - img_h) // 2
        #    else:
        #        old_w = int(input_shape[1] * img_h / input_shape[0])
        #        offset_w = (old_w - img_w) // 2
        
        # detections[:, 0:4] *= np.array(
        #    [old_w, old_h, old_w, old_h], dtype=np.float32)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = nms_boxes_yolo(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        # if letter_box:
        #    xx = xx - offset_w
        #    yy = yy - offset_h
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        # boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = np.concatenate([xx, yy, (xx+ww), (yy+hh)], axis=1)
        # boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
        
    all_detections = []
    detections = []
    for b, s, c in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        det_dict = dict(label = int(c), confidence = float(s), 
                        bbox = [b[0], b[1], b[2], b[3]])
        # det_dict = dict(label = c, confidence = s, bbox = b)
        detections.append(det_dict)
    all_detections.append(detections)
    # print(detections)
    return all_detections
    # return boxes, scores, classes

def torch_dtype_to_trt(dtype):
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)

    
class TRTModel(object):
    
    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):
        
        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # self.stream = torch.cuda.Stream()
        # self.stream = torch.cuda.default_stream()

        if self.engine.has_implicit_batch_dimension:
            print("engine is built from uff model")
            self.input_shape = tuple(self.engine.get_binding_shape(self.engine[0])[1:])
        else:
            print("engine is built from onnx model")
            self.input_shape = tuple(self.engine.get_binding_shape(self.engine[0])[2:])
        
        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names
            
        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names
            
        self.final_shapes = final_shapes
        
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
    
    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size, ) + self.final_shapes[i]
            else:
                # print("output binding shape : ",  idx, self.engine.get_binding_shape(idx))
                if self.engine.has_implicit_batch_dimension:
                    shape = (batch_size, ) + tuple(self.engine.get_binding_shape(idx))
                else:
                    shape = tuple(self.engine.get_binding_shape(idx))
            
            # print("output shape : " , shape)
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
        return outputs
    
    def execute(self, *inputs):
        batch_size = inputs[0].shape[0]

        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            
            # convert to appropriate format
            inputs_torch[i] = torch.from_numpy(inputs[i])
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)), memory_format=torch.contiguous_format)
            inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            
            bindings[idx] = int(inputs_torch[i].data_ptr())
            
        output_buffers = self.create_output_buffers(batch_size)
        
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            bindings[idx] = int(output_buffers[i].data_ptr())
        
        # self.context.execute_async(batch_size, bindings, stream_handle = self.stream.cuda_stream)
        # self.context.execute_v2(batch_size, bindings)
        # self.context.execute_v2(bindings)
        # self.stream.synchronize()

        if self.engine.has_implicit_batch_dimension:
            self.context.execute(batch_size, bindings)
        else:
            self.context.execute_v2(bindings)
        
        outputs = [buffer.cpu().numpy() for buffer in output_buffers]
   
        # self.stream.synchronize()
        
        return outputs
    
    def __call__(self, *inputs):
        return self.execute(*inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()
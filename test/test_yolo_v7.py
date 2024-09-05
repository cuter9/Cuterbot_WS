from PIL import Image, ImageDraw, ImageColor
import cv2
from jetbot import bgr8_to_jpeg
import os, sys
import wget
import matplotlib.pyplot as plt

WINDOW_NAME = 'TrtSsdModelTest'

from jetbot import ObjectDetector

# follower_model = "/workspace/model_repo/object_detection/yolo_v7.engine"
follower_model = "/home/cuterbot/model_repo/object_detection/yolo_v7.engine"
# follower_model = "/home/cuterbot/Data_Repo/Model_Conversion/SSD_mobilenet/ONNX_Model/Repo/ssd_mobilenet_v2_320x320_coco17_tpu-8_tf_v2.engine"

# detector = ObjectDetector(follower_model, type_model='YOLO_v7', conf_th=0.3)
detector = ObjectDetector()
detector.engine_path = follower_model
detector.type_model_od = "YOLO_v7"
detector.load_od_engine()

[height, width] = detector.input_shape
print("model input size - width, height", width, height)

if not os.path.exists("test.jpg"):
    wget.download("http://images.cocodataset.org/val2017/000000088462.jpg", out="test.jpg")
    # wget.download("http://farm9.staticflickr.com/8048/8082815883_bf99431b2b_z.jpg", out="test_1.jpg")
# img = Image.open(test_img)
img_handle = cv2.imread("test.jpg")
img_height, img_width, _ = img_handle.shape
print(img_width, img_height)

detections = detector.execute_od(img_handle)
print(detections[0])
for det in detections[0]:
    bbox = det['bbox']
    cv2.rectangle(img_handle, (int(img_width * bbox[0]), int(img_height * bbox[1])),
                  (int(img_width * bbox[2]), int(img_height * bbox[3])), (255, 0, 0), 2)
image = bgr8_to_jpeg(img_handle)
cv2.imshow(WINDOW_NAME, img_handle)
cv2.waitKey(0)
# cv2.destroyAllWindows()

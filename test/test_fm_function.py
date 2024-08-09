# from main_scripts import Object_Follower
import os.path

from jetbot import FleeterTRT
import cv2
from jetbot import bgr8_to_jpeg

type_follower_model = "YOLO_v7"  # "SSD", "SSD_FPN", "YOLO", "YOLO_v7"
# follower_model='ssd_mobilenet_v2_coco_onnx.engine'
# follower_model='ssd_mobilenet_v2_fpnlite_320x320_coco17.engine'
# follower_model='ssd_mobilenet_v2_320x320_coco17.engine'
# follower_model='ssd_mobilenet_v2_fpnlite_640x640_coco17.engine'
# follower_model='ssd_mobilenet_v1_fpn_640x640_coco17.engine'
# follower_model='yolov3_tiny_288.engine'
# follower_model='yolov4_tiny_288.engine'
# follower_model='yolov4_tiny_416.eigine'
# follower_model='yolov4_288.engine'
# follower_model='yolov4_416.engine'
follower_model = 'yolo_v7.engine'
# follower_model='yolo_v7-tiny.engine'

type_cruiser_model = "resnet"
cruiser_model = 'resnet18'

path_follower_model = os.path.join(os.getcwd(), "test", follower_model)
path_cruiser_model = os.path.join(os.getcwd(), "test", 'best_steering_model_xy_trt_resnet18.pth')
FL = FleeterTRT(follower_model=path_follower_model, type_follower_model=type_follower_model,
                cruiser_model=path_cruiser_model, type_cruiser_model=type_cruiser_model, conf_th=0.3)


def execute(change):

    FL.current_image = change['new']
    FL.run_objects_detection()

    for det in FL.detections[0]:
        bbox = det['bbox']
        cv2.rectangle(FL.current_image, (int(FL.img_width * bbox[0]), int(FL.img_height * bbox[1])),
                      (int(FL.img_width * bbox[2]), int(FL.img_height * bbox[3])), (255, 0, 0), 2)

    image = bgr8_to_jpeg(FL.current_image)
    cv2.imshow("test", FL.current_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


FL.capturer.observe(execute, names="value")
# execute(FL.capturer.value)

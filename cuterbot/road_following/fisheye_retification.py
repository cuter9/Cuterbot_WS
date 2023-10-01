import numpy as np
import cv2
DIM = (300, 300)
points = np.array([[[200, 150]]]).astype(np.float32)
newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, DIM, None, balance=1)
dst = cv2.fisheye.undistortPoints(points, K, D, None, newcameramtx)
# [[[323.35104 242.06458]]]
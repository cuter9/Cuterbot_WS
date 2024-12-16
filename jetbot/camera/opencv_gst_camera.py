# import time
import traitlets
import atexit
import cv2
import threading
import numpy as np
from .camera_base import CameraBase
# import os
import time

SudoPass = 'cuterbot'


def capture_frames(cam):
    while True:

        re, image = cam.cap.read()
        if re:
            cam.value = image

        else:
            print("No frame was captured, check the cam capture function works normally, "
                  "and the cam capture thread will be terminated!")
            break


class OpenCvGstCamera(CameraBase):
    value = traitlets.Any()

    # config
    # width = traitlets.Integer(default_value=224).tag(config=True)
    # height = traitlets.Integer(default_value=224).tag(config=True)
    width = traitlets.Integer(default_value=640).tag(config=True)
    height = traitlets.Integer(default_value=640).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    # capture_width = traitlets.Integer(default_value=816).tag(config=True)
    # capture_height = traitlets.Integer(default_value=616).tag(config=True)
    capture_width = traitlets.Integer(default_value=1640).tag(config=True)
    capture_height = traitlets.Integer(default_value=1232).tag(config=True)

    # cap_time = traitlets.Float(default_value=0).tag(config=True)

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        # self.value = np.empty((self.capture_height, self.capture_width, 3), dtype=np.uint8)
        self.stop_thread = threading.Event()
        super().__init__(self, *args, **kwargs)

        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()
            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()

        except:
            self.stop()
            raise RuntimeError('Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):

        while True:

            if self.stop_thread.is_set():
                break

            re, image = self.cap.read()
            if re:
                self.value = image
                # print(image)
            else:
                if not self.cap.isOpened():
                    self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
                    print("No frame was captured and camera is not opened, try to re-opening the cap again!")
                    # else:
                # print("No frame was captured, check the cam capture function works normally, "
                #      "and the cam capture thread will be terminated!")
                # break

    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % \
            (self.capture_width, self.capture_height, self.fps, self.width, self.height)
        # return 'nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % \
        #   (self.capture_width, self.capture_height, self.fps, self.width, self.height)
        # return 'nvarguscamerasrc sensor-mode=3 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
        #    self.width, self.height)

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)

        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            # self.thread = threading.Thread(target=capture_frames, args=(self,))
            self.thread.start()

    def stop(self):

        if hasattr(self, 'thread'):
            self.stop_thread.set()
            self.thread.join()
            time.sleep(1)
            print("Capture thread is stopped")

        if hasattr(self, 'cap'):
            self.cap.release()
            print("Camera operation is released! ")

    #           os.popen("sudo -S %s"%('service nvargus-daemon restart'), 'w').write(SudoPass)
    #           print("service nvargus-daemon is restarted ! ")

    # if hasattr(self, 'thread'):
    #   self.thread.join(timeout=2.0)
    #    print("Capture thread is stopped")

    def restart(self):
        self.stop()
        self.start()

    @staticmethod
    def instance(*args, **kwargs):
        return OpenCvGstCamera(*args, **kwargs)

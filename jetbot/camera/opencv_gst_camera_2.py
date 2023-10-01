import traitlets
import atexit
import cv2
import threading
import numpy as np
from .camera_base import CameraBase


class OpenCvGstCamera(CameraBase):
    
    value = traitlets.Any()
    
    # config
    # width = traitlets.Integer(default_value=224).tag(config=True)
    # height = traitlets.Integer(default_value=224).tag(config=True)
    width = traitlets.Integer(default_value=300).tag(config=True)
    height = traitlets.Integer(default_value=300).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    # capture_width = traitlets.Integer(default_value=816).tag(config=True)
    # capture_height = traitlets.Integer(default_value=616).tag(config=True)
    capture_width = traitlets.Integer(default_value=1920).tag(config=True)
    capture_height = traitlets.Integer(default_value=1080).tag(config=True)

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.thread_running = False
        super().__init__(self, *args, **kwargs)

        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        
        while self.thread_running:
            
            nc = 0
            re, image = self.cap.read()
            
            if image is None:
                
                if nc <= 1:
                    nc += 1
                    continue
                
                else:
                    print("image is not captured !")
                    break
                
            self.value = image
            
        print("thread is not running") 
        self.thread_running = False
        
    def _gst_str(self):
        return 'nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                self.capture_width, self.capture_height, self.fps, self.width, self.height)
        
        # return 'nvarguscamerasrc sensor_mode=3 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (self.width, self.height)

    
    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        # if not self.thread_running or not self.thread.isAlive():
        if not self.thread_running:
            self.thread_running = True
            self.thread = threading.Thread(target=self._capture_frames)
            # self.thread = threading.Thread(target=capture_frames, args=(self,))
            self.thread.start()

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
            print("camera is stopped")
        if self.thread_running:
            self.thread_running = False
            self.thread.join(timeout=2.0)
            print("thread is stopped") 

            
    def restart(self):
        self.stop()
        self.start()
        
    @staticmethod
    def instance(*args, **kwargs):
        return OpenCvGstCamera(*args, **kwargs)

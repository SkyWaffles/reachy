import cv2
import threading

class Resolution:
    width = 320
    height = 180

class CameraThread(threading.Thread):
    def __init__(self, view_name, camera_id):
        threading.Thread.__init__(self)
        self.view_name = view_name
        self.camera_id = camera_id
    def run(self):
        print(f"Starting {self.view_name}")
        CameraView(self.view_name, self.camera_id)

def CameraView(view_name, camera_id):
    cv2.namedWindow(view_name)
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Resolution.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Resolution.height)
    if camera.isOpened():  # try to get the first frame
        rval, frame = camera.read()
    else:
        rval = False
        print(f"{view_name} with camera id {camera_id} couldn't be opened.")

    while rval:
        cv2.imshow(view_name, frame)
        rval, frame = camera.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(view_name)

# Create two camera threads
thread1 = CameraThread("Camera 1", 1)
thread2 = CameraThread("Camera 2", 2)
thread1.start()
thread2.start()
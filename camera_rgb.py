import cv2
import datetime
import threading

def get_timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

class MaxResolution:
    width = 320
    height = 240

class CameraSpecs:
    def __init__(self,
        fourcc: int = None,
        fps: float = None,
        height: int = None,
        width: int = None) -> None:

        # handle resolutions that are too high for USB port to handle
        if width and width > MaxResolution.width:
            print(ValueError(f"Video width resolution can't be greater than {MaxResolution.width}"))
            width = None
        self.width = width if width else MaxResolution.width
        if height and height > MaxResolution.height:
            print(ValueError(f"Video height resolution can't be greater than {MaxResolution.height}"))
            height = None

        self.fourcc = fourcc if fourcc else cv2.VideoWriter_fourcc(*'XVID')
        self.fps = fps if fps else 24.0
        self.width = width if width else MaxResolution.width
        self.height = height if height else MaxResolution.height
            

class CameraThread(threading.Thread):
    def __init__(self, view_name: str, camera_id: int, camera_specs: CameraSpecs):
        threading.Thread.__init__(self)
        self.view_name = view_name
        self.camera_id = camera_id
        self.specs = camera_specs
        
    def run(self):
        print(f"Starting {self.view_name}")
        CameraView(self.view_name, self.camera_id, self.specs)

def CameraView(view_name, camera_id, specs):
    cv2.namedWindow(view_name)
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, specs.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, specs.height)
    if camera.isOpened():  # try to get the first frame
        rval, frame = camera.read()
    else:
        rval = False
        print(f"{view_name} with camera id {camera_id} couldn't be opened.")

    is_recording = False
    while rval:
        cv2.imshow(view_name, frame)
        rval, frame = camera.read()
        timestamp = get_timestamp()
        key = cv2.waitKey(20)

        if key == 27:  # ESC: exit
            break
        elif key == 115:  # s: save current image
            img_name = f"{timestamp}_cam_{camera_id}.jpg"
            print(f"Attempting to write frame to {img_name}...")
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved successfully")
        elif key == 32:  # space-bar: toggle stream recording
            if not is_recording:
                # transition to recording
                stream_name = f"{timestamp}_cam_{camera_id}.avi"
                print(f"Started recording to {stream_name}...")
                recording = cv2.VideoWriter(stream_name, specs.fourcc, specs.fps, (specs.width, specs.height))
                is_recording = True
            else:
                # stop recording
                print(f"Stopped recording")
                recording.release()
                is_recording = False
        
        if is_recording:
            recording.write(frame)


    camera.release()
    cv2.destroyWindow(view_name)

# Create two camera threads
spec_1 = CameraSpecs()
spec_2 = CameraSpecs()
thread1 = CameraThread("Camera 1", 1, spec_1)
thread2 = CameraThread("Camera 2", 2, spec_2)
thread1.start()
thread2.start()
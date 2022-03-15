import re
import cv2
import datetime
import threading


def get_timestamp():
    return datetime.datetime.now().strftim


class Resolution:
    # OpenCV only allows certain resolutions when recording
    opencv_dimensions = {
        "120p": (160, 120),
        "144p": (176, 144),
        "240p": (320, 240),
        "288p": (352, 288),
        "480p": (640, 480),
        "768p": (1024, 768),
        "1024p": (1280, 1024),
    }

    def __init__(self, *args: str) -> None:
        if len(args) == 1:
            # assume progressive scan value has been passed in
            if isinstance(args, str):
                progressive_scan = args[0]
                self.width = self.opencv_dimensions[progressive_scan](0)
                self.height = self.opencv_dimensions[progressive_scan](1)
            else:
                raise ValueError("Single argument assumes a string, progessive scan value (like '240p')")
        elif len(args) == 2:
            # assume width and height have been passed in
            self.width = args[0]
            self.height = args[1]
        elif len(args) > 2:
            raise ValueError("Instantiating Resolution: Too many arguments")
 
    def __leq__(self, other: 'Resolution'):
        if self.width <= other.width and self.height <= other.height:
            # this resolution clearly fits inside the other
            return True
        else:
            # all other cases, this resolution doesn't fully fit inside other
            return False
    
    def __str__(self) -> str:
        return f"({self.width} x {self.height})"


class CameraSpecs:
    # handle resolutions that are too high for USB port to handle
    max_resolution = Resolution(320, 240)
    # OpenCV only allows certain resolutions when recording
    max_recording_resolution = Resolution("240p")
    # Video Encoding, might require additional installs (codecs: http://www.fourcc.org/codecs.php)
    video_type = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    def __init__(self,
        enable_recording: bool = False,
        resolution: Resolution = None,
        fourcc: int = None,
        fps: float = None) -> None:

        # if width and width > self.max_resolution.width:
        #     print(ValueError(f"Video width resolution can't be greater than {self.max_resolution.width}"))
        #     width = None
        # self.width = width if width else Resolution.width
        # if height and height > Resolution.height:
        #     print(ValueError(f"Video height resolution can't be greater than {Resolution.height}"))
        #     height = None

        if resolution:
            if resolution <= self.max_resolution:
                raise ValueError(f"Video resolution must be within {self.max_resolution}; they're currently set to {self.resolution}")
            
            if enable_recording and self.max_recording_resolution < resolution:  # TODO, check!
                raise ValueError(f"")

        self.resolution = resolution if resolution else self.max_resolution
        self.fourcc = fourcc if fourcc else self.video_type['avi']
        self.fps = fps if fps else 24.0


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
spec_1 = CameraSpecs(enable_recording=True, resolution=)
spec_2 = CameraSpecs(enable_recording=True)
thread1 = CameraThread("Camera 1", 1, spec_1)
thread2 = CameraThread("Camera 2", 2, spec_2)
thread1.start()
thread2.start()
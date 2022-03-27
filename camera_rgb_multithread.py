import cv2
import datetime
import threading


def get_timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


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
            progressive_scan = args[0]
            self.width = self.opencv_dimensions[progressive_scan][0]
            self.height = self.opencv_dimensions[progressive_scan][1]
        elif len(args) == 2:
            # assume width and height have been passed in
            self.width = args[0]
            self.height = args[1]
        elif len(args) > 2:
            raise ValueError("Instantiating Resolution: Too many arguments")
 
    def get(self):
        return (self.width, self.height)

    def __le__(self, other: 'Resolution'):
        if other is None:
            return False

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
    max_resolution = Resolution(320, 320)
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

        self.enable_recording = enable_recording
        if self.enable_recording and (not resolution <= self.max_recording_resolution):
            print(ValueError(f"Recording in OpenCV is only allowed at certain resolutions. The current {resolution} is not one of them."))
            print(f"changing resolution to max allowed recording resolution {self.max_recording_resolution}...")
            resolution = self.max_recording_resolution
        elif not (resolution <= self.max_resolution):
            print(ValueError(f"Video resolution must be within {self.max_resolution}; they're currently set to {resolution}."))
            print(f"changing resolution to max allowed resolution {self.max_resolution}...")
            resolution = self.max_resolution

        self.resolution = resolution
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
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, specs.resolution.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, specs.resolution.height)
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
        elif key == 32 and specs.enable_recording:  # space-bar: toggle stream recording
            if not is_recording:
                # transition to recording
                stream_name = f"{timestamp}_cam_{camera_id}.avi"
                print(f"Started recording to {stream_name}...")
                recording = cv2.VideoWriter(stream_name, specs.fourcc, specs.fps, (specs.resolution.width, specs.resolution.height))
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

if __name__=='__main__':
    # create camera threads
    res = Resolution('240p')
    spec_1 = CameraSpecs(enable_recording=False, resolution=res)
    thread1 = CameraThread("Camera 1", 1, spec_1)
    thread1.start()
    spec_2 = CameraSpecs(enable_recording=True, resolution=res)
    thread2 = CameraThread("Camera 2", 2, spec_2)
    thread2.start()

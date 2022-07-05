import cv2
from config import get_timestamp, Resolution


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
        resolution = resolution if resolution else self.max_recording_resolution
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

class Camera:
    def __init__(self, camera_id, specs=None) -> None:
        self.id = camera_id
        self.specs = specs if specs else CameraSpecs()
    
    def start(self):
        camera = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.specs.resolution.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.specs.resolution.height)
        self.camera = camera
    
    def is_on(self):
        return self.camera.isOpened()
    
    def stop(self):
        if self.camera.isOpened():
            self.camera.release()

if __name__=='__main__':
    cam_left = Camera(camera_id=0)
    cam_right = Camera(camera_id=1)

    # Open both cameras
    cam_left.start()
    cam_right.start()

    is_recording = False
    while(cam_right.is_on() and cam_left.is_on()):
        timestamp = get_timestamp()

        succes_left, frame_left = cam_left.camera.read()
        succes_right, frame_right = cam_right.camera.read()

        # Show the frames
        cv2.imshow(f"left: camera {cam_left.id}", frame_left)
        cv2.imshow(f"right: camera {cam_right.id}", frame_right) 

        key = cv2.waitKey(20)
        if key == 27:  # ESC: exit
            break
        elif key == ord('s'):
            img_name = f"{timestamp}_cam_{cam_left.id}.jpg"
            print(f"Attempting to write frame to {img_name}...")
            cv2.imwrite(img_name, frame_left)
            print(f"{img_name} saved successfully")

            img_name = f"{timestamp}_cam_{cam_right.id}.jpg"
            print(f"Attempting to write frame to {img_name}...")
            cv2.imwrite(img_name, frame_right)
            print(f"{img_name} saved successfully")
        elif key == 32:  # space-bar: toggle stream recording_left
            if not is_recording:
                # transition to recording_left
                stream_name = f"{timestamp}_cam_{cam_left.id}.avi"
                print(f"Started recording_left to {stream_name}...")
                recording_left = cv2.VideoWriter(stream_name, cam_left.specs.fourcc, cam_left.specs.fps, (cam_left.specs.resolution.width, cam_left.specs.resolution.height))
                is_recording = True

                stream_name = f"{timestamp}_cam_{cam_right.id}.avi"
                print(f"Started recording_right to {stream_name}...")
                recording_right = cv2.VideoWriter(stream_name, cam_right.specs.fourcc, cam_right.specs.fps, (cam_right.specs.resolution.width, cam_right.specs.resolution.height))
                is_recording = True
            else:
                # stop recording_left
                print(f"Stopped recording_left")
                recording_left.release()
                recording_right.release()
                is_recording = False
        
        if is_recording:
            recording_left.write(frame_left)
            recording_right.write(frame_right)


    # Release and destroy all windows before termination
    cam_left.stop()
    cam_right.stop()

    cv2.destroyAllWindows()
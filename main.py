'''
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2018, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*****************************************************************************************
 ** This sample demonstrates how to capture stereo images and calibration parameters    **
 ** from the ZED camera with OpenCV without using the ZED SDK.                          **
 *****************************************************************************************/
'''

import numpy as np
import os
import argparse
import configparser
import sys
import cv2
import wget
from math import sin, cos, radians, pi
from os.path import join 
from enum import Enum
import websocket
import json
import threading
from datetime import datetime
from threading import Thread
from reachy import Reachy, parts
from reachy.trajectory.player import TrajectoryPlayer
from scipy.spatial.transform import Rotation as R

camera_upside_down = True
rectify = False
capture_files = False
find_circles = True
color_buffer = 90  # +/- this value to filter color in different lighting
circle_coordinates = {
    1: (0, 0),
    2: (0, 0),
    3: (0, 0),
    4: (0, 0),
    5: (0, 0),
    6: (0, 0),
    7: (0, 0),
    8: (0, 0),
    9: (0, 0)
}
distance_to_button = 100

grid_sticker_angles = [
    55,  # 1
    57,  # 2
    64,  # 3
    70,  # 4
    80,  # 5
    90,  # 6
    100,  # 7
    112,  # 8
    122  # 9
]

current_mode = 'play'  # 'play' or 'build'
song_playing = False
current_genre = ''
current_group = 'button_a'
play_songs_group = 'button_e'
reachy_build_song_queue = []
reachy_play_song_queue = []

current_button = 0

button_number_group_mapping = {
    'button_1': ('button_a', 'button_1'),
    'button_2': ('button_a', 'button_2'),
    'button_3': ('button_a', 'button_3'),
    'button_4': ('button_a', 'button_4'),
    'button_5': ('button_b', 'button_1'),
    'button_6': ('button_b', 'button_2'),
    'button_7': ('button_b', 'button_3'),
    'button_8': ('button_b', 'button_4'),
    'button_9': ('button_c', 'button_1'),
    'button_10': ('button_c', 'button_2'),
    'button_11': ('button_c', 'button_3'),
    'button_12': ('button_c', 'button_4'),
    'button_13': ('button_d', 'button_1'),
    'button_14': ('button_d', 'button_2'),
    'button_15': ('button_d', 'button_3'),
    'button_16': ('button_d', 'button_4')
}

song_mapping = {
    'One More Time': 'button_1',
    'Robot Rock': 'button_2',
}

maschine_buttons = ["GROUP_A",
                    "GROUP_B",
                    "GROUP_C",
                    "GROUP_D",
                    "GROUP_E",
                    "GROUP_F",
                    "GROUP_G",
                    "GROUP_H",
                    "SAMPLE_13",
                    "SAMPLE_14",
                    "SAMPLE_15",
                    "SAMPLE_16",
                    "SAMPLE_9",
                    "SAMPLE_10",
                    "SAMPLE_11",
                    "SAMPLE_12",
                    "SAMPLE_5",
                    "SAMPLE_6",
                    "SAMPLE_7",
                    "SAMPLE_8",
                    "SAMPLE_1",
                    "SAMPLE_2",
                    "SAMPLE_3",
                    "SAMPLE_4"]

maschine_button_columns = {  # button ID -> column index 1 through 9
    "GROUP_A": 1,
    "GROUP_B": 2,
    "GROUP_C": 3,
    "GROUP_D": 4,
    "GROUP_E": 1,
    "GROUP_F": 2,
    "GROUP_G": 3,
    "GROUP_H": 4,
    "SAMPLE_13": 6,
    "SAMPLE_14": 7,
    "SAMPLE_15": 8,
    "SAMPLE_16": 9,
    "SAMPLE_9": 6,
    "SAMPLE_10": 7,
    "SAMPLE_11": 8,
    "SAMPLE_12": 9,
    "SAMPLE_5": 6,
    "SAMPLE_6": 7,
    "SAMPLE_7": 8,
    "SAMPLE_8": 9,
    "SAMPLE_1": 6,
    "SAMPLE_2": 7,
    "SAMPLE_3": 8,
    "SAMPLE_4": 9
}

maschine_button_distances = {  # button ID -> approximate distance in pixels from sticker to button
    "GROUP_A": 80,
    "GROUP_B": 79,
    "GROUP_C": 78,
    "GROUP_D": 77,
    "GROUP_E": 65,
    "GROUP_F": 64,
    "GROUP_G": 63,
    "GROUP_H": 62,
    "SAMPLE_13": 104,
    "SAMPLE_14": 105,
    "SAMPLE_15": 109,
    "SAMPLE_16": 117,
    "SAMPLE_9": 86,
    "SAMPLE_10": 86,
    "SAMPLE_11": 90,
    "SAMPLE_12": 94,
    "SAMPLE_5": 66,
    "SAMPLE_6": 67,
    "SAMPLE_7": 68,
    "SAMPLE_8": 72,
    "SAMPLE_1": 40,
    "SAMPLE_2": 41,
    "SAMPLE_3": 43,
    "SAMPLE_4": 46,
}

class RobotMode:
    REAL = 'real'
    SIM = 'sim'

# crude way to work around old way to initialize Reachy
# actual initialization happens in __main__
# TODO: nice to have: refactor so that Reachy is not a global variable that happens on startup
reachy = None

# region Robot methods
def relax(arm):
    assert arm == 'left' or arm == 'right'
    if arm == 'left':
        arm_motors = reachy.left_arm.motors
    elif arm == 'right':
        arm_motors = reachy.right_arm.motors
    # relax all motors into compliant mode for arm
    for m in arm_motors:
        m.compliant = True


def stiffen(arm):
    assert arm == 'left' or arm == 'right'
    if arm == 'left':
        arm_motors = reachy.left_arm.motors
    elif arm == 'right':
        arm_motors = reachy.right_arm.motors
    # relax all motors into compliant mode for arm
    for m in arm_motors:
        m.compliant = False


def goto_arm_joint_solution(arm_choice, joint_solution, duration, wait):
    """
    parameters:
        arm_choice (str): choice of arm to which to give 'go to' instructions
        joint_solution (array): 4x4 array providing joint destination location
        duration (int): time in seconds to get to joint_solution
        wait (bool): whether to wait or not

    Moves `arm_choice` to `joint_solution` within a time frame of `duration` and `wait`s.
    """
    if arm_choice == "left":
        arm_motors = reachy.left_arm.motors
    else:
        arm_motors = reachy.right_arm.motors
    reachy.goto({
        m.name: j
        for j, m in zip(joint_solution, arm_motors)
    }, duration=duration, wait=wait)


right_buttons = ['button_1', 'button_2', 'button_3', 'button_4',
                 'button_5', 'button_6', 'button_7', 'button_8',
                 'button_9', 'button_10', 'button_11', 'button_12',
                 'button_13', 'button_14', 'button_15', 'button_16',
                 'control_choke']
left_buttons = ['button_a', 'button_b', 'button_c', 'button_d',
                'button_e', 'button_f', 'button_g', 'button_h',
                'control_play', 'control_record', 'control_stop'
                                                  'control_shift', 'control_mute', 'control_pattern']


def left_button_press(button_letters):
    global current_group
    for b in button_letters:
        assert b in left_buttons
    for button_letter in button_letters:
        stiffen(arm='left')
        my_loaded_trajectory = np.load(button_letter + '1.npz')
        trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
        trajectory_player.play(wait=True, fade_in_duration=0.4)

        reachy.left_arm.hand.open(end_pos=1, duration=0.3)

        if button_letter in ['button_a', 'button_b', 'button_c', 'button_d',
                             'button_e', 'button_f', 'button_g', 'button_h']:
            current_group = button_letter
        # close left-hand gripper
        reachy.left_arm.hand.close(duration=0.3)

        my_loaded_trajectory = np.load(button_letter + '2.npz')
        trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
        trajectory_player.play(wait=True, fade_in_duration=0.1)
        relax(arm='left')


def play_song(song_name):
    global reachy_moving
    reach_moving = True
    global current_group
    global song_playing
    assert song_name in song_mapping
    button = song_mapping[song_name]
    assert button in right_buttons
    if song_playing:
        choke()

    if current_group != play_songs_group:
        left_button_press([play_songs_group])

    right_button_press([button])
    song_playing = True
    relax(arm='left')
    reachy_moving = False


def select_pattern(button_numbers, hold_button='control_pattern'):
    global current_group
    global reachy_moving
    reachy_moving = True
    for b in button_numbers:
        assert b in right_buttons
    assert hold_button in left_buttons
    for button_number in button_numbers:
        button_group = button_number_group_mapping[button_number][0]
        second_button = button_number_group_mapping[button_number][1]
        print('mapping ' + button_number + ' to group ' + button_group + ' and second button ' + second_button)

        if current_group != button_group:
            left_button_press([button_group])

        stiffen(arm='left')
        if hold_button == 'mute':
            reachy.left_arm.hand.open()
        my_loaded_trajectory = np.load(hold_button + '1.npz')
        trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
        trajectory_player.play(wait=True, fade_in_duration=0.4)

        right_button_press([second_button])

        my_loaded_trajectory = np.load(hold_button + '2.npz')
        trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
        trajectory_player.play(wait=True, fade_in_duration=0.1)

        reachy.left_arm.hand.close()
        relax(arm='left')
    reachy_moving = False


def right_button_press(button_letters):
    for b in button_letters:
        assert b in right_buttons
    for button_letter in button_letters:
        stiffen(arm='right')
        my_loaded_trajectory = np.load(button_letter + '.npz')
        trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
        trajectory_player.play(wait=True, fade_in_duration=0.4)
        relax(arm='right')


# Hold shift + mute = choke
def choke():
    hold_button = 'control_shift'

    stiffen(arm='left')
    my_loaded_trajectory = np.load(hold_button + '1.npz')
    trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
    trajectory_player.play(wait=True, fade_in_duration=0.4)

    right_button_press(['control_choke'])

    my_loaded_trajectory = np.load(hold_button + '2.npz')
    trajectory_player = TrajectoryPlayer(reachy, my_loaded_trajectory)
    trajectory_player.play(wait=True, fade_in_duration=0.1)


def reset_reachy():
    # TODO add any extra reset steps
    global reachy_play_song_queue
    global reachy_build_song_queue
    global song_playing
    global reachy_moving
    reachy_play_song_queue = []
    reachy_build_song_queue = []

    if song_playing:
        reachy_moving = True
        choke()
        reachy_moving = False
    song_playing = False


def change_mode(new_mode):
    global reachy_play_song_queue
    global reachy_build_song_queue
    global current_mode

    # Destroy any remaining items in the queue
    if current_mode != new_mode:
        if current_mode == 'play':
            reachy_play_song_queue = []
        elif current_mode == 'build':
            reachy_build_song_queue = []
    # else do nothing...

    current_mode = new_mode
    reset_reachy()
# endregion

# region Websocket methods
def change_genre(new_genre):
    global current_genre

    current_genre = new_genre


def connect_websocket():
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://3q99jw33n1.execute-api.us-east-1.amazonaws.com/prod",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()


def on_message(ws, message):
    global current_mode
    # print(message)
    body = json.loads(message)

    if 'button' in body:
        button = body['button']
        print('adding button to queue: ' + button)
        reachy_build_song_queue.append(button)
    elif 'session' in body:
        session_action = body['session']
        print('session: ' + str(session_action))
        if session_action == 'stop':
            reset_reachy()
    elif 'mode' in body:
        change_mode(new_mode=body['mode'])
        print('switched to mode: ' + str(current_mode))
    elif 'genre' in body:
        change_genre(new_genre=body['genre'])
        print('switched to genre: ' + str(body['genre']))
    elif 'song' in body:
        reachy_play_song_queue.append(body['song'])
        print('adding song to queue: ' + str(body['song']))
    else:
        print('ERROR: Unknown message received' + str(message))


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    # print('### websocket closed ###')
    print('### Reconnecting Websocket ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' ###')
    connect_websocket()


def on_open(ws):
    print('### websocket opened ###')
# endregion

# region Camera methods
def point_pos(x0, y0, d, theta):
    theta_rad = pi / 2 - radians(theta)
    return int(round(x0 + d * cos(theta_rad))), int(round(y0 + d * sin(theta_rad)))


def download_calibration_file(serial_number):
    if os.name == 'nt':
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else:
        hidden_path = '/usr/local/zed/settings/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url + str(serial_number), out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file


def init_calibration(calibration_file, image_size):
    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208:
        resolution_str = '2K'
    elif image_size.width == 1920:
        resolution_str = 'FHD'
    elif image_size.width == 1280:
        resolution_str = 'HD'
    elif image_size.width == 672:
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_' + resolution_str] if 'TY_' + resolution_str in config['STEREO'] else 0),
                   float(
                       config['STEREO']['TZ_' + resolution_str] if 'TZ_' + resolution_str in config['STEREO'] else 0)])

    left_cam_cx = float(
        config['LEFT_CAM_' + resolution_str]['cx'] if 'cx' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_cy = float(
        config['LEFT_CAM_' + resolution_str]['cy'] if 'cy' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_fx = float(
        config['LEFT_CAM_' + resolution_str]['fx'] if 'fx' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_fy = float(
        config['LEFT_CAM_' + resolution_str]['fy'] if 'fy' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k1 = float(
        config['LEFT_CAM_' + resolution_str]['k1'] if 'k1' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k2 = float(
        config['LEFT_CAM_' + resolution_str]['k2'] if 'k2' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p1 = float(
        config['LEFT_CAM_' + resolution_str]['p1'] if 'p1' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p2 = float(
        config['LEFT_CAM_' + resolution_str]['p2'] if 'p2' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p3 = float(
        config['LEFT_CAM_' + resolution_str]['p3'] if 'p3' in config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k3 = float(
        config['LEFT_CAM_' + resolution_str]['k3'] if 'k3' in config['LEFT_CAM_' + resolution_str] else 0)

    right_cam_cx = float(
        config['RIGHT_CAM_' + resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_cy = float(
        config['RIGHT_CAM_' + resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_fx = float(
        config['RIGHT_CAM_' + resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_fy = float(
        config['RIGHT_CAM_' + resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k1 = float(
        config['RIGHT_CAM_' + resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k2 = float(
        config['RIGHT_CAM_' + resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p1 = float(
        config['RIGHT_CAM_' + resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p2 = float(
        config['RIGHT_CAM_' + resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p3 = float(
        config['RIGHT_CAM_' + resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k3 = float(
        config['RIGHT_CAM_' + resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_' + resolution_str] else 0)

    R_zed = np.array(
        [float(config['STEREO']['RX_' + resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
         float(config['STEREO']['CV_' + resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
         float(config['STEREO']['RZ_' + resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                                  [0, left_cam_fy, left_cam_cy],
                                  [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                                   [0, right_cam_fy, right_cam_cy],
                                   [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1,
                                                         (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2,
                                                           (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution:
    width = 1280
    height = 720
# endregion

# region Instrument visualizations
class Instruments(Enum):
    DRUMS = 1  # 1-4
    BASS = 2  # 5-8
    KEYBOARD = 3  # 9-12
    VOCALS = 4  # 13-16

IMG_ROOT = './img'
ICON_PATHS = {
    Instruments.DRUMS: join(IMG_ROOT, 'drums.png'),
    Instruments.BASS: join(IMG_ROOT, 'bass.png'),
    Instruments.KEYBOARD: join(IMG_ROOT, 'keyboard.png'),
    Instruments.VOCALS: join(IMG_ROOT, 'vocals.png'),
}
def get_instrument_number(num):
    """
    formula to get the right instrument based on button pressed
    """
    num = int(num)
    if num < 1 or num > 16:
        raise ValueError("No instruments found at number {num}")

    return (num-1) // 4 + 1
# endregion

def main(args, robot):
    global color_buffer
    global current_button
    global circle_coordinates
    global distance_to_button
    global reachy_moving

    # Open the ZED camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0:
        exit(-1)

    image_size = Resolution()
    image_size.width = 1280
    image_size.height = 720

    # Set the video resolution to HD720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)

    if args.camera_mode == RobotMode.REAL:
        calibration_file = args.camera_config_path
        if args.camera_config_path is None:
            serial_number = args.camera_id
            calibration_file = download_calibration_file(serial_number)
            if calibration_file == "":
                print("No camera calibration file found. Exiting.")
                exit(1)
        print("Calibration file found. Loading...")

        camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(
            calibration_file, image_size)

    i = 0
    while True:
        # region visualization of incoming tasks
        # TODO: ask Tom about play vs build song modes
        queue = reachy_play_song_queue if len(reachy_play_song_queue) > 0 else reachy_build_song_queue

        if len(queue) > 0:
            button_num = queue[0].split("_")[-1]
            current_instrument = Instruments(get_instrument_number(button_num))
            img = cv2.imread(ICON_PATHS[current_instrument])
            window_name = 'Instrumentation'

            # Write text over image
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale              = .75
            fontColor              = (0, 0, 0)
            thickness              = 2
            lineType               = 2

            text = f'{current_mode}ing: {current_instrument.name}'
            cv2.putText(img, text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            
            if len(queue) > 1:
                # then there in something that's next in the queue
                next_button_num = queue[1].split("_")[-1]
                next_instrument = Instruments(get_instrument_number(next_button_num))
                text = f'next: {next_instrument.name}'
                bottomLeftCornerOfText = (10, 290)
            
                cv2.putText(img, text, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

            # Display the image
            cv2.imshow(window_name, img)
        # endregion

        if robot:
            # check queue
            if not reachy_moving and current_mode == 'play' and len(reachy_play_song_queue) > 0:
                reachy_moving = True
                next_item = reachy_play_song_queue.pop(0)
                print('playing next song in queue: ' + next_item + ', remaining queue items: ' + ', '.join(reachy_play_song_queue))
                Thread(target=play_song, args=(next_item,)).start()
            elif not reachy_moving and current_mode == 'build' and len(reachy_build_song_queue) > 0:
                reachy_moving = True
                next_item = reachy_build_song_queue.pop(0)
                print('pressing button in queue: ' + next_item + ', remaining queue items: ' + ', '.join(reachy_build_song_queue))
                Thread(target=select_pattern, args=([next_item],)).start()

        if args.camera_mode == RobotMode.REAL:
            # region camera handling and computer vision
            # Get a new frame from camera
            retval, frame = cap.read()
            # Extract left and right images from side-by-side
            left_right_image = np.split(frame, 2, axis=1)
            if camera_upside_down:
                left_image_raw = cv2.flip(left_right_image[0], -1)
                right_image_raw = cv2.flip(left_right_image[1], -1)
            else:
                left_image_raw = left_right_image[0]
                right_image_raw = left_right_image[1]

            # Display images
            cv2.imshow("left RAW", left_image_raw)
            cv2.imshow("right RAW", right_image_raw)

            if rectify:
                left_rect = cv2.remap(left_image_raw, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_image_raw, map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

                cv2.imshow("left RECT", left_rect)
                cv2.imshow("right RECT", right_rect)

                if capture_files and i > 90:
                    cv2.imwrite('left_raw2.jpg', left_image_raw)
                    cv2.imwrite('right_raw2.jpg', right_image_raw)
                    cv2.imwrite('left_rect2.jpg', left_rect)
                    cv2.imwrite('right_rect2.jpg', right_rect)
                    break

            if find_circles:
                image = left_image_raw
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # fontScale
                fontScale = 1

                # Blue color in BGR
                color = (255, 0, 0)
                color_arm = (0, 0, 255)

                # Line thickness of 2 px
                thickness = 2

                board_min_y = args.board_edge_bottom
                board_max_y = args.board_edge_top
                arm_min_y = 0
                arm_max_y = 90
                y = 173
                x = 0
                h = 200
                w = 670
                lower_red = np.array([1, 200, 70])
                upper_red = np.array([15, 255, 175])
                image = image[y:y + h, x:x + w]

                import heapq

                def closest_points(list_of_tuples, x_value, n=9):
                    return heapq.nsmallest(n, list_of_tuples, lambda pnt: abs(pnt[0] - x_value))

                output = image.copy()

                # RGB
                # bottom = [5, 20, 76]
                # upper = [21, 60, 167]

                # HSV
                bottom = [2, 207, 80]
                upper = [9, 243, 166]

                bottom = [max(bottom[0] - color_buffer, 0), max(bottom[1] - color_buffer, 0),
                        max(bottom[2] - color_buffer, 0)]
                upper = [min(upper[0] + color_buffer, 179), min(upper[1] + color_buffer, 255),
                        min(upper[2] + color_buffer, 255)]
                # print(bottom)
                # print(upper)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask_red = cv2.inRange(hsv, np.array(bottom), np.array(upper))
                res_red = cv2.bitwise_and(image, image, mask=mask_red)
                gray_masked = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
                cv2.imshow('res_red', res_red)
                cv2.imshow('gray_masked', gray_masked)
                # cv2.waitKey(0)

                # detect circles in the image
                circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT,
                                        minDist=33,
                                        dp=1.1,
                                        param1=130,
                                        param2=8,
                                        minRadius=4,
                                        maxRadius=12)
                # ensure at least some circles were found
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    board_circles = [(x, y, r) for (x, y, r) in circles if board_min_y <= y <= board_max_y]
                    board_circles = closest_points(list_of_tuples=board_circles, x_value=278, n=9)
                    board_circles = sorted(
                        [(i[0], i[1], i[2]) for i in board_circles if board_min_y <= i[1] <= board_max_y],
                        key=lambda l: l[0])
                    # loop over the (x, y) coordinates and radius of the circles
                    i = 1
                    if len(board_circles) == 9:
                        for (x, y, r) in board_circles:
                            # draw the circle in the output image, then draw a rectangle
                            # corresponding to the center of the circle
                            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                            # print(str(x) + ',' + str(y) + ': ' + str(image[y,x]))
                            cv2.putText(output, str(i), (x - 10, y + 40), font, fontScale, color, thickness, cv2.LINE_AA)
                            circle_coordinates[i] = (x,y)
                            # x1, y1 = point_pos(x0=x, y0=y, d=100, theta=grid_sticker_angles[i - 1] + 90)
                            # cv2.line(output, (x, y), (x1, y1), 255, 2)
                            i += 1
                    current_button_str = maschine_buttons[current_button]
                    column_num = maschine_button_columns[current_button_str]
                    origin_coordinates = circle_coordinates[column_num]
                    x0 = origin_coordinates[0]
                    y0 = origin_coordinates[1]
                    button_distance = maschine_button_distances[current_button_str]
                    # button_distance = distance_to_button
                    theta = grid_sticker_angles[column_num - 1] + 90
                    x1, y1 = point_pos(x0=x0, y0=y0, d=button_distance, theta=theta)
                    cv2.line(output, (origin_coordinates[0], origin_coordinates[1]), (x1, y1), 255, 2)

                    arm_circles = sorted([(i[0], i[1], i[2]) for i in circles if arm_min_y <= i[1] <= arm_max_y],
                                        key=lambda l: l[0])
                    for (x, y, r) in arm_circles:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle
                        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 255), -1)
                    if len(arm_circles) == 4:
                        arm_l1 = [arm_circles[0][0], arm_circles[0][1]]
                        arm_l2 = [arm_circles[1][0], arm_circles[1][1]]
                        arm_r1 = [arm_circles[3][0], arm_circles[3][1]]
                        arm_r2 = [arm_circles[2][0], arm_circles[2][1]]

                        cv2.line(output, (arm_l1[0], arm_l1[1]), (arm_l2[0], arm_l2[1]), 255, 2)
                        cv2.line(output, (arm_r1[0], arm_r1[1]), (arm_r2[0], arm_r2[1]), 255, 2)

                        cv2.putText(output, 'L1', (arm_l1[0], arm_l1[1]), font, fontScale, color_arm, thickness,
                                    cv2.LINE_AA)
                        cv2.putText(output, 'L2', (arm_l2[0], arm_l2[1]), font, fontScale, color_arm, thickness,
                                    cv2.LINE_AA)
                        cv2.putText(output, 'R1', (arm_r1[0], arm_r1[1]), font, fontScale, color_arm, thickness,
                                    cv2.LINE_AA)
                        cv2.putText(output, 'R2', (arm_r2[0], arm_r2[1]), font, fontScale, color_arm, thickness,
                                    cv2.LINE_AA)

                    else:
                        # print('Cannot find orange arm dot coordinates')
                        ok = True
                    # show the output image
                    cv2.imshow("output", np.hstack([image, output]))
                    # cv2.waitKey(0)
            key = cv2.waitKey(30)
            if key == 105:  # i on keyboard
                color_buffer += 5
                print('color_buffer' + str(color_buffer))
            elif key == 108:  # l on keyboard
                color_buffer -= 5
                print('color_buffer' + str(color_buffer))
            elif key == 106:  # j on keyboard
                distance_to_button += 1
                print('distance ' + str(distance_to_button))
            elif key == 107:  # k on keyboard
                distance_to_button -= 1
                print('distance ' + str(distance_to_button))
            elif key == 116:  # t on keyboard
                if current_button == len(maschine_buttons) - 1:
                    current_button = 0
                else:
                    current_button += 1

                current_button_str = maschine_buttons[current_button]
                column_num = maschine_button_columns[current_button_str]
                origin_coordinates = circle_coordinates[column_num]
                print(current_button_str)
                print(column_num)
                print(origin_coordinates)
            elif key >= 0:
                break

            i += 1
            # endregion

    exit(0)


if __name__ == "__main__":
    cli_commands = {
        'robot_mode': {
            'default': RobotMode.SIM,
            'type': str,
            'help': "Mode to run the robot in. Options: 'SIM' for simulator and 'REAL' for real hardware."
        },
        'camera_mode': {
            'default': RobotMode.REAL,
            'type': str,
            'help': "Mode to run the camera in. Options: Use 'SIM' when no camera is available " \
                "(which currently just ignorse camera part of code, but might add in preloaded stream someday)" \
                "and 'REAL' for when real camera is available."
        },
        'camera_id': {
            'default': '15618',
            'type': int,
            'help': 'Camera serial number.'
        },
        'camera_config_path': {
            'default': './SN15618.conf',
            'type': str,
            'help': 'Path to camera config file if manually supplying one.'
        },
        'board_edge_top': {
            'default': 200,
            'type': int,
            'help': '(Max Y-value): Pixel value  of the top edge of the board face with the orange circles.'
        },
        'board_edge_bottom': {
            'default': 90,
            'type': int,
            'help': '(Min Y-value): Pixel value of the bottom edge of the board face with the orange circles.'
        },
    }
    parser = argparse.ArgumentParser(description='Input for Reachy with Zed camera.')
    for command, values in cli_commands.items():
        parser.add_argument(
            f"--{command}",
            default=values['default'],
            type=values['type'],
            help=values['help']
        )
    args = parser.parse_args()

    connect_websocket()
    if args.robot_mode == RobotMode.REAL:
        if reachy is None:
            reachy = Reachy(
                right_arm=parts.RightArm(
                    io='/dev/ttyUSB*',
                    hand='force_gripper',
                ),
                left_arm=parts.LeftArm(
                io='/dev/ttyUSB*',
                hand='force_gripper')
            )
        stiffen(arm='left')
        stiffen(arm='right')
        relax(arm='left')
        relax(arm='right')
    main(args, reachy)

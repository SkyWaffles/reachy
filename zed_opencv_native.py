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
import configparser
import sys
import cv2
import wget
import time
from math import sin, cos, radians, pi

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

current_button = 0

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


def main():
    global color_buffer
    global current_button
    global circle_coordinates
    global distance_to_button

    if len(sys.argv) == 1:
        print('Please provide ZED serial number')
        exit(1)

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

    serial_number = int(sys.argv[1])
    calibration_file = download_calibration_file(serial_number)
    if calibration_file == "":
        exit(1)
    print("Calibration file found. Loading...")

    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(
        calibration_file, image_size)
    i = 0
    while True:
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

            board_min_y = 90
            board_max_y = 200
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

            # print('607, 105')
            # x = 607
            # y = 105
            # print(hsv[y][x])
            # cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
            #
            # x = 561
            # y = 112
            # print('561, 112')
            # print(hsv[y][x])
            # cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
            #
            # x = 646
            # y = 104
            # print('646, 104')
            # print(hsv[y][x])
            # cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

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

    exit(0)


if __name__ == "__main__":
    main()

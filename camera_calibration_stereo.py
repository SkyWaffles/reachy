import numpy as np
import cv2 as cv
import glob
import camera_rgb_multithread

MODE = 'FULL'
CHESSBOARD_DIM = (4, 3)
FRAME_DIM = camera_rgb_multithread.Resolution('240p').get()
LEFT_CAMERA = 'cam_0'
RIGHT_CAMERA = 'cam_1'

if MODE == 'TEST':
    left_image = 'img_calibration/2022_03_10_16_25_40_cam_1.jpg'
    left_color = cv.imread(left_image)
    left_gray = cv.cvtColor(left_color, cv.COLOR_BGR2GRAY)
    while True:
        cv.imshow('img left', left_gray)

        key = cv.waitKey(20)

        if key == 27:  # ESC: exit
            break
        
elif MODE == 'FULL':
    # FIND CHESSBOARD CORNERS - 3D point clound and 2D image coordinates ----------------------------------
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    point_cloud = np.zeros((CHESSBOARD_DIM[0] * CHESSBOARD_DIM[1], 3), np.float32)
    point_cloud[:,:2] = np.mgrid[0:CHESSBOARD_DIM[0],0:CHESSBOARD_DIM[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    point_clouds = [] # 3d point in real world space
    left_coords = [] # 2d points in image plane.
    right_coords = [] # 2d points in image plane.

    left_images = glob.glob(f'img_calibration/2022_03_31/*{LEFT_CAMERA}.jpg')
    right_images = glob.glob(f'img_calibration/2022_03_31/*{RIGHT_CAMERA}.jpg')

    for left_image, right_image in zip(left_images, right_images):

        left_color = cv.imread(left_image)
        right_color = cv.imread(right_image)
        left_gray = cv.cvtColor(left_color, cv.COLOR_BGR2GRAY)
        right_gray = cv.cvtColor(right_color, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        left_found, left_corners = cv.findChessboardCorners(left_gray, CHESSBOARD_DIM, None)
        right_found, right_corners = cv.findChessboardCorners(right_gray, CHESSBOARD_DIM, None)

        # If found, add object points, image points (after refining them)
        if left_found and right_found == True:

            point_clouds.append(point_cloud)

            left_corners = cv.cornerSubPix(left_gray, left_corners, (11,11), (-1,-1), criteria)
            left_coords.append(left_corners)

            right_corners = cv.cornerSubPix(right_gray, right_corners, (11,11), (-1,-1), criteria)
            right_coords.append(right_corners)

            # Draw and display the corners
            cv.drawChessboardCorners(left_color, CHESSBOARD_DIM, left_corners, left_found)
            cv.imshow('img left', left_color)
            cv.drawChessboardCorners(right_color, CHESSBOARD_DIM, right_corners, right_found)
            cv.imshow('img right', right_color)
            cv.waitKey(1000)


    cv.destroyAllWindows()

    # ------------- Camera Calibration -------------------------------------------------

    left_found, left_camera_matrix, left_distortion_coeffs, left_rotation_vectors, left_translation_vectors = \
        cv.calibrateCamera(point_clouds, left_coords, FRAME_DIM, None, None)
    left_height, left_width, left_channels = left_color.shape
    new_left_camera_matrix, left_valid_pixel_roi = \
        cv.getOptimalNewCameraMatrix(left_camera_matrix, left_distortion_coeffs, (left_width, left_height), 1, (left_width, left_height))

    right_found, right_camera_matrix, right_distortion_coeffs, right_rotation_vectors, right_translation_vectors = \
        cv.calibrateCamera(point_clouds, right_coords, FRAME_DIM, None, None)
    right_height, right_width, right_channels = right_color.shape
    new_right_camera_matrix, right_valid_pixel_roi = cv.getOptimalNewCameraMatrix(right_camera_matrix, right_distortion_coeffs, (right_width, right_height), 1, (right_width, right_height))

    # --------- Stereo Calibration ---------------------------------------------------

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # flag to restrict intrinsic camera matrixes so that only rotation, translation, essential, and fundamental matrices are calculated.

    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    stereo_found, new_left_camera_matrix, left_distortion_coeffs, new_right_camera_matrix, right_distortion_coeffs, rotation, translation, essential_matrix, fundamental_matrix = \
        cv.stereoCalibrate(point_clouds, left_coords, right_coords, new_left_camera_matrix, left_distortion_coeffs, new_right_camera_matrix, right_distortion_coeffs, left_gray.shape[::-1], criteria_stereo, flags)

    # --------- Stereo Rectification -------------------------------------------------

    rectifyScale= 1
    left_rect_trans, right_rect_trans, left_project_matrix, right_proj_matrix, Q, left_valid_pixel_roi, right_valid_pixel_roi = \
        cv.stereoRectify(new_left_camera_matrix, left_distortion_coeffs, new_right_camera_matrix, right_distortion_coeffs, left_gray.shape[::-1], rotation, translation, rectifyScale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(new_left_camera_matrix, left_distortion_coeffs, left_rect_trans, left_project_matrix, left_gray.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(new_right_camera_matrix, right_distortion_coeffs, right_rect_trans, right_proj_matrix, right_gray.shape[::-1], cv.CV_16SC2)

    print("Saving parameters!")
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])

    cv_file.release()
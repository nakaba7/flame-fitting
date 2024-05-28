import numpy as np
import matplotlib.pyplot as plt
from Coordinate_convertor import image2camera_coordinates, camera2image_coordinates, camera2world_coordinates, world2camera_coordinates, mm2pixel, pixel2mm

def transform_camera2_to_camera1(camera2_points, R, T):
    # Apply rotation and translation to transform camera 2 points to camera 1 coordinates
    camera1_points = np.dot(R.T, (camera2_points - T.T).T).T
    return camera1_points

def transform_to_world(camera_points, world_R, world_T):
    # Apply rotation and translation to transform camera points to world coordinates
    world_points = np.dot(world_R, camera_points.T).T + world_T.T
    return world_points

camera1_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/mouth/test0_annotated.npy")
camera2_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test0_0_annotated.npy")
camera3_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/righteye/test0_1_annotated.npy")

camera1_z_pixel = mm2pixel(45, 96)#鼻下とカメラの距離
camera2_z_pixel = mm2pixel(30, 96)#左目とカメラの距離
camera3_z_pixel = mm2pixel(30, 96)

camera1_points = image2camera_coordinates(camera1_image_points, camera1_z_pixel, "CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
camera2_points = image2camera_coordinates(camera2_image_points, camera2_z_pixel, "CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
camera3_points = image2camera_coordinates(camera3_image_points, camera3_z_pixel, "CameraCalibration/Parameters/ChessBoard_eye_right_mtx.npy")

# Rotation matrix and translation vector from camera 1 to camera 2 (example values)
#R = np.load("CameraCalibration/Parameters/R_eye_left_mouth_left.npy")
#T = np.load("CameraCalibration/Parameters/T_eye_left_mouth_left.npy")

R_mouth2lefteye = np.load("CameraCalibration/Parameters/R_mouth_left_eye_left.npy")
T_mouth2lefteye = np.load("CameraCalibration/Parameters/T_mouth_left_eye_left.npy")

R_mouth2righteye = np.load("CameraCalibration/Parameters/R_mouth_right_eye_right.npy")
T_mouth2righteye = np.load("CameraCalibration/Parameters/T_mouth_right_eye_right.npy")

# Transform camera 2 points to camera 1 coordinates
camera2_points_in_camera1 = transform_camera2_to_camera1(camera2_points, R_mouth2lefteye, T_mouth2lefteye)
camera3_points_in_camera1 = transform_camera2_to_camera1(camera3_points, R_mouth2righteye, T_mouth2righteye)
# Combine all points in camera 1 coordinates
all_camera1_points = np.vstack((camera1_points, camera2_points_in_camera1))

# 3Dプロットを作成します
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カメラ1のポイントをプロット
ax.scatter(camera1_points[:, 0], camera1_points[:, 1], camera1_points[:, 2], c='b', marker='o', label='Camera 1 Points')

# カメラ2のポイントをプロット（カメラ1座標系に変換済み）
ax.scatter(camera2_points_in_camera1[:, 0], camera2_points_in_camera1[:, 1], camera2_points_in_camera1[:, 2], c='r', marker='^', label='Camera 2 Points in Camera 1')

ax.scatter(camera3_points_in_camera1[:, 0], camera3_points_in_camera1[:, 1], camera3_points_in_camera1[:, 2], c='g', marker='^', label='Camera 3 Points in Camera 1')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

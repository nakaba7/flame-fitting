import numpy as np
import matplotlib.pyplot as plt
from Coordinate_convertor import image2camera_coordinates, mm2pixel
from Convert.Lmk3d_2_2d import lmk3d_2_2d

def transform_camera2_to_camera1(camera2_points, R, T):
    # Apply rotation and translation to transform camera 2 points to camera 1 coordinates
    camera1_points = np.dot(R.T, (camera2_points - T.T).T).T
    return camera1_points

camera_mouth_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/mouth/test0_annotated.npy")
camera_lefteye_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test0_0_annotated.npy")
camera_righteye_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/righteye/test0_1_annotated.npy")

camera_mouth_mtx = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
camera_lefteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
camera_righteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_right_mtx.npy")

camera_mouth_z_pixel = mm2pixel(35, 96)  # 鼻下とカメラの距離
camera_lefteye_z_pixel = mm2pixel(23, 96)  # 左目とカメラの距離
camera_righteye_z_pixel = mm2pixel(23, 96)

camera_mouth_points = image2camera_coordinates(camera_mouth_image_points, camera_mouth_z_pixel, camera_mouth_mtx, True)
camera_lefteye_points = image2camera_coordinates(camera_lefteye_image_points, camera_lefteye_z_pixel, camera_lefteye_mtx, False)
camera_righteye_points = image2camera_coordinates(camera_righteye_image_points, camera_righteye_z_pixel, camera_righteye_mtx, False)

R_mouth2lefteye = np.load("CameraCalibration/Parameters/R_mouth_left_eye_left.npy")
T_mouth2lefteye = np.load("CameraCalibration/Parameters/T_mouth_left_eye_left.npy")

R_mouth2righteye = np.load("CameraCalibration/Parameters/R_mouth_right_eye_right.npy")
T_mouth2righteye = np.load("CameraCalibration/Parameters/T_mouth_right_eye_right.npy")

# Transform camera 2 points to camera 1 coordinates
camera_lefteye_points_in_camera_mouth = transform_camera2_to_camera1(camera_lefteye_points, R_mouth2lefteye, T_mouth2lefteye)
camera_righteye_points_in_camera_mouth = transform_camera2_to_camera1(camera_righteye_points, R_mouth2righteye, T_mouth2righteye)

# Combine all points in camera 1 coordinates
all_camera_mouth_points = np.vstack((camera_mouth_points, camera_lefteye_points_in_camera_mouth, camera_righteye_points_in_camera_mouth))

# 3Dプロットを作成します
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カメラ1のポイントをプロット
ax.scatter(camera_mouth_points[:, 0], camera_mouth_points[:, 1], camera_mouth_points[:, 2], c='b', marker='o', label='Camera 1 Points')

# カメラ2のポイントをプロット（カメラ1座標系に変換済み）
ax.scatter(camera_lefteye_points_in_camera_mouth[:, 0], camera_lefteye_points_in_camera_mouth[:, 1], camera_lefteye_points_in_camera_mouth[:, 2], c='r', marker='^', label='Camera 2 Points in Camera 1')
ax.scatter(camera_righteye_points_in_camera_mouth[:, 0], camera_righteye_points_in_camera_mouth[:, 1], camera_righteye_points_in_camera_mouth[:, 2], c='g', marker='^', label='Camera 3 Points in Camera 1')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

all_camera_mouth_points_2d = lmk3d_2_2d(all_camera_mouth_points)

# 2Dプロットを作成します
plt.figure()
plt.scatter(all_camera_mouth_points_2d[:, 0], all_camera_mouth_points_2d[:, 1], c='b', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Projection of 3D Points (Viewed from Negative Z-axis)')
plt.show()

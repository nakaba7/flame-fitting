import numpy as np
import matplotlib.pyplot as plt
from Coordinate_convertor import image2camera_coordinates, camera2image_coordinates, camera2world_coordinates, world2camera_coordinates, mm2pixel, pixel2mm

def transform_camera2_to_camera1(camera2_points, R, T):
    """
    Transform points from camera 2 coordinates to camera 1 coordinates.
    
    :param camera2_points: Numpy array of shape (N, 3) representing points in camera 2 coordinates.
    :param R: 3x3 rotation matrix from camera 1 to camera 2.
    :param T: 3x1 translation vector from camera 1 to camera 2.
    :return: Transformed points in camera 1 coordinates.
    """
    # Apply rotation and translation to transform camera 2 points to camera 1 coordinates
    camera1_points = np.dot(R.T, (camera2_points - T.T).T).T
    return camera1_points

def transform_to_world(camera_points, world_R, world_T):
    """
    Transform camera points to world coordinates.
    
    :param camera_points: Numpy array of shape (N, 3) representing points in camera coordinates.
    :param world_R: 3x3 rotation matrix from camera to world.
    :param world_T: 3x1 translation vector from camera to world.
    :return: Transformed points in world coordinates.
    """
    # Apply rotation and translation to transform camera points to world coordinates
    world_points = np.dot(world_R, camera_points.T).T + world_T.T
    return world_points

camera1_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test0_0_annotated.npy")
camera2_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/mouth/test0_annotated.npy")

camera1_z_pixel = mm2pixel(35, 96)#35mmは「カメラとフェイスカバーの距離＋5mm」
camera2_z_pixel = mm2pixel(45, 96)#45mmは鼻下とBracketの距離

camera1_points = image2camera_coordinates(camera1_image_points, camera1_z_pixel, "CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
camera2_points = image2camera_coordinates(camera2_image_points, camera2_z_pixel, "CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")

# Rotation matrix and translation vector from camera 1 to camera 2 (example values)
R = np.load("CameraCalibration/Parameters/R_eye_left_mouth_left.npy")
T = np.load("CameraCalibration/Parameters/T_eye_left_mouth_left.npy")
#T*=10
# Transform camera 2 points to camera 1 coordinates
camera2_points_in_camera1 = transform_camera2_to_camera1(camera2_points, R, T)

# Combine all points in camera 1 coordinates
all_camera1_points = np.vstack((camera1_points, camera2_points_in_camera1))

# Optional: Transform to world coordinates
# If the world coordinate system is the same as camera 1 coordinate system, use identity for R and zero for T
world_R = np.eye(3)
world_T = np.zeros((3, 1))

world_points = transform_to_world(all_camera1_points, world_R, world_T)

# 3Dプロットを作成します
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カメラ1のポイントをプロット
ax.scatter(camera1_points[:, 0], camera1_points[:, 1], camera1_points[:, 2], c='b', marker='o', label='Camera 1 Points')

# カメラ2のポイントをプロット（カメラ1座標系に変換済み）
ax.scatter(camera2_points_in_camera1[:, 0], camera2_points_in_camera1[:, 1], camera2_points_in_camera1[:, 2], c='r', marker='^', label='Camera 2 Points in Camera 1')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from Coordinate_convertor import image2camera_coordinates, mm2pixel
import cv2
from Convert.Lmk3d_2_2d import lmk3d_2_2d

def transform_camera2_to_camera1(camera2_points, R, T):
    # Apply rotation and translation to transform camera 2 points to camera 1 coordinates
    camera1_points = np.dot(R.T, (camera2_points - T.T).T).T
    return camera1_points

def lmk_sort(lmk_2d):
    """
    顔の2次元ランドマークをflame-fittingの順番に並び替える関数. add_nose_lmk関数の後に使用する.
    """
    new_lmks = lmk_2d.copy()
    new_lmks[5:10] = lmk_2d[42:47]#左眉毛
    new_lmks[10:14] = lmk_2d[47:51]#鼻上部
    new_lmks[14:19] = lmk_2d[11:16]#鼻下部   
    new_lmks[19:25] = lmk_2d[5:11]#右目
    new_lmks[25:31] = lmk_2d[36:42]#左目
    new_lmks[31:51] = lmk_2d[16:36]#口
    return new_lmks

def add_nose_lmk(presort_lmk_2d):
    """
    左目, 右目, 口のランドマークが合わさった2次元ランドマークに鼻のランドマーク4つを追加する関数.
    Args:
        presort_lmk_2d: ソート前の2次元ランドマークの配列
    """
    # 30番目の点と44番目の点の中点を計算
    midpoint = (presort_lmk_2d[30] + presort_lmk_2d[44]) / 2

    # 中点から13番の方向へ一定間隔で3つの点を追加
    direction = presort_lmk_2d[2] - midpoint
    interval = np.linalg.norm(direction) / 4  # 一定間隔を計算
    unit_direction = direction / np.linalg.norm(direction)

    new_points = [midpoint + unit_direction * interval * (i) for i in range(4)]
    new_points = np.array(new_points)

    # 新しい点をpresort_lmk_2dに追加
    presort_lmk_2d = np.concatenate((presort_lmk_2d, new_points), axis=0)
    return presort_lmk_2d

camera_mouth_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/mouth/a1_annotated.npy")
camera_lefteye_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test6_0_annotated.npy")
camera_righteye_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/righteye/test6_1_annotated.npy")

camera_mouth_mtx = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
camera_lefteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
camera_righteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_right_mtx.npy")

#z座標をmm単位で指定
camera_mouth_z_pixel = 50 
camera_lefteye_z_pixel = 35
camera_righteye_z_pixel = 35

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


# Equal aspect ratio
ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

# Set equal scaling
scaling = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

plt.show()

all_camera_mouth_points_2d = lmk3d_2_2d(all_camera_mouth_points)
all_camera_mouth_points_2d = add_nose_lmk(all_camera_mouth_points_2d)
#all_camera_mouth_points_2d = lmk_sort(all_camera_mouth_points_2d)
"""
rvec = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_rvecs.npy")[3]
tvec = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_tvecs.npy")[3]
print(rvec)
print(tvec)
all_camera_mouth_points_2d, _ = cv2.projectPoints(all_camera_mouth_points, rvec, tvec, camera_mouth_mtx, None)
all_camera_mouth_points_2d = np.squeeze(all_camera_mouth_points_2d, axis=1)
"""
# 2Dプロットを作成します
plt.figure()
plt.scatter(all_camera_mouth_points_2d[:, 0], -all_camera_mouth_points_2d[:, 1], c='b', marker='o')  # y軸を反転

# 各点の近くにインデックス番号を表示
for i, point in enumerate(all_camera_mouth_points_2d):
    plt.text(point[0], -point[1], str(i), fontsize=9, ha='right', va='bottom')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Projection of 3D Points (Viewed from Negative Z-axis)')
plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for 2D plot
plt.show()

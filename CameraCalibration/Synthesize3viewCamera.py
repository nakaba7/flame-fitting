import numpy as np
import cv2
import matplotlib.pyplot as plt

def transform_points_to_world(points, K, R, T):
    """
    カメラ座標系からワールド座標系へ特徴点を変換する関数。
    
    Args:
    - points (numpy.ndarray): カメラ座標系における特徴点の配列 (N, 2)。
    - K (numpy.ndarray): カメラの内的パラメータ行列 (3, 3)。
    - R (numpy.ndarray): カメラの回転行列 (3, 3)。
    - T (numpy.ndarray): カメラの平行移動ベクトル (3, 1)。
    
    Returns:
    - world_points (numpy.ndarray): ワールド座標系における特徴点の配列 (N, 3)。
    """
    # 特徴点を同次座標に変換
    num_points = points.shape[0]
    hom_points = np.hstack([points, np.ones((num_points, 1))])
    
    # 内的パラメータ行列の逆行列を計算
    K_inv = np.linalg.inv(K)
    
    # カメラ座標系からワールド座標系への変換
    world_points = []
    for hom_point in hom_points:
        cam_point = np.dot(K_inv, hom_point)
        cam_point = cam_point / cam_point[2]  # Z=1に正規化
        world_point = np.dot(R.T, cam_point - T.flatten())
        world_points.append(world_point)
    
    return np.array(world_points)

def project_points_to_image(points, K, R, T):
    """
    ワールド座標系の特徴点をカメラの画像平面に投影する関数。
    
    Args:
    - points (numpy.ndarray): ワールド座標系における特徴点の配列 (N, 3)。
    - K (numpy.ndarray): カメラの内的パラメータ行列 (3, 3)。
    - R (numpy.ndarray): カメラの回転行列 (3, 3)。
    - T (numpy.ndarray): カメラの平行移動ベクトル (3, 1)。
    
    Returns:
    - image_points (numpy.ndarray): カメラの画像平面における特徴点の配列 (N, 2)。
    """
    # ワールド座標系からカメラ座標系への変換
    cam_points = np.dot(R, points.T).T + T.T
    
    # カメラ座標系から画像座標系への投影
    hom_cam_points = np.hstack([cam_points[:, :2] / cam_points[:, 2, np.newaxis], np.ones((cam_points.shape[0], 1))])
    image_points = np.dot(K, hom_cam_points.T).T
    return image_points[:, :2]

# カメラの内的パラメータ行列（例）
K_a = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
K_b = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
K_c = np.load("CameraCalibration/Parameters/ChessBoard_eye_right_mtx.npy")


# カメラAとカメラBの相対回転行列と平行移動ベクトル
R_ab = np.load("CameraCalibration/Parameters/R_eye_left_mouth_left.npy")
T_ab = np.load("CameraCalibration/Parameters/T_eye_left_mouth_left.npy")

# カメラBとカメラCの相対回転行列と平行移動ベクトル
R_bc = np.load("CameraCalibration/Parameters/R_eye_right_mouth_right.npy")
T_bc = np.load("CameraCalibration/Parameters/T_eye_right_mouth_right.npy")

# カメラA, B, Cの特徴点（例）
points_cam_a = np.load("FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")
points_cam_b = np.load("FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")
points_cam_c = np.load("FaceData/FaceImages_Annotated/NPYs/test6_1_annotated.npy")

# カメラAの特徴点をワールド座標系に変換
world_points_cam_a = transform_points_to_world(points_cam_a, K_a, np.eye(3), np.zeros((3, 1)))

# カメラCの特徴点をワールド座標系に変換
world_points_cam_c = transform_points_to_world(points_cam_c, K_c, np.eye(3), np.zeros((3, 1)))

# カメラBの特徴点はそのまま使用
world_points_cam_b = transform_points_to_world(points_cam_b, K_b, np.eye(3), np.zeros((3, 1)))

# ワールド座標系の特徴点をカメラBの画像平面に投影
image_points_cam_a_to_b = project_points_to_image(world_points_cam_a, K_b, np.eye(3), np.zeros((3, 1)))
image_points_cam_c_to_b = project_points_to_image(world_points_cam_c, K_b, R_bc.T, -np.dot(R_bc.T, T_bc))

print("World a: ", image_points_cam_a_to_b)
print("World b: ", points_cam_b)
print("World c: ", image_points_cam_c_to_b)

# プロットの設定
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# カメラAの特徴点をプロット
axes[0].scatter(points_cam_a[:, 0], points_cam_a[:, 1], color='r', label='Camera A')
axes[0].set_title('Camera A Coordinates')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].invert_yaxis()  # 画像座標系の上下を反転
axes[0].legend()

# カメラBの特徴点をプロット
axes[1].scatter(points_cam_b[:, 0], points_cam_b[:, 1], color='g', label='Camera B')
axes[1].set_title('Camera B Coordinates')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].invert_yaxis()  # 画像座標系の上下を反転
axes[1].legend()

# カメラCの特徴点をプロット
axes[2].scatter(points_cam_c[:, 0], points_cam_c[:, 1], color='b', label='Camera C')
axes[2].set_title('Camera C Coordinates')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].invert_yaxis()  # 画像座標系の上下を反転
axes[2].legend()

# カメラBの座標系に変換後の特徴点をプロット
axes[3].scatter(image_points_cam_a_to_b[:, 0], image_points_cam_a_to_b[:, 1], color='r', label='Camera A (to B)')
axes[3].scatter(points_cam_b[:, 0], points_cam_b[:, 1], color='g', label='Camera B')
axes[3].scatter(image_points_cam_c_to_b[:, 0], image_points_cam_c_to_b[:, 1], color='b', label='Camera C (to B)')
axes[3].set_title('Transformed to Camera B Coordinates')
axes[3].set_xlabel('X')
axes[3].set_ylabel('Y')
axes[3].invert_yaxis()  # 画像座標系の上下を反転
axes[3].legend()

plt.tight_layout()
plt.show()

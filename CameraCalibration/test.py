import numpy as np
import matplotlib.pyplot as plt

def transform_points(points, K, R, T):
    """
    カメラ座標系から他のカメラ座標系へ特徴点を変換する関数。
    
    Args:
    - points (numpy.ndarray): カメラ座標系における特徴点の配列 (N, 2)。
    - K (numpy.ndarray): カメラの内的パラメータ行列 (3, 3)。
    - R (numpy.ndarray): 相対回転行列 (3, 3)。
    - T (numpy.ndarray): 相対平行移動ベクトル (3, 1)。
    
    Returns:
    - transformed_points (numpy.ndarray): 他のカメラ座標系における特徴点の配列 (N, 3)。
    """
    # 特徴点を同次座標に変換
    num_points = points.shape[0]
    hom_points = np.hstack([points, np.ones((num_points, 1))])
    
    # 内的パラメータ行列の逆行列を計算
    K_inv = np.linalg.inv(K)
    
    # カメラ座標系から他のカメラ座標系への変換
    transformed_points = []
    for hom_point in hom_points:
        cam_point = np.dot(K_inv, hom_point)
        transformed_point = np.dot(R, cam_point) + T.flatten()
        transformed_points.append(transformed_point)
    
    return np.array(transformed_points)

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

# カメラAの特徴点をカメラBの座標系に変換
R_a_to_b = R_ab
T_a_to_b = T_ab
transformed_points_cam_a = transform_points(points_cam_a, K_a, R_a_to_b, T_a_to_b)

# カメラCの特徴点をカメラBの座標系に変換
R_c_to_b = R_bc.T
T_c_to_b = -np.dot(R_bc.T, T_bc)
transformed_points_cam_c = transform_points(points_cam_c, K_c, R_c_to_b, T_c_to_b)

print("World a: ", transformed_points_cam_a)
print("World b: ", points_cam_b)
print("World c: ", transformed_points_cam_c)

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
axes[3].scatter(transformed_points_cam_a[:, 0], transformed_points_cam_a[:, 1], color='r', label='Camera A (to B)')
axes[3].scatter(points_cam_b[:, 0], points_cam_b[:, 1], color='g', label='Camera B')
axes[3].scatter(transformed_points_cam_c[:, 0], transformed_points_cam_c[:, 1], color='b', label='Camera C (to B)')
axes[3].set_title('Transformed to Camera B Coordinates')
axes[3].set_xlabel('X')
axes[3].set_ylabel('Y')
axes[3].legend()

plt.tight_layout()
plt.show()

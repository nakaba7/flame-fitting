import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_camera_parameters(camera_name):
    mtx = np.load(f"Parameters/ChessBoard_{camera_name}_mtx.npy")
    dist = np.load(f"Parameters/ChessBoard_{camera_name}_dist.npy")
    return mtx, dist

def load_stereo_parameters(camera0, camera1):
    R = np.load(f"Parameters/R_{camera0}_{camera1}.npy")
    T = np.load(f"Parameters/T_{camera0}_{camera1}.npy")
    return R, T

def undistort_points(points, mtx, dist):
    points = np.array(points, dtype=np.float32)
    points = points.reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(points, mtx, dist, P=mtx)
    return undistorted_points.reshape(-1, 2)

def transform_points(points, R, T):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(R, points.T).T + T.T
    return transformed_points[:, :2]

def project_points(points, mtx):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    projected_points = np.dot(mtx, points.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    return projected_points

def main():
    # カメラ名を指定
    camera0 = 'eye_left'
    camera1 = 'mouth'
    camera2 = 'eye_right'
    
    # カメラパラメータを読み込む
    mtx0, dist0 = load_camera_parameters(camera0)
    mtx1, dist1 = load_camera_parameters(camera1+"_left")
    mtx2, dist2 = load_camera_parameters(camera2)
    
    # ステレオキャリブレーション結果を読み込む
    R_ab, T_ab = load_stereo_parameters(camera0, camera1+"_left")
    R_bc, T_bc = load_stereo_parameters(camera1+"_right", camera2)
    
    # 特徴点の読み込み
    points_cam_a = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")
    points_cam_b = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")
    points_cam_c = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_1_annotated.npy")
    
    # 歪み補正
    undistorted_points_cam_a = undistort_points(points_cam_a, mtx0, dist0)
    undistorted_points_cam_b = undistort_points(points_cam_b, mtx1, dist1)
    undistorted_points_cam_c = undistort_points(points_cam_c, mtx2, dist2)
    
    # カメラAの特徴点をカメラBの座標系に変換
    transformed_points_a_to_b = transform_points(undistorted_points_cam_a, R_ab, T_ab)

    # カメラCの特徴点をカメラBの座標系に変換
    transformed_points_c_to_b = transform_points(undistorted_points_cam_c, np.linalg.inv(R_bc), -np.dot(np.linalg.inv(R_bc), T_bc))

    # プロットの設定
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # カメラAの特徴点をプロット
    axes[0].scatter(points_cam_a[:, 0], points_cam_a[:, 1], color='r', label='Camera A')
    axes[0].set_title('Camera left Coordinates')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].invert_yaxis()  # 画像座標系の上下を反転
    axes[0].legend()

    # カメラBの特徴点をプロット
    axes[1].scatter(points_cam_b[:, 0], points_cam_b[:, 1], color='g', label='Camera B')
    axes[1].set_title('Camera mouth Coordinates')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].invert_yaxis()  # 画像座標系の上下を反転
    axes[1].legend()

    # カメラCの特徴点をプロット
    axes[2].scatter(points_cam_c[:, 0], points_cam_c[:, 1], color='b', label='Camera C')
    axes[2].set_title('Camera right Coordinates')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].invert_yaxis()  # 画像座標系の上下を反転
    axes[2].legend()

    # カメラBの座標系に変換後の特徴点をプロット
    axes[3].scatter(transformed_points_a_to_b[:, 0], transformed_points_a_to_b[:, 1], color='r', label='Camera left (to mouth)')
    axes[3].scatter(points_cam_b[:, 0], points_cam_b[:, 1], color='g', label='Camera mouth')
    axes[3].scatter(transformed_points_c_to_b[:, 0], transformed_points_c_to_b[:, 1], color='b', label='Camera right (to mouth)')
    axes[3].set_title('Transformed to Camera B Coordinates')
    axes[3].set_xlabel('X')
    axes[3].set_ylabel('Y')
    axes[3].invert_yaxis()  # 画像座標系の上下を反転
    axes[3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

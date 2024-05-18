import numpy as np
import cv2
import matplotlib.pyplot as plt

def transform_2d_points(points, K1, K2, R, T):
    """
    片方のカメラから見た2次元座標をもう片方のカメラから見た座標へ変換する関数。

    Args:
    - points (numpy.ndarray): カメラ1での2次元座標の配列 (N, 2)。
    - K1 (numpy.ndarray): カメラ1の内部パラメータ行列 (3, 3)。
    - K2 (numpy.ndarray): カメラ2の内部パラメータ行列 (3, 3)。
    - R (numpy.ndarray): カメラ1からカメラ2への回転行列 (3, 3)。
    - T (numpy.ndarray): カメラ1からカメラ2への平行移動ベクトル (3, 1)。

    Returns:
    - transformed_points (numpy.ndarray): カメラ2での2次元座標の配列 (N, 2)。
    """
    # 内部パラメータ行列の逆行列を計算
    K1_inv = np.linalg.inv(K1)

    # 同次座標系に変換
    num_points = points.shape[0]
    hom_points = np.hstack([points, np.ones((num_points, 1))]).T

    # カメラ1の画像座標をカメラ1のカメラ座標に変換
    cam1_points = K1_inv @ hom_points

    # カメラ1のカメラ座標をカメラ2のカメラ座標に変換
    cam2_points = R @ cam1_points + T

    # カメラ2のカメラ座標をカメラ2の画像座標に変換
    hom_cam2_points = K2 @ cam2_points

    # 正規化して2次元座標に変換
    transformed_points = (hom_cam2_points[:2] / hom_cam2_points[2]).T

    return transformed_points

def main():
    # カメラキャリブレーションデータの読み込み
    K1 = np.load("Parameters/ChessBoard_eye_left_mtx.npy")
    K2 = np.load("Parameters/ChessBoard_mouth_left_mtx.npy")
    R = np.load("Parameters/R_eye_left_mouth_left.npy")
    T = np.load("Parameters/T_eye_left_mouth_left.npy")

    # 変換したい特徴点の読み込み
    points = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")

    # a.npyファイルの特徴点を読み込み
    a_points = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")  # (N, 2)

    # 座標変換
    transformed_points = transform_2d_points(points, K1, K2, R, T)

    # 結果を表示
    print("Original Points:\n", points)
    print("Transformed Points:\n", transformed_points)
    print("a.npy Points:\n", a_points)

    # プロットの設定
    plt.figure(figsize=(10, 5))

    # カメラ1の特徴点をプロット
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], color='r', label='Camera 1 Points')
    plt.title('Camera 1 Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # 画像座標系の上下を反転
    plt.legend()

    # 変換後のカメラ2の特徴点をプロット
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='b', label='Transformed to Camera 2')
    plt.scatter(a_points[:, 0], a_points[:, 1], color='g', label='a.npy Points')
    plt.title('Transformed to Camera 2 Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # 画像座標系の上下を反転
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

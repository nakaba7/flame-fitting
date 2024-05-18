import numpy as np
import matplotlib.pyplot as plt

def scale_points(points, image_height, image_width, scale_factor):
    """
    特徴点を拡大縮小する関数

    Parameters:
    - points: np.ndarray, 特徴点の座標を含む2次元配列
    - image_height: int, 画像の高さ
    - image_width: int, 画像の幅
    - scale_factor: float, 拡大縮小の倍率

    Returns:
    - scaled_points: np.ndarray, 拡大縮小された特徴点の座標を含む2次元配列
    """
    # 画像の中心
    center_x = image_width / 2
    center_y = image_height / 2

    # 特徴点を拡大縮小
    scaled_points = (points - np.array([center_x, center_y])) * scale_factor + np.array([center_x, center_y])
    
    return scaled_points

if __name__ == '__main__':
    # 例として関数を呼び出す
    image_height = 1232
    image_width = 1640
    scale_factor = 4.5 / 2.5

    # 特徴点を読み込む
    points = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")

    # 特徴点を拡大縮小
    scaled_points = scale_points(points, image_height, image_width, scale_factor)

    # プロット
    plt.figure(figsize=(20, 10))

    # 元の特徴点
    plt.subplot(1, 2, 1)
    plt.imshow(np.zeros((image_height, image_width)))
    plt.scatter(points[:, 0], points[:, 1], c='b', label='Original Points')
    for i, point in enumerate(points):
        plt.text(point[0], point[1], str(i), color="white", fontsize=12, ha='right')
    plt.title('Original Points')
    plt.legend()

    # 拡大縮小後の特徴点
    plt.subplot(1, 2, 2)
    plt.imshow(np.zeros((image_height, image_width)))
    plt.scatter(scaled_points[:, 0], scaled_points[:, 1], c='r', label='Scaled Points')
    for i, point in enumerate(scaled_points):
        plt.text(point[0], point[1], str(i), color="white", fontsize=12, ha='right')
    plt.title('Scaled Points')
    plt.legend()

    plt.show()

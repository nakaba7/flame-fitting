import cv2
import numpy as np

def undistort_image(image, mtx, dist):
    """
    画像の歪み補正を行う関数。

    Parameters:
    image (numpy.ndarray): 入力画像
    mtx (numpy.ndarray): カメラ行列
    dist (numpy.ndarray): 歪み係数

    Returns:
    numpy.ndarray: 歪み補正後の画像
    """
    # 歪み補正のための新しいカメラ行列とROIを取得
    h, w = image.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 歪み補正を実行
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_camera_mtx)

    # ROIでトリミング（オプション）
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image

# 使用例
if __name__ == "__main__":
    # カメラ行列と歪み係数の例
    mtx = np.load("Parameters/ChessBoard_eye_left_mtx.npy")
    dist = np.load("Parameters/ChessBoard_eye_left_dist.npy")

    # 入力画像の読み込み
    image = cv2.imread(f'../FaceImages/Nakabayashi/lefteye/test3_0.jpg')

    # 歪み補正を行う
    undistorted_image = undistort_image(image, mtx, dist)

    # 補正後の画像を保存または表示
    #cv2.imwrite("undistorted_image.jpg", undistorted_image)
    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

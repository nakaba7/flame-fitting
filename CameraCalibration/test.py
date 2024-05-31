import cv2
import numpy as np

def undistort_img(img, mtx, dist):
    # 歪み補正のための新しいカメラ行列とROIを取得
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 歪み補正を実行
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # ROIでトリミング（オプション）
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    return undistorted_img

# 使用例
if __name__ == "__main__":
    # カメラ行列と歪み係数の例
    mtx = np.load("Parameters/ChessBoard_eye_left_mtx.npy")
    dist = np.load("Parameters/ChessBoard_eye_left_dist.npy")

    # 入力画像の読み込み
    img = cv2.imread(f'../FaceImages/Nakabayashi/lefteye/test0_0.jpg')

    # 歪み補正を行う
    undistorted_img = undistort_img(img, mtx, dist)

    # 補正後の画像を保存または表示
    #cv2.imwrite("undistorted_img.jpg", undistorted_img)
    cv2.imshow("Original Image", img)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import numpy as np
import cv2
import matplotlib.pyplot as plt

mtx = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
dist = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_dist.npy")

# 歪み補正を行う画像をロード
image_path = "FaceImages/Nakabayashi/mouth/test14.jpg"
#image_path = "CameraCalibration/ChessBoard_mouth_right/test1.jpg"
img = cv2.imread(image_path)

# 歪み補正を行う
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 歪み補正された画像を取得
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# ROIを取得し、画像をトリミング
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]

# 結果を表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Undistorted Image')
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

plt.show()

# 歪み補正された画像を保存（オプション）
#cv2.imwrite('undistorted_image.jpg', dst)
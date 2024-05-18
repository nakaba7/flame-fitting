import cv2
import numpy as np
import glob
from cv2 import aruco

# Charucoボードの設定
square_size = 3.0  # 正方形の1辺のサイズ[cm]
marker_size = 1.5  # Arucoマーカーのサイズ[cm]
pattern_size = (4, 6)  # Charucoボードの内のチェスボードパターンのサイズ
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
charuco_board = aruco.CharucoBoard_create(pattern_size[0], pattern_size[1], square_size, marker_size, aruco_dict)

# 画像のパスを指定
image_path = 'StereoImage_mouth_left/test0.jpg'  # ここにテストしたい画像のパスを指定してください

# 画像を読み込み
img = cv2.imread(image_path)
if img is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# マーカーを検出
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

# Charucoボードのコーナーを検出
if len(corners) > 0:
    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
    if ret > 0:
        print(f"Charuco corners detected: {len(charuco_corners)}")
        # 検出されたコーナーを描画
        aruco.drawDetectedMarkers(img, corners, ids)
        aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
    else:
        print("Charuco corners were not detected.")
else:
    print("Markers were not detected.")

# 画像を表示
cv2.imshow('Charuco Board', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

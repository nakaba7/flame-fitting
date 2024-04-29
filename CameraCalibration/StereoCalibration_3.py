import cv2
import numpy as np
import glob
import os

def draw_matches_three_cam(img1, img2, img3, points1, points2, points3, output_file):
    # 画像を横に連結
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    h = max(h1, h2, h3)
    w = w1 + w2 + w3
    img_matches = np.zeros((h, w, 3), dtype="uint8")
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:w1+w2] = img2
    img_matches[:h3, w1+w2:w] = img3

    # 対応点を線で結ぶ
    for p1, p2, p3 in zip(points1, points2, points3):
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        p3 = tuple(np.round(p3).astype(int) + np.array([w1 + w2, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_matches, p1, p2, color, 1)
        cv2.line(img_matches, p2, p3, color, 1)

    # 画像を保存
    cv2.imwrite(output_file, img_matches)
output_folder = "ChessBoard_Correspondences"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# キャリブレーションデータの読み込み
# ここでは、mtx_a, dist_a, mtx_b, dist_b, R, T などがキャリブレーションから得られたパラメータとします。
mtx_a = np.load("Parameters/ChessBoard_left_mtx.npy")
dist_a = np.load("Parameters/ChessBoard_left_dist.npy")
mtx_b = np.load("Parameters/ChessBoard_right_mtx.npy")
dist_b = np.load("Parameters/ChessBoard_right_dist.npy")
mtx_c = np.load("Parameters/ChessBoard_mouth_mtx.npy")
dist_c = np.load("Parameters/ChessBoard_mouth_dist.npy")

# チェスボードの設定
square_size = 2.5
pattern_size = (6, 8)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
# 各カメラからの画像セットへのパス
images1 = glob.glob('ChessBoard_left/*.jpg')
images2 = glob.glob('ChessBoard_right/*.jpg')
images3 = glob.glob('ChessBoard_mouth/*.jpg')

# チェスボードのコーナーを格納するリストの初期化
objpoints = []  # 3Dポイント
imgpoints1 = []  # カメラ1の2Dポイント
imgpoints2 = []  # カメラ2の2Dポイント
imgpoints3 = []  # カメラ3の2Dポイント
idx = 0
for img_file1, img_file2, img_file3 in zip(images1, images2, images3):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    img3 = cv2.imread(img_file3)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    
    # チェスボードのコーナーを探す
    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)
    ret3, corners3 = cv2.findChessboardCorners(gray3, pattern_size)

    # 3つの画像でコーナーが見つかった場合、そのポイントをリストに追加
    if ret1 and ret2 and ret3:
        objpoints.append(pattern_points)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
        imgpoints3.append(corners3)
        draw_matches_three_cam(img1, img2, img3, corners1[:,0,:], corners2[:,0,:], corners3[:,0,:], os.path.join(output_folder, f"matches_{idx+1}.jpg"))
    idx += 1

# A-BおよびB-Cカメラペアに対してステレオキャリブレーションを行う
# A-Bカメラペアのキャリブレーション
retval_ab, cameraMatrix1_ab, distCoeffs1_ab, cameraMatrix2_ab, distCoeffs2_ab, R_ab, T_ab, E_ab, F_ab = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1])

# B-Cカメラペアのキャリブレーション
retval_bc, cameraMatrix1_bc, distCoeffs1_bc, cameraMatrix2_bc, distCoeffs2_bc, R_bc, T_bc, E_bc, F_bc = cv2.stereoCalibrate(
    objpoints, imgpoints2, imgpoints3, mtx_b, dist_b, mtx_c, dist_c, gray2.shape[::-1])

# ステレオキャリブレーションの結果を保存または使用
np.save('Parameters/R_ab.npy', R_ab)
np.save('Parameters/T_ab.npy', T_ab)
np.save('Parameters/R_bc.npy', R_bc)
np.save('Parameters/T_bc.npy', T_bc)

# Rectificationを計算
R1_ab, R2_ab, P1_ab, P2_ab, Q_ab, roi1_ab, roi2_ab = cv2.stereoRectify(
    cameraMatrix1_ab, distCoeffs1_ab, cameraMatrix2_ab, distCoeffs2_ab, gray1.shape[::-1], R_ab, T_ab)

R1_bc, R2_bc, P1_bc, P2_bc, Q_bc, roi1_bc, roi2_bc = cv2.stereoRectify(
    cameraMatrix1_bc, distCoeffs1_bc, cameraMatrix2_bc, distCoeffs2_bc, gray2.shape[::-1], R_bc, T_bc)

# Rectification結果を保存
np.save('Parameters/P1_ab.npy', P1_ab)
np.save('Parameters/P2_ab.npy', P2_ab)
np.save('Parameters/P1_bc.npy', P1_bc)
np.save('Parameters/P2_bc.npy', P2_bc)

print("Calibration and rectification parameters saved!")

import cv2
import numpy as np
import glob
import os

# 画像を横に連結して対応点を線で結ぶ関数
def draw_matches(img1, img2, points1, points2, output_file):
    # 画像を横に連結
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    img_matches = np.zeros((h, w, 3), dtype="uint8")
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:w1+w2] = img2

    # 対応点を線で結ぶ
    for p1, p2 in zip(points1, points2):
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_matches, p1, p2, color, 1)

    # 画像を保存
    cv2.imwrite(output_file, img_matches)

# 処理の例
# 対応点の描画と保存を行う
# 以下の points1 と points2 は、対応するコーナー点のリスト
# img1, img2 は対応する画像
# この部分は対応する点のデータと実際の画像データに基づいて適宜調整してください
output_folder = "ChessBoard_Correspondences"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# キャリブレーションデータの読み込み
# ここでは、mtx_a, dist_a, mtx_b, dist_b, R, T などがキャリブレーションから得られたパラメータとします。
mtx_a = np.load("ChessBoard_a_mtx.npy")
dist_a = np.load("ChessBoard_a_dist.npy")
mtx_b = np.load("ChessBoard_b_mtx.npy")
dist_b = np.load("ChessBoard_b_dist.npy")

# チェスボードの設定
square_size = 2.5
pattern_size = (6, 8)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

# 各カメラからの画像セットへのパス
images1 = glob.glob('ChessBoard_a/*.jpg')
images2 = glob.glob('ChessBoard_b/*.jpg')

# 両方のカメラからの画像で共通して見つかったチェスボードのコーナーを格納するリスト
objpoints = []  # 3Dポイント
imgpoints1 = []  # カメラ1の2Dポイント
imgpoints2 = []  # カメラ2の2Dポイント
idx=0
for img_file1, img_file2 in zip(images1, images2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(img_file1, img_file2)
    
    # チェスボードのコーナーを探す
    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)
    
    # 両方の画像でコーナーが見つかった場合、そのポイントをリストに追加
    if ret1 and ret2:
        objpoints.append(pattern_points)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
        draw_matches(img1, img2, corners1[:,0,:], corners2[:,0,:], os.path.join(output_folder, f"matches_{idx+1}.jpg"))
    idx += 1
# ステレオキャリブレーションを実行
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1])

# ステレオキャリブレーションの結果を保存または使用
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)
# 必要に応じて他のパラメータ（E, F）を保存または使用

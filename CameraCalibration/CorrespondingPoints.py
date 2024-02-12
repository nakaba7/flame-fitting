import cv2
import numpy as np
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

for idx, (corners1, corners2) in enumerate(zip(imgpoints1, imgpoints2)):
    img1 = cv2.imread(images1[idx])
    img2 = cv2.imread(images2[idx])
    draw_matches(img1, img2, corners1[:,0,:], corners2[:,0,:], os.path.join(output_folder, f"matches_{idx+1}.jpg"))

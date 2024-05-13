import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

"""
2つのカメラで撮影したチェスボードの画像からステレオキャリブレーションを行うスクリプト．

Usage:
    python StereoCalibration.py -f eye_[left|right] mouth_[left|right]

Args:
    -f: Enter the folder names of the images from the two cameras.
"""

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
    
def main(args):
    # 処理の例
    # 対応点の描画と保存を行う
    # 以下の points1 と points2 は、対応するコーナー点のリスト
    # img1, img2 は対応する画像
    # この部分は対応する点のデータと実際の画像データに基づいて適宜調整してください
    camera0 = args.f[0]
    camera1 = args.f[1]
    output_folder = f"ChessBoard_Correspondences/{camera0}_{camera1}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # フォルダが存在する場合、.jpgファイルを検索して削除
        for jpg_file in glob.glob(os.path.join(output_folder, "*.jpg")):
            os.remove(jpg_file)
        print("Deleted all .jpg files in the folder.")
    
    # キャリブレーションデータの読み込み
    # ここでは、mtx_a, dist_a, mtx_b, dist_b, R, T などがキャリブレーションから得られたパラメータとします。
    mtx_a = np.load(f"Parameters/ChessBoard_{camera0}_mtx.npy")
    dist_a = np.load(f"Parameters/ChessBoard_{camera0}_dist.npy")
    mtx_b = np.load(f"Parameters/ChessBoard_{camera1}_mtx.npy")
    dist_b = np.load(f"Parameters/ChessBoard_{camera1}_dist.npy")

    # チェスボードの設定
    square_size = 2.5
    pattern_size = (6, 8)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # 各カメラからの画像セットへのパス
    images1 = glob.glob(f'StereoImage_{camera0}/*.jpg')
    images2 = glob.glob(f'StereoImage_{camera1}/*.jpg')

    # 両方のカメラからの画像で共通して見つかったチェスボードのコーナーを格納するリスト
    objpoints = []  # 3Dポイント
    imgpoints1 = []  # カメラ1の2Dポイント
    imgpoints2 = []  # カメラ2の2Dポイント
    idx=0
    for idx, (img_file1, img_file2) in enumerate(tqdm(zip(images1, images2), total=len(images1), desc="Processing images")):
        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #print(img_file1, img_file2)
        
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
    # RとTをファイルに保存
    np.save(f'Parameters/R_{camera0}_{camera1}.npy', R)
    print("R saved!")
    np.save(f'Parameters/T_{camera0}_{camera1}.npy', T)
    np.save('Parameters/T.npy', T)
    print("T saved!")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1], R, T)#射影行列の計算

    print("P1:\n", P1)
    print("P2:\n", P2)

    np.save(f'Parameters/P1_{camera0}_{camera1}.npy', P1)
    print("P1 saved!")
    np.save(f'Parameters/P2_{camera0}_{camera1}.npy', P2)
    print("P2 saved!")

    print("Retval: ", retval)
    with open(f"Parameters/retval_{camera0}_{camera1}.txt", "w") as f:
        f.write(str(retval))

    # カメラ間の距離を計算
    distance = np.linalg.norm(T)
    print(f"Distance between cameras: {distance*square_size} cm")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',nargs=2 , type=str,  help='Enter the folder names of the images from the two cameras.')
    main(parser.parse_args())



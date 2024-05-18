import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
from cv2 import aruco

"""
2つのカメラで撮影したチャルコボードの画像からステレオキャリブレーションを行うスクリプト.

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
        cv2.line(img_matches, p1, p2, color, 3)

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
    output_folder = f"Charuco_Correspondences/{camera0}_{camera1}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # フォルダが存在する場合、.jpgファイルを検索して削除
        for jpg_file in glob.glob(os.path.join(output_folder, "*.jpg")):
            os.remove(jpg_file)
        print("Deleted all .jpg files in the folder.")
    
    # キャリブレーションデータの読み込み
    mtx_a = np.load(f"Parameters/Charuco_{camera0}_mtx.npy")
    dist_a = np.load(f"Parameters/Charuco_{camera0}_dist.npy")
    mtx_b = np.load(f"Parameters/Charuco_{camera1}_mtx.npy")
    dist_b = np.load(f"Parameters/Charuco_{camera1}_dist.npy")

    # Charucoボードの設定
    square_size = 3.0  # このサイズはボードの実際のサイズに合わせて変更
    marker_size = 1.5  # このサイズはボードの実際のサイズに合わせて変更
    pattern_size = (4, 6)  # Charucoボードの内のチェスボードパターンのサイズ
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    board = aruco.CharucoBoard_create(pattern_size[0], pattern_size[1], square_size, marker_size, aruco_dict)

    # 各カメラからの画像セットへのパス
    images1 = glob.glob(f'StereoImage_{camera0}/*.jpg')
    images2 = glob.glob(f'StereoImage_{camera1}/*.jpg')
    # images1 = sorted(images1)
    # images2 = sorted(images2)

    # 両方のカメラからの画像で共通して見つかったCharucoボードのコーナーを格納するリスト
    objpoints = []  # 3Dポイント
    imgpoints1 = []  # カメラ1の2Dポイント
    imgpoints2 = []  # カメラ2の2Dポイント
    idx = 0

    for idx in range(len(images1)):
        left_img_filename_base = "test{}_0.jpg".format(idx)
        right_img_filename_base = "test{}_1.jpg".format(idx)
        mouth_img_filename_base = "test{}.jpg".format(idx)

        if camera0.startswith("eye") and camera1.startswith("mouth"):
            if camera0.endswith("left"):
                img_file1 = os.path.join(f'StereoImage_{camera0}', left_img_filename_base)
            elif camera0.endswith("right"):
                img_file1 = os.path.join(f'StereoImage_{camera0}', right_img_filename_base)
            img_file2 = os.path.join(f'StereoImage_{camera1}', mouth_img_filename_base)
        elif camera1.startswith("eye") and camera0.startswith("mouth"):
            if camera1.endswith("left"):
                img_file2 = os.path.join(f'StereoImage_{camera1}', left_img_filename_base)
            elif camera1.endswith("right"):
                img_file2 = os.path.join(f'StereoImage_{camera1}', right_img_filename_base)
            img_file1 = os.path.join(f'StereoImage_{camera0}', mouth_img_filename_base)
        else:
            print("Invalid args. Choose eye_left, eye_right, mouth_left or mouth_right.")
            exit()
        print("\nf1: {}\nf2: {}".format(img_file1, img_file2))

        if not (os.path.exists(img_file1) and os.path.exists(img_file2)):
            print("File does not exist.")
            continue

        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img_file2, cv2.COLOR_BGR2GRAY)

        # Charucoボードのコーナーを探す
        corners1, ids1, rejectedImgPoints1 = aruco.detectMarkers(gray1, aruco_dict)
        corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(gray2, aruco_dict)
        
        if len(corners1) > 0 and len(corners2) > 0:
            ret1, corners1, ids1 = aruco.interpolateCornersCharuco(corners1, ids1, gray1, board)
            ret2, corners2, ids2 = aruco.interpolateCornersCharuco(corners2, ids2, gray2, board)
            if ret1 > 20 and ret2 > 20:  # Charucoボードのコーナーが見つかった場合
                print("charuco corners were detected in both images.")

                objpoints.append(board.chessboardCorners)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                
                draw_matches(img1, img2, corners1, corners2, os.path.join(output_folder, f"matches_{idx}.jpg"))

    if not objpoints:
        print("No valid images for calibration.")
        return

    # ステレオキャリブレーションを実行
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1], criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))

    # ステレオキャリブレーションの結果を保存または使用
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    np.save(f'Parameters/R_{camera0}_{camera1}.npy', R)
    np.save(f'Parameters/T_{camera0}_{camera1}.npy', T)
    np.save('Parameters/T.npy', T)
    print("Calibration parameters saved.")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1], R, T)

    print("P1:\n", P1)
    print("P2:\n", P2)

    np.save(f'Parameters/P1_{camera0}_{camera1}.npy', P1)
    np.save(f'Parameters/P2_{camera0}_{camera1}.npy', P2)

    print("Retval: ", retval)

    distance = np.linalg.norm(T)
    print(f"Distance between cameras: {distance * square_size} cm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs=2, type=str, help='Enter the folder names of the images from the two cameras.')
    main(parser.parse_args())

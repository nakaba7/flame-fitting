import cv2
import os
import numpy as np
import glob
import argparse
from tqdm import tqdm

"""
カメラキャリブレーションを行うスクリプト. 1つのカメラずつで行う. カメラ行列と歪みパラメータを保存する.
カレントディレクトリがCameraCalibrationの状態で実行すること.

Usage:
    python CameraCalibration.py -f eye_left
    python CameraCalibration.py -f eye_right
    python CameraCalibration.py -f mouth_left
    python CameraCalibration.py -f mouth_right

Args:  
    -f: Choose left or right or mouth. Specify the camera to calibrate.
"""

def main(args):
    if args.f not in ['eye_left', 'eye_right', 'mouth_left', 'mouth_right']:
        print("Invalid args. Choose eye_left, eye_right, mouth_left or mouth_right.")
        exit() 

    square_size = 25      # 正方形の1辺のサイズ[mm]
    pattern_size = (6, 8)  # 交差ポイントの数
    chessname = f"ChessBoard_{args.f}"
    folder_name = chessname
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    print("Use images in ", folder_name)
    # 新しいフォルダーの名前
    output_folder_name = "Processed_Images"
    # folder_nameの下に新しいフォルダーを作成（存在しない場合）
    output_path = os.path.join(folder_name, output_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)  # チェスボード（X,Y,Z）座標の指定 (Z=0)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    objpoints = []
    imgpoints = []

    images = glob.glob(f'{folder_name}/*.jpg')
    count = 0
    for filepath in tqdm(images):
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        if ret:
            print("detected corner!")
            print(f"{len(objpoints)+1}:{filepath}")
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            imgpoints.append(corners.reshape(-1, 2))
            objpoints.append(pattern_points)

            # コーナーを描画
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            # 画像を新しいフォルダーに保存
            img_filename = os.path.basename(filepath)
            cv2.imwrite(os.path.join(output_path, img_filename), img)
            
        count += 1

    print("calculating camera parameter...")
    # 内部パラメータを計算
    # ret: Root Mean Square (RMS), mtx: カメラ行列, dist: 歪みパラメータ, rvecs: 回転ベクトル, tvecs: 並進ベクトル
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 計算結果を保存
    param_path = os.path.join("Parameters", chessname)
    if not os.path.exists("Parameters"):
        os.makedirs("Parameters")
    
    np.save(f"{param_path}_mtx.npy", mtx)  # カメラ行列
    np.save(f"{param_path}_dist.npy", dist.ravel())  # 歪みパラメータ
    np.save(f"{param_path}_rvecs.npy", rvecs)  # 回転ベクトル
    np.save(f"{param_path}_tvecs.npy", tvecs)  # 並進ベクトル
    # 計算結果を表示
    print("RMS = ", ret)
    with open(f"{folder_name}/RMS.txt", "w") as f:
        f.write(str(ret))
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    print()
    print("Start undistortion...")
    # 画像の歪み補正
    for filepath in tqdm(images):
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 補正
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # 補正後の画像を保存
        img_filename = "calib_" + os.path.basename(filepath)
        cv2.imwrite(os.path.join(output_path, img_filename), dst)
        #print(f"Calibrated image saved as {os.path.join(output_path, img_filename)}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str, help='Choose eye_left, eye_right, mouth_left or mouth_right.')
    main(parser.parse_args())

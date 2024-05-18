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
    if args.f != 'eye_left' and args.f != 'eye_right' and args.f != 'mouth_left' and args.f != 'mouth_right':
        print("Invalid args. Choose eye_left, eye_right, mouth_left or mouth_right.")
        exit() 
    square_size = 2.5      # 正方形の1辺のサイズ[cm]
    pattern_size = (6, 8)  # 交差ポイントの数
    chessname = f"ChessBoard_{args.f}"
    folder_name = chessname
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    print("Use images in ",folder_name)
    #reference_img = 50 # 参照画像の枚数

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
        #print("size of image: ", gray.shape[::-1])
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
            
        #if count >= reference_img - 1:
         #   break
        count += 1

    print("calculating camera parameter...")
    # 内部パラメータを計算
    # ret: Root Men Square(RMS), mtx: カメラ行列, dist: 歪みパラメータ, rvecs: 回転ベクトル, tvecs: 並進ベクトル
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 計算結果を保存
    np.save(os.path.join(f"Parameters/{chessname}_mtx"), mtx)  # カメラ行列
    np.save(os.path.join(f"Parameters/{chessname}_dist"), dist.ravel())  # 歪みパラメータ
    np.save(os.path.join(f"Parameters/{chessname}_rvecs"), rvecs)  # 回転ベクトル
    np.save(os.path.join(f"Parameters/{chessname}_tvecs"), tvecs)  # 並進ベクトル
    # 計算結果を表示
    print("RMS = ", ret)
    with open(f"{folder_name}/RMS.txt", "w") as f:
        f.write(str(ret))
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='hoge', type=str,  help='whether ChessBoard_eye_left, ChessBoard_eye_right, ChessBoard_mouth_left or ChessBoard_mouth_right. Choose eye_left, eye_right, mouth_left or mouth_right.')
    main(parser.parse_args())
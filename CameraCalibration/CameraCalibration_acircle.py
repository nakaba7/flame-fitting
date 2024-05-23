import cv2
import numpy as np
import glob
import os
import sys
import argparse

def main(args):
    # サークルグリッドのサイズ
    pattern_size = (5, 11)
    input_dir = args.f
    # サークルのワールド座標を作成
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= 39.5  # 例えば、サークルの間隔を39.5mmとします

    # 3Dポイントと2Dポイントのリスト
    objpoints = []
    imgpoints = []

    # jpgファイルをすべて取得
    images = glob.glob(os.path.join(input_dir, '*.jpg'))
    output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # サークルグリッドのコーナーを見つける
        ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret:
            objpoints.append(objp)
            imgpoints.append(centers)

            # サークルグリッドを描画
            cv2.drawChessboardCorners(img, pattern_size, centers, ret)
            output_fname = os.path.join(output_dir, os.path.basename(fname))
            cv2.imwrite(output_fname, img)
            print(f"{fname} is detected and saved as {output_fname}")

    # カメラキャリブレーション
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("ret:", ret)
    print("カメラ行列:\n", mtx)
    print("歪み係数:\n", dist)
    """
    # 画像の補正
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 補正
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        output_fname = os.path.join(output_dir, 'calib_' + os.path.basename(fname))
        cv2.imwrite(output_fname, dst)
        print(f"Calibrated image saved as {output_fname}")
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='acircle_eye_left', type=str,  help='Choose acircle_eye_left, acircle_eye_right, acircle_mouth_left or acircle_mouth_right.')
    main(parser.parse_args())

import cv2
import numpy as np
import os
import glob
import argparse
"""
CameraCalibration.pyで保存したカメラ行列と歪みパラメータを使って画像の歪み補正を行うスクリプト.

Usage:
    python Undistort.py -i input_folder -c camera_matrix -d dist_coeffs
Args:
    -i: 歪み補正を行う画像が保存されているディレクトリ
    -c: カメラ行列のnumpyファイルのパス
    -d: 歪みパラメータのnumpyファイルのパス
"""
def undistort_image(image_path, camera_matrix, dist_coeffs):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 新しいカメラ行列とROIを取得
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 歪み補正
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

    # ROIで切り取る
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def main(args):
    # カメラ行列と歪みパラメータを読み込む
    camera_matrix = np.load(args.camera_matrix)
    dist_coeffs = np.load(args.dist_coeffs)

    # 入力画像を取得
    images = glob.glob(f'{args.input_folder}/*.jpg')
    if len(images) == 0:
        print(f'No images found in {args.input_folder}.')
        return

    for image_path in images:
        # 歪み補正を行う
        undistorted_img = undistort_image(image_path, camera_matrix, dist_coeffs)

        # 補正後の画像を表示
        cv2.imshow('Undistorted Image', undistorted_img)
        cv2.waitKey(0)  # キー入力待ち（任意のキーを押すまで表示）
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Folder containing the images to undistort.')
    parser.add_argument('-c', '--camera_matrix', type=str, required=True, help='Path to the numpy file containing the camera matrix.')
    parser.add_argument('-d', '--dist_coeffs', type=str, required=True, help='Path to the numpy file containing the distortion coefficients.')
    args = parser.parse_args()
    main(args)

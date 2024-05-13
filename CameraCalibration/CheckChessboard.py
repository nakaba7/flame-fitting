import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

"""
カメラ1とカメラ2で撮影したチェスボードの画像からコーナーを検出し, 順番を表示するスクリプト.
usage:  
    python CheckChessboard.py -f [camera1] [camera2]
"""


def draw_corners(image, corners):
    for i, corner in enumerate(corners):
        cv2.circle(image, tuple(corner.ravel()), 5, (0, 0, 255), -1)
        cv2.putText(image, str(i), tuple(corner.ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def main(args):
    camera0 = args.f[0]
    camera1 = args.f[1]

    pattern_size = (6, 8)

    images1 = sorted(glob.glob(f'StereoImage_{camera0}/*.jpg'))
    images2 = sorted(glob.glob(f'StereoImage_{camera1}/*.jpg'))
    print(f"Found {len(images1)} images for {camera0} and {len(images2)} images for {camera1}.")

    # 画像の順番を逆にする
    #images1.reverse()
    #images2.reverse()

    for idx, (img_file1, img_file2) in enumerate(tqdm(zip(images1, images2), total=len(images1), desc="Processing images")):
        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)

        if ret1:
            draw_corners(img1, corners1)
        if ret2:
            draw_corners(img2, corners2)

        cv2.imshow(f'Image 1 with corners - {idx + 1}', img1)
        cv2.imshow(f'Image 2 with corners - {idx + 1}', img2)
        cv2.waitKey(0)  # 各画像ごとにキー入力を待つ
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs=2, type=str, help='Enter the folder names of the images from the two cameras.')
    main(parser.parse_args())

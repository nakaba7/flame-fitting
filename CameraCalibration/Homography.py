import cv2
import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt
"""
2つのカメラで撮影した画像からホモグラフィ行列を計算するスクリプト
usage:
    python Homography.py -f eye_[left|right] mouth_[left|right]
    python Homography.py -f mouth_[left|right] eye_[left|right]
args:
    -f: 2つのカメラを指定
"""

def homography(args):
    camera0 = args.f[0]
    camera1 = args.f[1]
    # カメラ行列と歪み係数を読み込む
    mtx_a = np.load(f"Parameters/ChessBoard_{camera0}_mtx.npy")
    dist_a = np.load(f"Parameters/ChessBoard_{camera0}_dist.npy")
    mtx_b = np.load(f"Parameters/ChessBoard_{camera1}_mtx.npy")
    dist_b = np.load(f"Parameters/ChessBoard_{camera1}_dist.npy")

    # 回転行列と並進ベクトルを読み込む
    R = np.load(f"Parameters/R_{camera0}_{camera1}.npy")
    T = np.load(f"Parameters/T_{camera0}_{camera1}.npy")

    # 画像を読み込む
    images1 = sorted(glob.glob(f'StereoImage_{camera0}/*.jpg'))
    images2 = sorted(glob.glob(f'StereoImage_{camera1}/*.jpg'))

    # 画像の歪み補正
    def undistort_image(img, mtx, dist):
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
        return undistorted_img

    for idx in range(len(images1)):
            left_img_filename_base = "test{}_0.jpg".format(idx)
            right_img_filename_base = "test{}_1.jpg".format(idx)
            mouth_img_filename_base = "test{}.jpg".format(idx)

            if camera0[0:3] == "eye" and camera1[0:5] == "mouth":
                if camera0.endswith("left"):
                    img_file1 = os.path.join(f'StereoImage_{camera0}', left_img_filename_base)
                elif camera0.endswith("right"):
                    img_file1 = os.path.join(f'StereoImage_{camera0}', right_img_filename_base)
                img_file2 = os.path.join(f'StereoImage_{camera1}', mouth_img_filename_base)
            elif camera1[0:3] == "eye" and camera0[0:5] == "mouth":
                if camera1.endswith("left"):
                    img_file2 = os.path.join(f'StereoImage_{camera1}', left_img_filename_base)
                elif camera1.endswith("right"):
                    img_file2 = os.path.join(f'StereoImage_{camera1}', right_img_filename_base)
                img_file1 = os.path.join(f'StereoImage_{camera0}', mouth_img_filename_base)
            else:
                print("Invalid args. Choose eye_left, eye_right, mouth_left or mouth_right.")
                exit()
            #print("\nf1: {}\nf2: {}".format(img_file1, img_file2))
        
            if((os.path.exists(img_file1) == False) or (os.path.exists(img_file2) == False)):
                print("File does not exist.")
                continue
            print(f"{img_file1} and {img_file2} loaded")
            img1 = cv2.imread(img_file1)
            img2 = cv2.imread(img_file2)

            #img1 = undistort_image(img1, mtx_a, dist_a)
            #img2 = undistort_image(img2, mtx_b, dist_b)

            # サークルグリッドの特徴点を検出
            pattern_size = (5, 11)  # サークルの数
            ret1, centers1 = cv2.findCirclesGrid(img1, pattern_size, None, cv2.CALIB_CB_ASYMMETRIC_GRID)
            ret2, centers2 = cv2.findCirclesGrid(img2, pattern_size, None, cv2.CALIB_CB_ASYMMETRIC_GRID)

            if ret1 and ret2:
                print("Circle grid found in both images.")
                # 2Dポイントを2列の形式に変換
                centers1 = centers1.reshape(-1, 2)
                centers2 = centers2.reshape(-1, 2)
                
                # 対応する特徴点を使用してホモグラフィ行列を計算
                H, status = cv2.findHomography(centers1, centers2)

                # 特徴点を変換
                transformed_points = cv2.perspectiveTransform(np.array([centers1]), H)
                print("Transformed points from Camera 1 to Camera 2 coordinate system:")
                print(transformed_points)
                
                # Matplotlibで結果を表示
                plt.figure(figsize=(12, 6))

                # 元の画像と特徴点
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                plt.scatter(centers1[:, 0], centers1[:, 1], c='r', marker='o')
                plt.title('Original Image 1 with Features')

                # 変換後の特徴点を元の画像2に重ねて表示
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                plt.scatter(centers2[:, 0], centers2[:, 1], c='r', marker='o')
                plt.title('Transformed Features on Image 2')

                plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs=2, type=str, required=True, help='Enter the folder names of the images from the two cameras.')
    homography(parser.parse_args())
import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

"""
2つのカメラで撮影した非対称サークルグリッドの画像からステレオキャリブレーションを行うスクリプト。

Usage:
    python StereoCalibration.py -f eye_left mouth_left

Args:
    -f: Enter the folder names of the images from the two cameras.
"""

# 画像を横に連結して対応点を線で結ぶ関数
def draw_matches(img1, img2, points1, points2, output_file):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    img_matches = np.zeros((h, w, 3), dtype="uint8")
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:w1+w2] = img2

    for p1, p2 in zip(points1, points2):
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_matches, p1, p2, color, 1)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_file, img_matches)

# 画像を横に並べてフルスクリーンで表示する関数
def show_side_by_side(img1, img2):
    # 画像サイズを確認し、異なる場合はリサイズ
    if img1.shape != img2.shape:
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (max_width, max_height))
        img2 = cv2.resize(img2, (max_width, max_height))

    combined_img = cv2.hconcat([img1, img2])
    
    cv2.namedWindow('Side by Side', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Side by Side', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Side by Side', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    return undistorted_img

def main(args):
    camera0 = args.f[0]
    camera1 = args.f[1]
    undistortion_flag = args.d
    if undistortion_flag:
        output_folder = f"CircleGrid_Correspondences/{camera0}_{camera1}_undistort"
    else:
        output_folder = f"CircleGrid_Correspondences/{camera0}_{camera1}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for jpg_file in glob.glob(os.path.join(output_folder, "*.jpg")):
            os.remove(jpg_file)
        print("Deleted all .jpg files in the folder.")
    
    # キャリブレーションデータの読み込み
    mtx_a = np.load(f"Parameters/ChessBoard_{camera0}_mtx.npy")
    dist_a = np.load(f"Parameters/ChessBoard_{camera0}_dist.npy")
    mtx_b = np.load(f"Parameters/ChessBoard_{camera1}_mtx.npy")
    dist_b = np.load(f"Parameters/ChessBoard_{camera1}_dist.npy")

    # サークルグリッドの設定
    square_size = 39.5  # サークルの間隔[mm]
    pattern_size = (5, 11)  # サークルの数 (5列, 11行)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points[:, 0] *= 2
    pattern_points *= square_size

    # 各カメラからの画像セットへのパス
    images1 = sorted(glob.glob(f'StereoImage_{camera0}/*.jpg'))
    images2 = sorted(glob.glob(f'StereoImage_{camera1}/*.jpg'))
    print(f"Found {len(images1)} images for {camera0} and {len(images2)} images for {camera1}.")

    objpoints = []  # 3Dポイント
    imgpoints1 = []  # カメラ1の2Dポイント
    imgpoints2 = []  # カメラ2の2Dポイント
    tmp = 0
    firstflag = True
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

        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)
        print(f"{img_file1} and {img_file2} loaded")

        if undistortion_flag:
            # 歪み補正を行う
            img1 = undistort_image(img1, mtx_a, dist_a)
            img2 = undistort_image(img2, mtx_b, dist_b)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if firstflag:
            firstflag = False
            tmp = gray1.shape[::-1]
        ret1, centers1 = cv2.findCirclesGrid(gray1, pattern_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        ret2, centers2 = cv2.findCirclesGrid(gray2, pattern_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        #print(ret1, ret2)
        if ret1 and ret2:
            print(f"Circle grid corners found in both images {idx}.")
            objpoints.append(pattern_points)
            imgpoints1.append(centers1)
            imgpoints2.append(centers2)
            if undistortion_flag:
                draw_matches(img1, img2, centers1[:,0,:], centers2[:,0,:], os.path.join(output_folder, f"matches_{idx+1}_undistort.jpg"))
            else:
                draw_matches(img1, img2, centers1[:,0,:], centers2[:,0,:], os.path.join(output_folder, f"matches_{idx+1}.jpg"))

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx_a, dist_a, mtx_b, dist_b, tmp)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_a, dist_a, mtx_b, dist_b, gray1.shape[::-1], R, T)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("P1:\n", P1)
    print("P2:\n", P2)
    print("Retval: ", retval)   
    if undistortion_flag:
        np.save(f'Parameters/R_{camera0}_{camera1}_undistort.npy', R)
        np.save(f'Parameters/T_{camera0}_{camera1}_undistort.npy', T)
        np.save(f'Parameters/P1_{camera0}_{camera1}_undistort.npy', P1)
        np.save(f'Parameters/P2_{camera0}_{camera1}_undistort.npy', P2)
        with open(os.path.join(output_folder, f"retval_{camera0}_{camera1}_undistort.txt"), "w") as f:
            f.write(str(retval))
    else:
        np.save(f'Parameters/R_{camera0}_{camera1}.npy', R)
        np.save(f'Parameters/T_{camera0}_{camera1}.npy', T)
        np.save(f'Parameters/P1_{camera0}_{camera1}.npy', P1)
        np.save(f'Parameters/P2_{camera0}_{camera1}.npy', P2)
        with open(os.path.join(output_folder, f"retval_{camera0}_{camera1}.txt"), "w") as f:
            f.write(str(retval))

    distance = np.linalg.norm(T)
    print(f"Distance between cameras: {distance * square_size}mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs=2, type=str, required=True, help='Enter the folder names of the images from the two cameras.')
    parser.add_argument('-d', default=False, type=bool, help='whether to undistort the images or not.')
    main(parser.parse_args())

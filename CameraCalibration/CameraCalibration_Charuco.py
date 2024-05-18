import cv2
import os
import numpy as np
import glob
import argparse
import cv2.aruco as aruco

"""
チャルコボードを使用してカメラキャリブレーションを行うスクリプト. 1つのカメラずつで行う. カメラ行列と歪みパラメータを保存する.
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
        
    square_size = 3.0  # 正方形の1辺のサイズ[cm]
    marker_size = 1.5  # Arucoマーカーのサイズ[cm]
    pattern_size = (4, 6)  # チャルコボードのチェスボードパターンのサイズ (必要に応じて変更)
    charuco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    charuco_board = aruco.CharucoBoard_create(pattern_size[0], pattern_size[1], square_size, marker_size, charuco_dict)
    
    folder_name = f"Charuco_{args.f}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    print("Use images in ", folder_name)
    
    output_folder_name = "Processed_Images"
    output_path = os.path.join(folder_name, output_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_corners = []
    all_ids = []
    img_size = None

    images = glob.glob(f'{folder_name}/*.jpg')
    for filepath in images:
        #print(f"Processing {filepath}")
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, charuco_dict)
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            if charuco_ids is not None and len(charuco_ids) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                if img_size is None:
                    img_size = gray.shape[::-1]

                # コーナーを描画
                aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                # 画像を新しいフォルダーに保存
                img_filename = os.path.basename(filepath)
                cv2.imwrite(os.path.join(output_path, img_filename), img)

    if len(all_corners) == 0:
        print("No Charuco corners found in any image.")
        return

    print("calculating camera parameters...")
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # 計算結果を保存
    np.save(os.path.join(f"Parameters/Charuco_{args.f}_mtx.npy"), mtx)  # カメラ行列
    np.save(os.path.join(f"Parameters/Charuco_{args.f}_dist.npy"), dist.ravel())  # 歪みパラメータ

    # 計算結果を表示
    print("RMS = ", ret)
    with open(f"{folder_name}/RMS.txt", "w") as f:
        f.write(str(ret))
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str, help='Specify the camera to calibrate: eye_left, eye_right, mouth_left or mouth_right.')
    main(parser.parse_args())

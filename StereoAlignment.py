import numpy as np
import matplotlib.pyplot as plt
from Convert.Coordinate_convertor import image2camera_coordinates
from Convert.Lmk3d_2_2d import lmk3d_2_2d
from Lmk_plot import plot_2d, plot_2d_3d_compare
import glob
from Convert.Lmk2d_2_3d import lmk2d_2_3d
import os
"""
口, 左目, 右目の2次元ランドマークを3次元ランドマークに変換し, 口のカメラ座標系に統一するスクリプト.
統一後の2次元ランドマークを保存する.

Usage:
    python StereoAlignment.py
"""

def transform_camera2_to_camera1(camera2_points, R, T):
    # Apply rotation and translation to transform camera 2 points to camera 1 coordinates
    camera1_points = np.dot(R.T, (camera2_points - T.T).T).T
    return camera1_points

def lmk_sort(lmk_2d):
    """
    顔の2次元ランドマークをflame-fittingの順番に並び替える関数. add_nose_lmk関数の後に使用する.
    """
    new_lmks = lmk_2d.copy()
    new_lmks[0:5] = lmk_2d[36:41]#右眉毛
    new_lmks[5:10] = lmk_2d[25:30]#左眉毛
    new_lmks[10:14] = lmk_2d[47:51]#鼻上部   
    new_lmks[14:19] = lmk_2d[0:5]#鼻下部
    new_lmks[19:25] = lmk_2d[41:47]#右目
    new_lmks[25:31] = lmk_2d[30:36]#左目
    new_lmks[31:51] = lmk_2d[5:25]#口
    return new_lmks

def add_nose_lmk(presort_lmk_2d):
    """
    左目, 右目, 口のランドマークが合わさった2次元ランドマークに鼻のランドマーク4つを追加する関数.
    Args:
        presort_lmk_2d: ソート前の2次元ランドマークの配列
    """
    # 30番目の点と44番目の点の中点を計算
    midpoint = (presort_lmk_2d[30] + presort_lmk_2d[44]) / 2
    # 鼻の中央の点から上方向に向かうようにする
    midpoint[0] = presort_lmk_2d[2][0]
    # 中点から13番の方向へ一定間隔で3つの点を追加
    direction = (presort_lmk_2d[2] - midpoint)
    
    interval = np.linalg.norm(direction) / 4  # 一定間隔を計算
    unit_direction = direction / np.linalg.norm(direction)

    new_points = [midpoint + unit_direction * interval * (i) for i in range(4)]
    new_points = np.array(new_points)

    # 新しい点をpresort_lmk_2dに追加
    presort_lmk_2d = np.concatenate((presort_lmk_2d, new_points), axis=0)
    return presort_lmk_2d

def adjust_eye_zpos(camera_eye_points, eye_side):
    """
    目のランドマークのz座標を調整する関数. 
    Args:
        camera_eye_points: カメラ座標系の目のランドマークの配列
        eye_side: 'left'または'right'を指定
    """
    middle_eye_offset = 5
    inner_corner_offset = 12

    if eye_side == 'left':
        #眉毛
        camera_eye_points[0][2] += inner_corner_offset
        camera_eye_points[1][2] += middle_eye_offset
        #目
        camera_eye_points[5][2] += inner_corner_offset
        camera_eye_points[6][2] += middle_eye_offset
        camera_eye_points[10][2] += middle_eye_offset
    elif eye_side == 'right':
        #眉毛
        camera_eye_points[3][2] += middle_eye_offset
        camera_eye_points[4][2] += inner_corner_offset
        #目
        camera_eye_points[7][2] += middle_eye_offset
        camera_eye_points[8][2] += inner_corner_offset
        camera_eye_points[9][2] += middle_eye_offset
    else:
        raise ValueError("Invalid eye side. Choose 'left' or 'right'.")
    return camera_eye_points

def stereo_alignment(image_mouth_points, image_lefteye_points, image_righteye_points, mouth_mtx, lefteye_mtx, righteye_mtx, R_mouth2lefteye, T_mouth2lefteye, R_mouth2righteye, T_mouth2righteye):
    """
    ステレオカメラのキャリブレーションパラメータを使用して口と目のランドマークを合わせる関数.
    """
    #z座標をmm単位で指定
    camera_mouth_z_pixel = 60
    camera_lefteye_z_pixel = 30
    camera_righteye_z_pixel = 30

    #画像座標系からカメラ座標系に変換
    camera_mouth_points = image2camera_coordinates(image_mouth_points, camera_mouth_z_pixel, mouth_mtx, True)
    camera_lefteye_points = image2camera_coordinates(image_lefteye_points, camera_lefteye_z_pixel, lefteye_mtx, False)
    camera_righteye_points = image2camera_coordinates(image_righteye_points, camera_righteye_z_pixel, righteye_mtx, False)

    camera_lefteye_points = adjust_eye_zpos(camera_lefteye_points, 'left')
    camera_righteye_points = adjust_eye_zpos(camera_righteye_points, 'right')

    #口カメラのカメラ座標系へ統一
    camera_lefteye_points_in_camera_mouth = transform_camera2_to_camera1(camera_lefteye_points, R_mouth2lefteye, T_mouth2lefteye)
    camera_righteye_points_in_camera_mouth = transform_camera2_to_camera1(camera_righteye_points, R_mouth2righteye, T_mouth2righteye)

    #3次元ランドマークを結合
    all_camera_mouth_points = np.vstack((camera_mouth_points, camera_lefteye_points_in_camera_mouth, camera_righteye_points_in_camera_mouth))

    return all_camera_mouth_points

def make_lmk2d_for_flamefitting(all_camera_mouth_points_3d):
    """
    3Dランドマークをflame-fittingの2Dランドマークに変換する関数.
    """
    all_camera_mouth_points_2d = lmk3d_2_2d(all_camera_mouth_points_3d)
    all_camera_mouth_points_2d = add_nose_lmk(all_camera_mouth_points_2d)
    all_camera_mouth_points_2d = lmk_sort(all_camera_mouth_points_2d)
    # 鼻下部の中央の点の特徴点を取得
    reference_point = all_camera_mouth_points_2d[16] 
    # 全体を平行移動させる
    all_camera_mouth_points_2d -= reference_point
    # y軸を反転させる
    all_camera_mouth_points_2d[:, 1] = -all_camera_mouth_points_2d[:, 1]
    # スケールを1/1000にする
    all_camera_mouth_points_2d /= 1000.0
    return all_camera_mouth_points_2d

if __name__ == '__main__':
    participant_name = "Nakabayashi"
    image_mouth_points_path_list = glob.glob(f'AnnotatedData/{participant_name}_Annotated/NPYs/mouth/*.npy')
    image_lefteye_points_path_list = glob.glob(f'AnnotatedData/{participant_name}_Annotated/NPYs/lefteye/*.npy')
    image_righteye_points_path_list = glob.glob(f'AnnotatedData/{participant_name}_Annotated/NPYs/righteye/*.npy')

    mouth_mtx = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_mtx.npy")
    lefteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
    righteye_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_right_mtx.npy")

    R_mouth2lefteye = np.load("CameraCalibration/Parameters/R_mouth_left_eye_left.npy")
    T_mouth2lefteye = np.load("CameraCalibration/Parameters/T_mouth_left_eye_left.npy")

    R_mouth2righteye = np.load("CameraCalibration/Parameters/R_mouth_right_eye_right.npy")
    T_mouth2righteye = np.load("CameraCalibration/Parameters/T_mouth_right_eye_right.npy")
    output_aligned_3d_dir = 'output_landmark/aligned_3d'
    output_2d_dir = 'output_landmark/estimated_2d'
    output_3d_dir = 'output_landmark/estimated_3d'
    model_path = 'models/DepthOnly_200000.pth'
    for idx in range(200):
        left_npy_filename_base = "test{}_0_annotated.npy".format(idx)
        right_npy_filename_base = "test{}_1_annotated.npy".format(idx)
        mouth_npy_filename_base = "test{}_annotated.npy".format(idx)
        image_mouth_points_path = os.path.join(f'AnnotatedData/{participant_name}_Annotated/NPYs/mouth', mouth_npy_filename_base)
        image_lefteye_points_path = os.path.join(f'AnnotatedData/{participant_name}_Annotated/NPYs/lefteye', left_npy_filename_base)
        image_righteye_points_path = os.path.join(f'AnnotatedData/{participant_name}_Annotated/NPYs/righteye', right_npy_filename_base)
        
        if not(image_mouth_points_path in image_mouth_points_path_list and image_lefteye_points_path in image_lefteye_points_path_list and image_righteye_points_path in image_righteye_points_path_list):
            continue
        
        image_mouth_points = np.load(image_mouth_points_path)
        image_lefteye_points = np.load(image_lefteye_points_path)
        image_righteye_points = np.load(image_righteye_points_path)
        if image_mouth_points.shape[0] != 25 or image_lefteye_points.shape[0] != 11 or image_righteye_points.shape[0] != 11:
            print(f"Invalid landmark shape: {image_mouth_points.shape}, {image_lefteye_points.shape}, {image_righteye_points.shape}")
            if image_mouth_points.shape[0] != 25:
                print(image_mouth_points_path)
            if image_lefteye_points.shape[0] != 11:
                print(image_lefteye_points_path)
            if image_righteye_points.shape[0] != 11:
                print(image_righteye_points_path)
            continue
        all_camera_mouth_points_3d = stereo_alignment(image_mouth_points, image_lefteye_points, image_righteye_points, mouth_mtx, lefteye_mtx, righteye_mtx, R_mouth2lefteye, T_mouth2lefteye, R_mouth2righteye, T_mouth2righteye)
        all_camera_mouth_points_2d = make_lmk2d_for_flamefitting(all_camera_mouth_points_3d)
        predicted_lmk_3d = lmk2d_2_3d(model_path, all_camera_mouth_points_2d)

        # 最初の5文字を取り出してファイル名として使用
        file_name = os.path.basename(image_mouth_points_path)[:6] + ".npy"
        np.save(os.path.join(output_aligned_3d_dir, file_name), all_camera_mouth_points_3d)
        print("Saved 3d aligned:", os.path.join(output_aligned_3d_dir, file_name))
        np.save(os.path.join(output_2d_dir, file_name), all_camera_mouth_points_2d)
        print("Saved 2d:", os.path.join(output_2d_dir, file_name))
        np.save(os.path.join(output_3d_dir, file_name), predicted_lmk_3d)
        print("Saved 3d:", os.path.join(output_3d_dir, file_name))
        print()

        

"""
rvec = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_rvecs.npy")[3]
tvec = np.load("CameraCalibration/Parameters/ChessBoard_mouth_left_tvecs.npy")[3]
print(rvec)
print(tvec)
all_camera_mouth_points_2d, _ = cv2.projectPoints(all_camera_mouth_points, rvec, tvec, mouth_mtx, None)
all_camera_mouth_points_2d = np.squeeze(all_camera_mouth_points_2d, axis=1)
"""


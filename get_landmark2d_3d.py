import numpy as np
import face_alignment
from skimage import io
import warnings
warnings.filterwarnings('ignore')   
import glob
import argparse
import os

"""
顔画像から2D, 3Dランドマークを検出し, それぞれnpyファイルに保存するスクリプト

Usage:
    python get_landmark2d_3d.py -f <folder_name> -o <output_folder>
    -f: 顔画像が保存されているフォルダのパス
    -o: 2D, 3Dランドマークを保存するフォルダのパス
"""

def get_landmark_2d_3d(image_path):
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    input = io.imread(image_path)

    landmarks_2d = np.array(fa_2d.get_landmarks(input))
    landmarks_3d = np.array(fa_3d.get_landmarks(input))

    if landmarks_2d.shape[0] != 1 or landmarks_3d.shape[0] != 1:
        print("Error: landmarks detection failed")
        return None, None

    landmarks_2d = np.squeeze(landmarks_2d)
    landmarks_3d = np.squeeze(landmarks_3d)
    if landmarks_2d.shape[0] != 68 or landmarks_3d.shape[0] != 68:
        print("Error: landmarks detection failed")
        return None, None
    landmarks_2d = landmarks_2d[17:]
    landmarks_3d = landmarks_3d[17:]

    return landmarks_2d, landmarks_3d

def main(args):
    folder_name = args.f
    output_folder = args.o
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder_2d = os.path.join(output_folder, "2d")
    output_folder_3d = os.path.join(output_folder, "3d")
    if not os.path.exists(output_folder_2d):
        os.makedirs(output_folder_2d)
    if not os.path.exists(output_folder_3d):
        os.makedirs(output_folder_3d)

    images = glob.glob(f'{folder_name}/*.jpg')
    for image_path in images:
        landmarks_2d, landmarks_3d = get_landmark_2d_3d(image_path)
        if landmarks_2d is None or landmarks_3d is None:
            continue
        # ファイル名の取得（拡張子なし）
        filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        # 出力ファイルパスの生成
        output_file_path_2d = os.path.join(output_folder_2d,f"{filename_without_ext}.npy")
        output_file_path_3d = os.path.join(output_folder_3d,f"{filename_without_ext}.npy")
        np.save(output_file_path_2d, landmarks_2d)
        print(f"2D landmarks saved to {output_file_path_2d}")
        np.save(output_file_path_3d, landmarks_3d)
        print(f"3D landmarks saved to {output_file_path_3d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str,  help='Face images folder path')
    parser.add_argument('-o', type=str,  help='Output folder path')
    main(parser.parse_args())
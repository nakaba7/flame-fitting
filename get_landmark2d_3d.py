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
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')

    input = io.imread(image_path)

    landmarks_2d_result = fa_2d.get_landmarks(input)
    landmarks_3d_result = fa_3d.get_landmarks(input)

    if landmarks_2d_result is None or landmarks_3d_result is None:# 顔が検出されなかった場合
        print(f"Error: No face detected in {image_path}")
        return None, None
    landmarks_2d = np.array(landmarks_2d_result)
    landmarks_3d = np.array(landmarks_3d_result)
    if landmarks_2d.shape[0] != 1 or landmarks_3d.shape[0] != 1:# 顔が複数検出された場合
        print(f"Error: Many faces detected in {image_path}")
        return None, None
    landmarks_2d = np.squeeze(landmarks_2d)
    landmarks_3d = np.squeeze(landmarks_3d)
    if landmarks_2d.shape[0] != 68 or landmarks_3d.shape[0] != 68:# ランドマークが正しい数で検出されなかった場合
        print(f"Error: Only few landmarks detected in {image_path}")
        return None, None
    landmarks_2d = landmarks_2d[17:]
    landmarks_3d = landmarks_3d[17:]

    return landmarks_2d, landmarks_3d

def main(args):
    folder_name = args.f
    output_folder = args.o
    annotation_num = args.n
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder_2d = os.path.join(output_folder, "2d")
    output_folder_3d = os.path.join(output_folder, "3d")
    if not os.path.exists(output_folder_2d):
        os.makedirs(output_folder_2d)
    if not os.path.exists(output_folder_3d):
        os.makedirs(output_folder_3d)
    print(f"Start searching {folder_name} for img files...")
    images = glob.glob(f'{folder_name}/*.jpg')
    print(f"Found {len(images)} images")
    count = 0
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
        #print(f"2D landmarks saved to {output_file_path_2d}")
        np.save(output_file_path_3d, landmarks_3d)
        #print(f"3D landmarks saved to {output_file_path_3d}")
        print(f"Image {count+1}/{annotation_num} processed")
        count += 1
        if count == annotation_num:
            print("Annotation finished")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default = 'img_align_celeba', type=str,  help='Face images folder path')
    parser.add_argument('-o', default = 'output_landmark', type=str,  help='Output folder path')
    parser.add_argument('-n', default=10, type=int,  help='Number of images to be annotated')
    main(parser.parse_args())
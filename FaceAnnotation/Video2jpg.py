import cv2
import os
import argparse

"""
動画ファイルのすべてのフレームを静止画の画像ファイルとして切り出して保存するサンプルコード。
指定したディレクトリに<basename>_<連番>.<拡張子>というファイル名で保存する。
元コード: https://note.nkmk.me/python-opencv-video-to-still-image/
"""

import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    # 絶対パスを取得する
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_video_path = os.path.join(script_dir, video_path)
    abs_dir_path = os.path.join(script_dir, dir_path)

    cap = cv2.VideoCapture(abs_video_path)

    if not cap.isOpened():
        raise IOError("Error: Video file could not be opened.")

    os.makedirs(abs_dir_path, exist_ok=True)
    base_path = os.path.join(abs_dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            print('All frames have been saved.')
            break

    cap.release()

def main(args):
    input_video_path = args.i
    output_folder_path = args.o
    output_folder_path = os.path.join('FaceData', output_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    save_all_frames(input_video_path, output_folder_path, 'frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='hoge.mp4', type=str,  help='input video file name')
    parser.add_argument('-o', default='hoge', type=str,  help='output folder name of the images to be saved')
    main(parser.parse_args())

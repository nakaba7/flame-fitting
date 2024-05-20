from CameraCalibration.Combine3lmks import combine3lmks
from Convert2d_2_3d import convert2d_2_3d
import numpy as np
import matplotlib.pyplot as plt
from lmk_plot import plot_2d_3d_compare
import torch
import os


if __name__ == '__main__':
    # 各.npyファイルからデータを読み込み
    lmk_left = np.load("FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")
    lmk_mouth = np.load("FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")
    lmk_right = np.load("FaceData/FaceImages_Annotated/NPYs/test6_1_annotated.npy")
    model_path = 'DepthOnly_200000.pth'
    all_data = combine3lmks(lmk_right, lmk_mouth, lmk_left)
    all_data_3d = convert2d_2_3d(model_path, all_data)
    all_data_3d = all_data_3d.cpu().detach().numpy()
    output_dir = 'output_landmark/estimated_3d'
    file_path = os.path.join(output_dir,"quest_3d.npy")
    np.save(file_path, all_data_3d)
    print(f"Saved to {file_path}")
    plot_2d_3d_compare(all_data, all_data_3d)
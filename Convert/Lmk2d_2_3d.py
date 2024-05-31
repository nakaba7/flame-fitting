import torch
import torch.nn as nn
import numpy as np
import os
import glob
import sys
# 親ディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)
from MachineLearning.SimpleNet import DepthPredictionModel
from MachineLearning.CNNModel import CNNPredictor
from MachineLearning.Centering_lmk import centering_lmk
from Lmk_plot import plot_3lmk

"""
2次元ランドマークを3次元ランドマークに変換するスクリプト. 学習済みの変換器を使う. 
Usage:
    python Convert/Lmk2d_2_3d.py
"""


def lmk2d_2_3d(model_path, landmark_2d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 2
    hidden_dim = 1024
    output_dim = 1
    #model = DepthPredictionModel(input_dim, hidden_dim, output_dim).to(device)
    model = CNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    landmark_2d = torch.tensor(landmark_2d, dtype=torch.float32).to(device)
    landmark_2d = landmark_2d.unsqueeze(0)  # バッチ次元の追加
    estimated_depth = model(landmark_2d)
    landmark_3d = torch.cat((landmark_2d.squeeze(0), estimated_depth.squeeze(0)), dim=1)
    landmark_3d_np = landmark_3d.detach().cpu().numpy()
    return landmark_3d_np

def main():
    model_path = 'DepthOnly_200000.pth'
    input_dir = 'output_landmark/2d/test'
    output_dir = 'output_landmark/estimated_3d/test'
    target_dir = 'output_landmark/3d/test'  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_data = glob.glob(f'{input_dir}/*.npy')
    target_data = glob.glob(f'{target_dir}/*.npy')
    for input, target in zip(input_data, target_data):
        landmark_2d = np.load(input)
        reference_point = landmark_2d[16]
        landmark_2d -= reference_point
        landmark_3d = lmk2d_2_3d(model_path, landmark_2d)
        target_3d = np.load(target)
        reference_point = target_3d[16]
        target_3d -= reference_point
        print("input:", input)
        print("correct:", target)

        np.save(f'{output_dir}/{os.path.basename(input)}', landmark_3d)
        
        plot_3lmk(landmark_2d, landmark_3d, target_3d)

if __name__ == '__main__':
    main()

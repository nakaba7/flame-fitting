import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from MachineLearning.SimpleNet import DepthPredictionModel
from Plot_3lmk import plot_3lmk

"""
学習済みモデルを用いて2Dランドマークを3Dランドマークに変換するスクリプト
"""

def convert_2d_3d(model_path, landmark_2d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 2
    hidden_dim = 1024
    output_dim = 3
    model = DepthPredictionModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    landmark_2d = torch.tensor(landmark_2d, dtype=torch.float32).to(device)
    landmark_2d = landmark_2d.unsqueeze(0)  # バッチ次元の追加
    landmark_3d = model(landmark_2d).squeeze(0)  # バッチ次元を削除
    return landmark_3d
    
def main():
    model_path = 'Simple_2d_2_3d_200000.pth'
    input_dir = 'output_landmark/2d/test'
    output_dir = 'output_landmark/estimated_3d/test'
    target_dir = 'output_landmark/3d/test'  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_data = glob.glob(f'{input_dir}/*.npy')
    target_data = glob.glob(f'{target_dir}/*.npy')
    for input, target in zip(input_data, target_data):
        landmark_2d = np.load(input)
        landmark_3d = convert_2d_3d(model_path, landmark_2d)
        target_3d = np.load(target)
        print("input:", input)
        print("correct:", target)

        np.save(f'{output_dir}/{os.path.basename(input)}', landmark_3d.cpu().detach().numpy())
        #landmark_2d_np = landmark_2d.detach().cpu().numpy()
        landmark_3d_np = landmark_3d.detach().cpu().numpy()
        if landmark_3d_np.shape[0] == 1:
            landmark_3d_np = np.squeeze(landmark_3d_np)
        plot_3lmk(landmark_2d, landmark_3d_np, target_3d)

if __name__ == '__main__':
    main()
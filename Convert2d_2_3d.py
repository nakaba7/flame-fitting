import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from MachineLearning.LSTMFeatureMappingModel import LSTMFeatureMappingModel
from Plot2d_3d import plot_landmarks
import glob

"""
学習済みモデルを用いて2Dランドマークを3Dランドマークに変換するスクリプト
"""

def convert_2d_3d(model_path, landmark_2d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 2
    hidden_dim = 128
    output_dim = 3
    num_layers = 2
    model = LSTMFeatureMappingModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    landmark_2d = torch.tensor(landmark_2d, dtype=torch.float32).to(device)
    landmark_2d = landmark_2d.unsqueeze(0)
    landmark_3d = model(landmark_2d)
    return landmark_3d
    
def main():
    model_path = 'MachineLearning/2d_2_3d_model.pth'
    input_dir = 'output_landmark/test_2d'
    output_dir = 'output_landmark/estimated_3d'
    target_dir = 'output_landmark/test_3d'  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_data = glob.glob(f'{input_dir}/*.npy')
    target_data = glob.glob(f'{target_dir}/*.npy')
    for input, target in zip(input_data, target_data):
        landmark_2d = np.load(input)
        landmark_3d = convert_2d_3d(model_path, landmark_2d)
        target_3d = np.load(target)
        print("2d:", landmark_2d)
        print("3d:", landmark_3d)
        print("target:", target_3d)
        np.save(f'{output_dir}/{os.path.basename(input)}', landmark_3d.cpu().detach().numpy())
        # 勾配追跡を停止し、CUDA TensorをCPUに移動し、NumPy配列に変換
        #landmark_3d_np = landmark_3d.detach().cpu().numpy()
        #plot_landmarks(landmark_2d, landmark_3d_np)



if __name__ == '__main__':
    main()

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from MachineLearning.ParamPredictorModel import ParamPredictorModel
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def load_model(model_path, input_channel, output_dim, device):
    model = ParamPredictorModel(input_channel, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # モデルを評価モードに切り替える
    return model

def predict(model, input_data, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data).float().to(device)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(1)  # Add batch and channel dimensions
        output = model(input_tensor)
        prediction = output.cpu().numpy()
    return prediction

def get_valid_indices(expr_dir, pose_dir):
    expr_files = sorted([f for f in os.listdir(expr_dir) if f.startswith('test') and f.endswith('_expr.npy')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.startswith('test') and f.endswith('_pose.npy')])
    valid_indices = set(int(f[4:9]) for f in expr_files) & set(int(f[4:9]) for f in pose_files)
    return sorted(valid_indices)

def main():
    # パラメータ設定
    input_channel = 1
    output_dim = 65
    model_path = 'ParamsPredictor.pth'
    input_csv = 'sensor_values/Nakabayashi/sensor_data_test_filtered.csv'
    output_dir = 'output_params/estimated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    expr_dir = 'output_params/Nakabayashi/expr'
    pose_dir = 'output_params/Nakabayashi/pose'

    valid_indices = get_valid_indices(expr_dir, pose_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 新しいデータを読み込む
    inputs_csv = pd.read_csv(input_csv, header=None).values

    # モデルを読み込む
    model = load_model(model_path, input_channel, output_dim, device)

    filename_count=0
    # 各行に対して予測を行い、保存する
    for i, input_data in tqdm(enumerate(inputs_csv)):
        prediction = predict(model, input_data, device)
        while True:
            if filename_count in valid_indices:
                output_npy_path = os.path.join(output_dir, f'predict{filename_count:05d}.npy')
                break
            else:
                filename_count+=1
            
        np.save(output_npy_path, prediction)
        #print(f"Prediction for row {i} saved to {output_npy_path}")
        filename_count+=1

if __name__ == '__main__':
    main()

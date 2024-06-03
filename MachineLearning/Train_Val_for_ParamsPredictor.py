import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
from EarlyStopping import EarlyStopping
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from ParamPredictorModel import ParamPredictorModel
#import wandb

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def combine_expr_pose_npy_files(expr_dir, pose_dir, output_file):
    expr_files = sorted([f for f in os.listdir(expr_dir) if f.endswith('_expr.npy')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('_pose.npy')])
    
    combined_data = []

    for expr_file, pose_file in tqdm(zip(expr_files, pose_files), desc="Combining expr and pose files", total=len(expr_files)):
        expr_path = os.path.join(expr_dir, expr_file)
        pose_path = os.path.join(pose_dir, pose_file)
        
        expr_data = np.load(expr_path)
        pose_data = np.load(pose_path)
        
        if expr_data.shape[0] != 50 or pose_data.shape[0] != 15:
            print(f"Skipping {expr_file} and {pose_file} due to mismatched shapes.")
            continue
        
        combined = np.concatenate((expr_data, pose_data))
        combined_data.append(combined)
    
    combined_data = np.array(combined_data)
    np.save(output_file, combined_data)
    print(f"Combined data saved to {output_file}")

def get_valid_indices(expr_dir, pose_dir):
    expr_files = sorted([f for f in os.listdir(expr_dir) if f.startswith('test') and f.endswith('_expr.npy')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.startswith('test') and f.endswith('_pose.npy')])
    valid_indices = set(int(f[4:9]) for f in expr_files) & set(int(f[4:9]) for f in pose_files)
    return sorted(valid_indices)

def filter_csv_by_indices(input_csv, output_csv, valid_indices):
    # CSVファイルを読み込む
    csv_data = pd.read_csv(input_csv, header=None)
    csv_data = csv_data.values
    new_csv_data = []
    for i in range(len(csv_data)):
        if i in valid_indices:
            new_csv_data.append(csv_data[i])
        else:
            print(f"Skipping index {i}")
    print(f"Filtered data size: {len(new_csv_data)}")
    np.savetxt(output_csv, new_csv_data, delimiter=',', fmt='%s')
    print(f"Filtered data saved to {output_csv}")

def main():
    # パラメータ設定
    input_dim = 16
    output_dim = 65
    num_epochs = 10000
    learning_rate = 1e-5
    batch_size = 64
    """
    wandb.init(
        project="Params_Prediction",
        config={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
    )
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_csv = 'sensor_values/Nakabayashi/sensor_data_test.csv'
    output_csv_for_traindata = 'sensor_values/Nakabayashi/sensor_data_test_filtered.csv'
    param_dir_list = glob.glob("output_params/Nakabayashi/expr/*.npy")
    param_file_num = len(param_dir_list)
    target_npy_filepath = f"output_params/Nakabayashi/target_params_{param_file_num}.npy"
    expr_dir = 'output_params/Nakabayashi/expr'
    pose_dir = 'output_params/Nakabayashi/pose'
    if not os.path.exists(target_npy_filepath):
        output_file = target_npy_filepath
        combine_expr_pose_npy_files(expr_dir, pose_dir, output_file)
        valid_indices = get_valid_indices(expr_dir, pose_dir)
        filter_csv_by_indices(input_csv, output_csv_for_traindata, valid_indices)

    inputs_csv = pd.read_csv(output_csv_for_traindata, header=None).values  # pandas DataFrameをnumpy arrayに変換
    targets_npy = np.load(target_npy_filepath)

    dataset = CustomDataset(inputs_csv, targets_npy)
    print("dataset size:", len(dataset))
    #print(dataset.inputs)
    print(dataset.inputs.shape, dataset.targets.shape)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ParamPredictorModel(input_dim, output_dim).to(device)
    earlystopping = EarlyStopping(patience=20, verbose=True, path='ParamsPredictor.pth')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()  # データをfloat型に変換
            print(inputs.shape)
            #inputs = inputs.unsqueeze(1)  # バッチ次元とチャネル次元を追加
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        #wandb.log({"Training Loss": loss.item()})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.float()  # データをfloat型に変換
                if inputs.dim() == 2:  # inputsが2次元の場合にのみ次元変更を行う
                    inputs = inputs.unsqueeze(1)  # バッチ次元とチャネル次元を追加
                elif inputs.dim() == 3:
                    inputs = inputs.transpose(1, 2)  # 3次元の場合、チャネル次元を変更
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_dataloader)
            earlystopping(val_loss, model)
            if earlystopping.early_stop:
                print("Early Stopping!")
                break

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, lr: {scheduler.get_last_lr()}')
        print("====================================")
        #wandb.log({"Validation Loss": val_loss})
        scheduler.step()

if __name__ == '__main__':
    main()

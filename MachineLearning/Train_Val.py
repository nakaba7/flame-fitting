import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from EarlyStopping import EarlyStopping
from SimpleNet import DepthPredictionModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from CNNModel import CNNPredictor
import wandb
"""
作成したネットワークで学習を行うスクリプト. flame-fittingディレクトリの下で実行する.
ネットワークの候補はSimpleNet.py, CNNModel.pyの中から選択する.
Usage:
    $ python Train_Val.py

"""

class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir, data_size):
        self.datasize = data_size
        self.inputs = self.load_data(input_dir, f"inputs_cache_{data_size}.npy", '2d')
        self.targets = self.load_data(target_dir, f"targets_cache_{data_size}.npy", '3d')
        self.targets = self.targets[:, :, 2]  # 3次元目のみを取得
        self.targets = np.expand_dims(self.targets, axis=2)
        #print("targets shape:", self.targets.shape)

    def load_data(self, directory, cache_file_name, data_type):
        #cache_path = os.path.join(directory, cache_file_name)
        cache_path = cache_file_name
        if os.path.exists(cache_path):
            data_list = np.load(cache_path)
        else:
            print(f"Cache not found. Loading data from {directory}")
            files = sorted([file for file in os.listdir(directory) if file.endswith('.npy')])
            data_list = []
            count = 0
            for file in tqdm(files):
                if count == self.datasize:
                    break
                data_path = os.path.join(directory, file)
                data = np.load(data_path)
                if data.shape[0] > 51:
                    data = data[:51]
                elif data.shape[0] < 51:
                    continue
                data_list.append(data)
                count += 1
            data_list = np.stack(data_list)
            np.save(cache_path, data_list)
        return data_list

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def weighted_mse_loss(input, target, weight=1):
    # input, targetの形状: [BatchSize, SequenceSize, 3]
    # 3次元目の誤差に重みをかける
    loss = (input - target) ** 2
    loss[:,:,2] *= weight  # 3次元目の誤差に重みをかける
    return loss.mean()

def main():
    # パラメータ設定
    input_dim = 2
    hidden_dim = 1024
    output_dim = 1
    dataset_size = 200000
    #model_path = f'Simple_2d_2_3d_{dataset_size}.pth'
    if output_dim == 1:
        model_path = f'DepthOnly_{dataset_size}.pth'
    elif output_dim == 3:
        model_path = f'2d_2_3d_{dataset_size}.pth'
    epoch_num = 1000
    learning_rate = 1e-5
    batch_size = 64
    
    wandb.init(
        project="Simple_2d_3d_landmark",
        config={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "model_path": model_path,
            "epoch_num": epoch_num,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset_size": dataset_size
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = 'output_landmark/2d'
    target_dir = 'output_landmark/3d'
    dataset = CustomDataset(input_dir, target_dir, dataset_size)
    print("dataset size:", len(dataset))
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #model = DepthPredictionModel(input_dim, hidden_dim, output_dim).to(device)
    model = CNNPredictor().to(device)
    earlystopping = EarlyStopping(patience=20, verbose=True, path=model_path)

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(epoch_num)):
        model.train()
        for inputs, targets in train_dataloader:
            #print("train",inputs.shape, targets.shape)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        wandb.log({"Training Loss": loss.item()})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_dataloader:
                #print("val",inputs.shape, targets.shape)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_dataloader)
            earlystopping(val_loss, model)
            if earlystopping.early_stop:
                print("Early Stopping!")
                break

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, lr: {scheduler.get_last_lr()}')
        print("====================================")
        wandb.log({"Validation Loss": val_loss})
        scheduler.step()

if __name__ == '__main__':
    main()


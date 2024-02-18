import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from EarlyStopping import EarlyStopping
from LSTMFeatureMappingModel import LSTMFeatureMappingModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import wandb


class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir, data_size):
        self.datasize = data_size
        self.inputs = self.load_data(input_dir, f"inputs_cache_{data_size}_normalized.npy")
        self.targets = self.load_data(target_dir, f"targets_cache_{data_size}_normalized.npy")

    def load_data(self, directory, cache_file_name):
        cache_path = os.path.join(directory, cache_file_name)
        print(f"cache_path: {cache_path}")
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            data_list = np.load(cache_path)
        else:
            print(f"Cache not found. Loading data from {directory}")
            files = sorted(os.listdir(directory))# ファイル名をソート
            data_list = []
            count = 0
            for file in tqdm(files):
                if count == self.datasize:
                    break
                if file.endswith('.npy'):
                    data_path = os.path.join(directory, file)
                    data = np.load(data_path)
                    if data.shape[0] > 51:# 特徴点の数を51に統一
                        data = data[:51]
                    elif data.shape[0] < 51:
                        break
                    data_list.append(data)
                    count += 1
            data_list = np.stack(data_list)
            np.save(cache_path, data_list)  # キャッシュとして保存
        return data_list

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def weighted_mse_loss(input, target, weight=10.0):
    # input, targetの形状: [BatchSize, SequenceSize, 3]
    # 3次元目の誤差に重みをかける
    loss = (input - target) ** 2
    loss[:,:,2] *= weight  # 3次元目の誤差に重みをかける
    return loss.mean()

def main():
    # パラメータ設定
    input_dim = 2
    hidden_dim = 256
    output_dim = 3
    num_layers = 2
    model_path = '2d_2_3d_model.pth'
    epoch_num = 1000
    learning_rate = 1e-5
    batch_size = 64
    dataset_size = 100000
    wandb.init(
        project="2D_3d_landmark",
        config={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "model_path": model_path,
            "epoch_num": epoch_num,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset_size": dataset_size
        }
        )

    # GPUが利用可能か確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットとデータローダーの準備
    input_dir = '../output_landmark'
    target_dir = '../output_landmark'
    dataset = CustomDataset(input_dir, target_dir, dataset_size)
    print("dataset size:",len(dataset))
    train_size = int(len(dataset) * 0.8)# 80%を訓練データに
    val_size = len(dataset) - train_size # 残りを検証データに
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルインスタンスを作成し、GPUに移動
    model = LSTMFeatureMappingModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    earlystopping = EarlyStopping(patience=20, verbose=True, path=model_path)

    # 損失関数と最適化関数
    #criterion = nn.L1Loss()
    criterion = nn.HuberLoss()
    #criterion = weighted_mse_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 10エポックごとに学習率を0.1倍する
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 訓練ループ
    for epoch in tqdm(range(epoch_num)):
        model.train()  # モデルを訓練モードに設定
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            #print("input: ", inputs.shape)
            #print("target: ", targets.shape)
            optimizer.zero_grad()        # 勾配をゼロにする
            outputs = model(inputs)       # モデルによる予測
            loss = criterion(outputs, targets)  # 損失の計算
            loss.backward()              # 誤差逆伝播
            optimizer.step()             # パラメータ更新
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        wandb.log({"Training Loss": loss.item()})

        # 評価モード
        model.eval()  # モデルを評価モードに設定
        with torch.no_grad():  # 勾配計算を無効化
            val_loss = 0
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_dataloader)
            earlystopping(val_loss, model)
            if earlystopping.early_stop:
                print("Early Stopping!")
                break
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, lr: {scheduler.get_last_lr()}')
        wandb.log({"Validation Loss": val_loss})
        scheduler.step()

if __name__ == '__main__':
    main()

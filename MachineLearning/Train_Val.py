import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from EarlyStopping import EarlyStopping
from LSTMFeatureMappingModel import LSTMFeatureMappingModel
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.inputs = self.load_data(input_dir)
        self.targets = self.load_data(target_dir)

    def load_data(self, directory):
        files = sorted(os.listdir(directory))  # ファイルリストを取得し、ソート
        data_list = []
        for file in files:
            if file.endswith('.npy'):
                data_path = os.path.join(directory, file)
                data = np.load(data_path)
                data_list.append(data)
        return np.concatenate(data_list, axis=0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def main():
    # パラメータ設定
    input_dim = 2
    hidden_dim = 128
    output_dim = 3
    num_layers = 2
    model_path = '2d_2_3d_model.pth'
    epoch_num = 100
    learning_rate = 0.001
    batch_size = 16

    # GPUが利用可能か確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットとデータローダーの準備
    input_dir = '../output_landmark/2d'
    target_dir = '../output_landmark/3d'
    dataset = CustomDataset(input_dir, target_dir)
    train_size = int(len(dataset) * 0.8)# 80%を訓練データに
    val_size = len(dataset) - train_size # 残りを検証データに
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルインスタンスを作成し、GPUに移動
    model = LSTMFeatureMappingModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    earlystopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    # 損失関数と最適化関数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練ループ
    for epoch in tqdm(range(epoch_num)):
        model.train()  # モデルを訓練モードに設定
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()        # 勾配をゼロにする
            outputs = model(inputs)       # モデルによる予測
            loss = criterion(outputs, targets)  # 損失の計算
            loss.backward()              # 誤差逆伝播
            optimizer.step()             # パラメータ更新
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

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
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

if __name__ == '__main__':
    main()

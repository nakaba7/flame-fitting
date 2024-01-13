import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from EarlyStopping import EarlyStopping
"""

このコードは、PyTorchを使用してランドマーク検出モデルを構築し、学習するための基本的なフレームワークです。ここには、以下のコンポーネントが含まれています

- `LandmarkDetectionNet`: 畳み込み層と全結合層を持つネットワークのクラス定義です。
- `FacialLandmarksDataset`: 画像とランドマークを読み込むためのカスタムデータセットクラスです。`transforms`を使用して、データの前処理と拡張を行います。
- `DataLoader`: バッチ処理、シャッフル、マルチスレッドデータ読み込みを行うためのデータローダーの設定です。
- 学習ループ: 各エポックでデータセットをイテレートし、バックプロパゲーションによりネットワークを学習させます。

実際にデータをロードし、ネットワークをトレーニングする前に、CSVファイルのパスやデータセットのディレクトリ、変換のパラメータ、学習率、エポック数など、
必要に応じてコード内のパラメータを調整してください。また、画像サイズやランドマークの数に応じて、ネットワークアーキテクチャ内のパラメータ
（特に`self.fc1`の入力次元数）も適宜変更する必要があります。

"""

# ネットワークアーキテクチャの定義
class LandmarkDetectionNet(nn.Module):
    def __init__(self):
        super(LandmarkDetectionNet, self).__init__()
        # 特徴抽出層
        self.conv1 = nn.Conv2D(32, 3, padding=1)
        self.conv2 = nn.Conv2D(64, 3, padding=1)
        self.conv3 = nn.Conv2D(128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # ランドマーク検出層
        self.fc1 = nn.Linear(128 * 12 * 12, 512)  # 画像サイズに応じて調整する
        self.fc2 = nn.Linear(512, 18)  # 9個のランドマークを想定（x, y）なので出力は18

    def forward(self, x):
        # 特徴抽出
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)  # Flatten the tensor
        # ランドマーク検出
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# データセットの定義
class FacialLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # CSVファイルからランドマークデータを読み込む
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # 画像読み込み
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    model_path = 'model.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    earlystopping = EarlyStopping(patience=5, verbose=True, path=model_path)
    # データセットとデータローダーの準備
    transformed_dataset = FacialLandmarksDataset(csv_file='landmarks.csv',
                                                root_dir='dataset/',
                                                transform=transforms.Compose([
                                                    Rescale(256),
                                                    RandomCrop(224),
                                                    ToTensor()
                                                ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    # ネットワークのインスタンス化と損失関数、オプティマイザの定義
    net = LandmarkDetectionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 200

    # 学習ループ
    for epoch in range(num_epochs):  # num_epochsはあらかじめ定義しておく必要があります
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # ネットワークへの入力データと正解ラベルを取得
            inputs, labels = data['image'], data['landmarks']
            # 勾配情報をゼロに初期化
            optimizer.zero_grad()
            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 統計情報の更新
            running_loss += loss.item()
            earlystopping(running_loss, net)
            if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                print("Early Stopping!")
                #print("val acc : ",last_val_acc.item())
                break
            if i % 1000 == 999:    # 1000ミニバッチごとに損失を出力
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Finished Training')

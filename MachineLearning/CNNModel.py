import torch
import torch.nn as nn

class CNNPredictor(nn.Module):
    def __init__(self):
        super(CNNPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 51, 128)
        self.fc2 = nn.Linear(128, 51)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 入力サイズを (batch_size, 2, 51) に変換
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # フラット化
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 51, 1)  # 出力サイズを (batch_size, 51, 1) に変換
        return x
"""
# モデルのインスタンス化
model = CNNPredictor()

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ダミーデータの作成 (バッチサイズ64、特徴点51個、2次元)
input_data = torch.randn(64, 51, 2)
target_data = torch.randn(64, 51, 1)

# トレーニングループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
"""
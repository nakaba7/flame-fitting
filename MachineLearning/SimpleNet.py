import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DepthPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim * 51, hidden_dim)  # 入力層
        self.bn1 = nn.BatchNorm1d(hidden_dim)             # バッチ正規化
        self.drop = nn.Dropout(0.3)                       # ドロップアウト
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # 中間層
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim * 51)  # 出力層

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 入力をフラット化
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.fc3(x)
        x = x.view(x.size(0), 51, 3)  # 出力を適切な形状にリシェイプ
        return x


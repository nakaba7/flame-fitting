import torch
import torch.nn as nn
import torch.nn.functional as F 

# GPUが利用可能か確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTMモデル定義
class LSTMFeatureMappingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMFeatureMappingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropped = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(51)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # LSTMの隠れ状態とセル状態の初期化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropped(x)

        x = self.hidden_layer(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropped(x)
        """
        x = self.hidden_layer(x)
        x = x.reshape(b, s, self.hidden_dim)
        x = x.permute(0,2,1)
        x = self.batch_norm(x)
        x = x.permute(0,2,1)
        x = F.relu(x)
        x = self.dropped(x)
        """
        x, _ = self.lstm(x, (h0, c0))
        x = F.relu(x)

        x = self.output_layer(x)

        return x


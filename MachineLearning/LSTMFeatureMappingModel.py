import torch
import torch.nn as nn

# LSTMモデル定義
class LSTMFeatureMappingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMFeatureMappingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTMレイヤー
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 出力層
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTMの隠れ状態とセル状態の初期化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM層
        out, _ = self.lstm(x, (h0, c0))
        
        # 出力層
        out = self.fc(out)
        return out


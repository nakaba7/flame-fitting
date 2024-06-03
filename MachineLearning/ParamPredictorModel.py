import torch
import torch.nn as nn
import torch.optim as optim

class ParamPredictorModel(nn.Module):
    def __init__(self, input_channels, num_params):
        super(ParamPredictorModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(64 * 4, 128)  # assuming input size is 16
        self.fc2 = nn.Linear(128, num_params)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # flatten the tensor
        out = self.fc1(out)
        out = self.fc2(out)
        return out

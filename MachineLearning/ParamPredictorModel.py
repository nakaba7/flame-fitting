import torch
import torch.nn as nn
import torch.optim as optim

class ParamPredictorModel(nn.Module):
    def __init__(self, input_channels, num_params):
        super(ParamPredictorModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input_channels = input_channels
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 4, 128)  # assuming input size is 16
        self.fc2 = nn.Linear(128, num_params)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

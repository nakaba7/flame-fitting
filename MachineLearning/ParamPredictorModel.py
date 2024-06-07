import torch
import torch.nn as nn
import torch.optim as optim

class ParamPredictorModel(nn.Module):
    def __init__(self, input_channels, num_params):
        super(ParamPredictorModel, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 4, 128)  # assuming input size is 16
        self.fc2 = nn.Linear(128, num_params)

    def forward(self, x):
        #print(1,x.shape)
        x = self.conv1(x)
        #print(2,x.shape)
        x = self.bn1(x)
        #print(3, x.shape)
        x = self.relu1(x)
        #print(4, x.shape)
        x = self.pool1(x)
        #print(5, x.shape)
        
        x = self.conv2(x)
        #print(6, x.shape)
        x = self.bn2(x)
        #print(7, x.shape)
        x = self.relu2(x)
        #print(8, x.shape)
        x = self.pool2(x)
        #print(9, x.shape)
        
        x = x.view(x.size(0), -1)  # flatten the tensor
        #print(10, x.shape)
        x = self.fc1(x)
        #print(11, x.shape)
        x = self.fc2(x)
        #print(12, x.shape)
        
        return x

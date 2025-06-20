# cnn_model_improved_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesCNNImprovedV2(nn.Module):
    def __init__(self, input_channels: int, seq_length: int):
        super(TimeSeriesCNNImprovedV2, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.dropout_conv1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))


        x = self.global_pool(x)
        x = x.squeeze(2)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)
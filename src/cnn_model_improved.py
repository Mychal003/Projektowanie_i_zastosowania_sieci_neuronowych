import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesCNNImproved(nn.Module):
    def __init__(self, input_channels: int, seq_length: int):
        super(TimeSeriesCNNImproved, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.identity = nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, input_channels)
        x = x.permute(0, 2, 1)  # -> (batch_size, input_channels, seq_length)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.squeeze(2)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

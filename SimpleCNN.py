import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # this is for my macbook pro

class SimpleCNN(nn.Module):
    '''
    architecture:
    - initial convolution layer to process input features and reduce dimensions to intermediate_channels (in_channels -> 32)
    - second convolution layer, processes features from conv1 (32 -> 32)
    - global average pooling and final classification layer (32 -> n_classes)
    '''
    def __init__(self, in_channels=248, n_classes=10, intermediate_channels=32, dropout_rate=0.2):
        super().__init__()
        # initial convolution layer to process input features and reduce dimensions to intermediate_channels
        # bc we know the input channels are redundant (95%+ of variance captured by 20 principal components)
        self.conv1 = nn.Conv1d(in_channels, intermediate_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        # second convolution layer, processes features from conv1
        self.conv2 = nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # global average pooling and final classification layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(intermediate_channels, n_classes)

    def forward(self, x):
        # process input through initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # second convolution layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # global pooling and classification
        x = self.gap(x)
        x = x.squeeze(-1)
        return self.fc(x)
    
    
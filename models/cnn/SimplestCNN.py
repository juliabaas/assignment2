import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

# most basic CNN imaginable
# 1. initial convolution layer (bottleneck) - kernel size fixed at 7
# 2. additional convolutional layers with specified kernel sizes
# 3. global average pooling and final classification layer)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=248, n_classes=10, intermediate_channels=32, dropout_rate=0.2, kernel_sizes_extra=[3, 3]):
        super().__init__()
        
        # initial convolution layer (bottleneck) - kernel size fixed at 7
        self.conv1 = nn.Conv1d(in_channels, intermediate_channels, kernel_size=7, padding=3) # padding = 7//2
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # additional convolutional layers with specified kernel sizes
        for ks in kernel_sizes_extra:
            self.conv_layers.append(nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=ks, padding=ks//2))
            self.bn_layers.append(nn.BatchNorm1d(intermediate_channels))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
        # global average pooling and final classification layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(intermediate_channels, n_classes)

    def forward(self, x):
        # initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # additional convolutional layers
        for i in range(len(self.conv_layers)):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))
            x = self.dropout_layers[i](x)
            
        # global pooling and classification
        x = self.gap(x)
        x = x.squeeze(-1)
        return self.fc(x)
    

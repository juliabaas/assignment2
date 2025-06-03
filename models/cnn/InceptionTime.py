import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, n_filters=32, kernels=[9, 19, 39], bottleneck=32, dropout=0.2):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1)
        self.bn1 = nn.BatchNorm1d(bottleneck)
        self.drop1 = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck, n_filters, k, padding=k//2, bias=False) 
            for k in kernels
        ])
        
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(bottleneck, n_filters, 1)
        self.bn2 = nn.BatchNorm1d(n_filters * len(kernels) + n_filters)
        self.drop2 = nn.Dropout(dropout)
        
        self.residual = None
        out_ch = n_filters * len(kernels) + n_filters
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, 1)
        
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.bottleneck(x)))
        x = self.drop1(x)
        
        conv_outs = [conv(x) for conv in self.convs]
        pool_out = self.pool_conv(self.pool(x))
        
        x = torch.cat(conv_outs + [pool_out], dim=1)
        x = self.bn2(x)
        x = self.drop2(x)
        
        if self.residual is not None:
            res = self.residual(res)
        x = x + res
        return F.relu(x)

class InceptionNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=10, n_filters=32, n_blocks=2, bottleneck=32, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, n_filters, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.drop1 = nn.Dropout(dropout)
        
        out_ch = n_filters * 4
        
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        
        in_channels = n_filters
        for i in range(n_blocks):
            self.blocks.append(InceptionBlock(in_channels, n_filters, bottleneck=bottleneck, dropout=dropout))
            if i > 0:
                self.residuals.append(nn.Conv1d(in_channels, out_ch, 1))
            in_channels = out_ch
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(out_ch, n_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0:
                res = self.residuals[i-1](res)
                x = x + res
            res = x
            
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.drop2(x)
        return self.fc(x)

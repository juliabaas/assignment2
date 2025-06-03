import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



'''
implemented from Zubarev et al. 2019
https://www.sciencedirect.com/science/article/pii/S1053811919303544

difference between the two models:
- LF_CNN: separate 1d convolutions to each channel (no interaction between channels)
- VAR_CNN: spatiotemporal convolution (models interactions between channels)
''' 

class LF_CNN(nn.Module):
    """linear filter cnn - applies separate 1d convolutions to each spatial component"""
    
    def __init__(self, n_channels=248, n_sources=32, n_classes=5, 
                 filter_len=7, dropout=0.5, l1_penalty=3e-4):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_sources = n_sources
        self.filter_len = filter_len
        self.l1_penalty = l1_penalty
        
        # spatial projection
        self.spatial_filters = nn.Linear(n_channels, n_sources, bias=False)
        
        # temporal convolutions
        self.temp_convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=filter_len, padding=0)
            for _ in range(n_sources)
        ])
        
        self.pooling = nn.MaxPool1d(2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)  # fixed output size
        
        self.classifier = nn.Linear(n_sources * 100, n_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, n_ch, T = x.shape
        
        # spatial filtering
        x = x.transpose(1, 2)
        x = self.spatial_filters(x)
        x = x.transpose(1, 2)
        
        # temporal convolution per source
        temp_outs = []
        for i in range(self.n_sources):
            src = x[:, i:i+1, :]
            conv_out = self.temp_convs[i](src)
            temp_outs.append(conv_out)
        
        x = torch.stack([out.squeeze(1) for out in temp_outs], dim=1)
        x = F.relu(x)
        x = self.pooling(x)
        
        x = self.adaptive_pool(x)
        x = x.view(B, -1)
        
        return self.classifier(self.dropout(x))
    

class VAR_CNN(nn.Module):
    """vector autoregressive cnn - models interactions between spatial components"""
    
    def __init__(self, n_channels=248, n_sources=32, n_classes=5, 
                 filter_len=7, dropout=0.5, l1_penalty=3e-4):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_sources = n_sources
        self.filter_len = filter_len
        self.l1_penalty = l1_penalty
        
        self.spatial_filters = nn.Linear(n_channels, n_sources, bias=False)
        
        # spatiotemporal conv
        self.spatiotemp_conv = nn.Conv2d(
            1, n_sources,
            kernel_size=(filter_len, n_sources),
            padding=0
        )
        
        self.pooling = nn.MaxPool1d(2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)
        
        self.classifier = nn.Linear(n_sources * 100, n_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, n_ch, T = x.shape
        
        # spatial filtering
        x = x.transpose(1, 2)
        x = self.spatial_filters(x)
        x = x.transpose(1, 2)
        
        # spatiotemporal convolution
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.spatiotemp_conv(x).squeeze(-1)
        x = F.relu(x)
        x = self.pooling(x)
        
        x = self.adaptive_pool(x)
        x = x.view(B, -1)
        
        return self.classifier(self.dropout(x))
    
    def l1_loss(self):
        """l1 regularization for output layer"""
        if self.classifier is not None:
            return torch.sum(torch.abs(self.classifier.weight))
        return torch.tensor(0.0)




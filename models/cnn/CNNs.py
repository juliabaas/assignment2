import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """spatial attention module - attention across channels"""
    
    def __init__(self, n_sources, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=n_sources,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(n_sources)
        
    def forward(self, x):
        # x: (batch, sources, time)
        B, n_sources, T = x.shape
        
        x = x.transpose(1, 2)  # (batch, time, sources)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        
        return out.transpose(1, 2)

class DilatedBlock(nn.Module):
    """dilated temporal convolution module"""
    
    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(channels, channels, kernel_size, 
                             padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        return out + residual

class TemporalProcessor(nn.Module):
    """multi-scale temporal processing with dilated convolutions"""
    
    def __init__(self, n_sources, dilations=[1, 2, 4]):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            DilatedBlock(n_sources, dilation=d) for d in dilations
        ])
        self.fusion = nn.Conv1d(n_sources * len(dilations), n_sources, 1)
        
    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        concat = torch.cat(outs, dim=1)
        return self.fusion(concat)


# MODELS


class DilatedCNN(nn.Module):
    """simplified meg cnn with dilated convolutions"""
    
    def __init__(self, n_channels=248, n_classes=5, dropout=0.3, n_sources=32, hidden_size=64):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.bn1 = nn.BatchNorm1d(n_sources)
        
        # temporal processing
        self.temp_proc = TemporalProcessor(n_sources, dilations=[1, 2, 4])
        
        # classifier
        self.classifier = self._make_classifier(n_sources, n_classes, dropout, hidden_size)
        
    def _make_classifier(self, n_sources, n_classes, dropout, hidden_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_sources, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        
    def forward(self, x):
        # spatial projection
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        
        # temporal processing
        x = self.temp_proc(x)
        
        return self.classifier(x)


class AttentionCNN(nn.Module):
    """simplified meg cnn with spatial attention"""
    
    def __init__(self, n_channels=248, n_classes=5, dropout=0.3, n_sources=32, n_heads=4, hidden_size=64):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.attention = SpatialAttention(n_sources, n_heads)
        
        # progressive temporal convolutions - principled kernel sizes
        self.temp_convs = nn.ModuleList([
            self._make_conv_block(n_sources, k) for k in [7, 5, 3]
        ])
        
        self.classifier = self._make_classifier(n_sources, n_classes, dropout, hidden_size)
        
    def _make_conv_block(self, channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def _make_classifier(self, n_sources, n_classes, dropout, hidden_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_sources, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        
    def forward(self, x):
        # spatial processing
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        
        # spatial attention with residual
        x = x + self.attention(x)
        
        # progressive temporal processing
        for conv_block in self.temp_convs:
            x = conv_block(x)
        
        return self.classifier(x)


class BasicCNN(nn.Module):
    """basic meg cnn without attention or dilated convolutions"""
    
    def __init__(self, n_channels=248, n_classes=5, dropout=0.3, n_sources=32, hidden_size=64):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.bn_spatial = nn.BatchNorm1d(n_sources)
        
        # progressive temporal convolutions
        self.temp_convs = nn.ModuleList([
            self._make_conv_block(n_sources, k) for k in [7, 5, 3]
        ])
        
        self.classifier = self._make_classifier(n_sources, n_classes, dropout, hidden_size)
        
    def _make_conv_block(self, channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def _make_classifier(self, n_sources, n_classes, dropout, hidden_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_sources, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        
    def forward(self, x):
        # spatial projection
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.bn_spatial(x)
        
        # progressive temporal processing
        for conv_block in self.temp_convs:
            x = conv_block(x)
        
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """most basic cnn - configurable dimensions"""
    
    def __init__(self, n_channels=248, n_classes=10, dropout=0.3, intermediate_channels=32, hidden_size=64):
        super().__init__()
        
        # progressive kernel sizes - start large, get smaller
        kernel_sizes = [7, 5, 3]
        
        # initial projection
        self.conv1 = nn.Conv1d(n_channels, intermediate_channels, 
                              kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        
        # additional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(intermediate_channels, intermediate_channels, 
                     kernel_size=k, padding=k//2) for k in kernel_sizes[1:]
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(intermediate_channels) for _ in kernel_sizes[1:]
        ])
        
        # configurable classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # progressive convolutions
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
            
        return self.classifier(x)
    

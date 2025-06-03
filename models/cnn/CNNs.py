import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def __init__(self, n_sources, dilations=[1, 2, 4, 8]):
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
    
    def __init__(self, n_channels=204, n_sources=32, n_classes=5, dropout=0.5):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.bn1 = nn.BatchNorm1d(n_sources)
        
        self.temp_proc = TemporalProcessor(n_sources)
        
        self.temp_blocks = nn.Sequential(
            DilatedBlock(n_sources, dilation=1),
            DilatedBlock(n_sources, dilation=2),
            DilatedBlock(n_sources, dilation=4)
        )
        
        self.pool = nn.AdaptiveMaxPool1d(32)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_sources * 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        B = x.size(0)
        
        # spatial projection
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        
        # temporal projection
        x = self.temp_proc(x)
        x = self.temp_blocks(x)
        
        x = self.pool(x)
        x = x.view(B, -1)
        
        return self.classifier(x)


class AttentionCNN(nn.Module):
    """simplified meg cnn with spatial attention"""
    
    def __init__(self, n_channels=204, n_sources=32, n_classes=5, 
                 n_heads=8, dropout=0.5):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.attention = SpatialAttention(n_sources, n_heads)
        
        # temporal convs
        self.conv1 = nn.Conv1d(n_sources, n_sources, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_sources)
        
        self.conv2 = nn.Conv1d(n_sources, n_sources, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_sources)
        
        self.conv3 = nn.Conv1d(n_sources, n_sources, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_sources)
        
        self.pool = nn.AdaptiveMaxPool1d(32)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_sources * 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        B = x.size(0)
        
        # spatial processing
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        
        x_attn = self.attention(x)
        x = x + x_attn
        
        # temporal processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.pool(x)
        x = x.view(B, -1)
        
        return self.classifier(x)


class BasicCNN(nn.Module):
    """basic meg cnn without attention or dilated convolutions"""
    
    def __init__(self, n_channels=204, n_sources=32, n_classes=5, dropout=0.5):
        super().__init__()
        
        self.proj = nn.Linear(n_channels, n_sources)
        self.bn_spatial = nn.BatchNorm1d(n_sources)
        
        self.conv1 = nn.Conv1d(n_sources, n_sources, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_sources)
        
        self.conv2 = nn.Conv1d(n_sources, n_sources, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_sources)
        
        self.conv3 = nn.Conv1d(n_sources, n_sources, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_sources)
        
        self.pool = nn.AdaptiveMaxPool1d(32)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_sources * 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        bs = x.size(0)
        
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.bn_spatial(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.pool(x)
        x = x.view(bs, -1)
        
        return self.fc(x)


# Most simple of all


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
    

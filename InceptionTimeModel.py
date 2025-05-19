import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, dropout_rate=0.2):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv_paths = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, kernel_size=k, stride=1, padding=k//2, bias=False) 
            for k in kernel_sizes
        ])
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(n_filters * len(kernel_sizes) + n_filters)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.residual_projection = None
        if in_channels != n_filters * len(kernel_sizes) + n_filters:
            self.residual_projection = nn.Conv1d(in_channels, n_filters * len(kernel_sizes) + n_filters, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn_bottleneck(self.bottleneck(x)))
        x = self.dropout1(x)
        conv_outputs = [conv(x) for conv in self.conv_paths]
        maxpool_output = self.maxpool_conv(self.maxpool(x))
        combined = torch.cat(conv_outputs + [maxpool_output], dim=1)
        x = self.bn(combined)
        x = self.dropout2(x)
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        x = x + residual
        return F.relu(x)

class InceptionTime(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, n_filters=32, n_blocks=2, bottleneck_channels=32, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.dropout1 = nn.Dropout(dropout_rate)
        inception_output_channels = n_filters * 4
        
        self.blocks = nn.ModuleList()
        input_channels = n_filters
        self.residual_projections = nn.ModuleList()
        
        for i in range(n_blocks):
            self.blocks.append(InceptionBlock(input_channels, n_filters, bottleneck_channels=bottleneck_channels, dropout_rate=dropout_rate))
            if i > 0:
                self.residual_projections.append(nn.Conv1d(input_channels, inception_output_channels, kernel_size=1))
            input_channels = inception_output_channels
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(inception_output_channels, n_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0:
                residual = self.residual_projections[i-1](residual)
                x = x + residual
            residual = x
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.dropout2(x)
        return self.fc(x)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, patience=5, lr_patience=3, lr_factor=0.1, weight_decay=1e-5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val = batch_X_val.to(device)
                batch_y_val = batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_running_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                val_total += batch_y_val.size(0)
                val_correct += (predicted_val == batch_y_val).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # best_model_state = model.state_dict() 
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                # if best_model_state:
                #    model.load_state_dict(best_model_state)
                break 
    
    return model, history


def plot_training_history(history):
    """Plots the training and validation loss, accuracy, and learning rate from the history object."""
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Training and Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training and Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
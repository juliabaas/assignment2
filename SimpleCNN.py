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
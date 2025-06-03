import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cpu')


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, patience=5, lr_patience=3, lr_factor=0.1, weight_decay=1e-5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # model-specific l1 regularization
            if hasattr(model, 'l1_penalty') and model.l1_penalty > 0 and hasattr(model, 'l1_regularization_loss'):
                loss += model.l1_penalty * model.l1_regularization_loss()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs, 1)[1] == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                val_correct += (torch.max(outputs, 1)[1] == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break 
    
    return model, history


def plot_training_history(history):
    """plots the training and validation loss, accuracy, and learning rate from the history object."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    # plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
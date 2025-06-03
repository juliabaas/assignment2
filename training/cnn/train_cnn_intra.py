#!/usr/bin/env python3
"""
Training script for AttentionMEG model on MEG data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils import prepare_pytorch_dataloader
from utils.train_utils import train_model, plot_training_history
from models.cnn.CNNs import *
from models.cnn.Zubarev import VAR_CNN
import matplotlib.pyplot as plt

def main():
    # data paths
    intra_train_path = 'data/Intra/train'
    intra_test_path = 'data/Intra/test'
    
    # parameters
    downsample_factor = 5
    normalize = True
    train_batch_size = 4
    test_batch_size = 4
    random_seed = 42
    
    # model parameters
    n_channels = 248
    n_sources = 16
    n_heads = 2
    dropout = 0.5
    
    # training parameters
    epochs = 50
    learning_rate = 0.0005
    patience = 15
    lr_patience = 7
    lr_factor = 0.1
    weight_decay = 1e-4
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    device = torch.device("cpu")
    print(f"using device: {device}")
    
    le_intra = None
    channel_scalers_intra = None

    print("--- processing intra-subject training data ---")
    train_loader_intra, le_intra, channel_scalers_intra = prepare_pytorch_dataloader(
        intra_train_path, 
        batch_size=train_batch_size, 
        shuffle=True, 
        label_encoder=None,
        num_workers=0,
        downsample_factor=downsample_factor,
        normalize=normalize
    )
    
    if train_loader_intra and le_intra: 
        print(f"intra-subject training dataloader created. label mapping: {dict(zip(le_intra.classes_, le_intra.transform(le_intra.classes_)))}")
    elif not train_loader_intra:
        print("failed to create intra-subject training dataloader. exiting.")
        return

    print("--- processing intra-subject test data ---")
    val_loader_intra = None
    if le_intra: 
        val_loader_intra, _, _ = prepare_pytorch_dataloader(
            intra_test_path, 
            batch_size=test_batch_size, 
            shuffle=False,
            label_encoder=le_intra,
            channel_scalers=channel_scalers_intra,
            num_workers=0,
            downsample_factor=downsample_factor,
            normalize=normalize
        )
        if val_loader_intra:
            print("intra-subject test data loaded as validation dataloader.")
        else:
            print("failed to create intra-subject validation dataloader.")
    else:
        print("skipping intra-subject test data processing as training label encoder is missing.")

    if le_intra:
        num_classes = len(le_intra.classes_)
        print(f"number of classes: {num_classes}")
    else:
        print("error: label encoder not available. cannot determine number of classes. exiting.")
        return

    print("\n--- initializing model ---")
    model = VAR_CNN(
        n_channels=n_channels,
        n_sources=n_sources,
        n_classes=num_classes,
        filter_len=7,
        dropout=dropout,
        # l1_penalty=3e-4
    ).to(device)

    print("\nmodel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\ntotal parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")

    print("\n--- training model ---")
    if train_loader_intra and val_loader_intra:
        model, history = train_model(
            model=model,
            train_loader=train_loader_intra, 
            val_loader=val_loader_intra,
            epochs=epochs,
            lr=learning_rate,
            patience=patience,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            weight_decay=weight_decay
        )
        
        print("\n--- plotting training history ---")
        plot_training_history(history)
    else:
        print("skipping model training as training or validation data loader is missing.")
        return

    print("\ntraining completed successfully!")

if __name__ == '__main__':
    main() 
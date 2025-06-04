#!/usr/bin/env python3
"""
Training script for AttentionMEG model on MEG data.
Based on the existing training scripts and AttentionMEG.py model.
Adapted for Cross-subject data training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils import prepare_pytorch_dataloader
from utils.train_utils import train_model, plot_training_history
from models.cnn.CNNs import *
from models.cnn.Zubarev import VAR_CNN

def main():
    # Hyperparameters
    # Data paths - adapted for Cross-subject data
    cross_train_path = 'C:/Users/baasj/OneDrive - Universiteit Utrecht/Master AI/Deep Learning/Programming assignments/Final Project data/Cross/train'
    cross_test1_path = 'C:/Users/baasj/OneDrive - Universiteit Utrecht/Master AI/Deep Learning/Programming assignments/Final Project data/Cross/test1'  # Test subject 1
    cross_test2_path = 'C:/Users/baasj/OneDrive - Universiteit Utrecht/Master AI/Deep Learning/Programming assignments/Final Project data/Cross/test2'  # Test subject 2
    cross_test3_path = 'C:/Users/baasj/OneDrive - Universiteit Utrecht/Master AI/Deep Learning/Programming assignments/Final Project data/Cross/test3'  # Test subject 3
    
    # Data processing parameters - with downsampling and normalization
    DOWNSAMPLE_FACTOR = 20  # Factor to downsample input signals
    NORMALIZE = True       # Whether to normalize data
    TRAIN_BATCH_SIZE = 4   # Batch size
    TEST_BATCH_SIZE = 4    # Batch size for test/validation
    RANDOM_SEED = 42
    
    # Model parameters for AttentionMEG - reduced complexity
    N_CHANNELS = 248  # Number of MEG channels
    N_SOURCES = 32    # Reduced number of spatial sources (was 32)
    N_HEADS = 2       # Reduced number of attention heads (was 8)
    DROPOUT = 0.5     # Reduced dropout rate
    
    # Training parameters
    EPOCHS = 15        # Reduced epochs for faster training
    LEARNING_RATE = 0.0005  # Slightly higher learning rate
    PATIENCE = 15       # Reduced patience
    LR_PATIENCE = 7    # Reduced LR patience
    LR_FACTOR = 0.1    
    WEIGHT_DECAY = 1e-4
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    le_cross = None
    channel_scalers_cross = None # To store fitted scalers

    print("--- Processing Cross-subject Training Data ---")
    train_loader_cross, le_cross, channel_scalers_cross = prepare_pytorch_dataloader(
        cross_train_path, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True, 
        label_encoder=None, # Fit a new encoder
        num_workers=0,
        downsample_factor=DOWNSAMPLE_FACTOR,
        normalize=NORMALIZE
    )
    
    if train_loader_cross and le_cross: 
        print(f"Cross-subject training DataLoader created. Label mapping: {dict(zip(le_cross.classes_, le_cross.transform(le_cross.classes_)))}")
        
        # Debug: Check the first batch to see actual data dimensions
        print("--- Debugging: Checking training data dimensions ---")
        try:
            train_data_iter = iter(train_loader_cross)
            sample_batch, sample_labels = next(train_data_iter)
            print(f"Training batch shape: {sample_batch.shape}")
            print(f"Training labels shape: {sample_labels.shape}")
            print(f"Sample data type: {sample_batch.dtype}")
        except Exception as e:
            print(f"Error checking training data dimensions: {e}")
            
        # plot_data_samples(train_loader_cross, "Training Data", le_cross, num_samples=2, num_channels_to_plot=5, device=device, DOWNSAMPLE_FACTOR=DOWNSAMPLE_FACTOR, NORMALIZE=NORMALIZE)
    elif not train_loader_cross:
        print("Failed to create Cross-subject training DataLoader. Exiting.")
        return

    print("--- Processing Cross-subject Test Data from All Three Subjects ---")
    test_loaders = {}
    test_paths = {
        'Subject 1': cross_test1_path,
        'Subject 2': cross_test2_path, 
        'Subject 3': cross_test3_path
    }
    
    if le_cross:
        for subject_name, test_path in test_paths.items():
            print(f"Loading test data for {subject_name}...")
            test_loader, _, _ = prepare_pytorch_dataloader(
                test_path, 
                batch_size=TEST_BATCH_SIZE, 
                shuffle=False,
                label_encoder=le_cross, # Use the fitted encoder from training
                channel_scalers=channel_scalers_cross, # Use the fitted scalers from training
                num_workers=0,
                downsample_factor=DOWNSAMPLE_FACTOR,
                normalize=NORMALIZE
            )
            if test_loader:
                test_loaders[subject_name] = test_loader
                print(f"Successfully loaded test data for {subject_name}")
                
                # Debug: Check test data dimensions
                print(f"--- Debugging: Checking {subject_name} test data dimensions ---")
                try:
                    test_data_iter = iter(test_loader)
                    sample_batch, sample_labels = next(test_data_iter)
                    print(f"{subject_name} batch shape: {sample_batch.shape}")
                    print(f"{subject_name} labels shape: {sample_labels.shape}")
                except Exception as e:
                    print(f"Error checking {subject_name} test data dimensions: {e}")
                
                # Plot samples from each test subject
                # plot_data_samples(test_loader, f"Test Data - {subject_name}", le_cross, 
                #                 num_samples=1, num_channels_to_plot=5, device=device, 
                #                 DOWNSAMPLE_FACTOR=DOWNSAMPLE_FACTOR, NORMALIZE=NORMALIZE)
            else:
                print(f"Failed to create test DataLoader for {subject_name}")
        
        # Use first available test loader as validation during training
        val_loader_cross = None
        if test_loaders:
            val_loader_cross = list(test_loaders.values())[0]  # Use first test subject for validation during training
            print(f"Using {list(test_loaders.keys())[0]} data for validation during training")
        else:
            print("No test data loaders created successfully.")
    else:
        print("Skipping Cross-subject test data processing as training label encoder is missing.")

    # Determine number of classes from the label encoder
    if le_cross:
        num_classes = len(le_cross.classes_)
        print(f"Number of classes determined from label encoder: {num_classes}")
    else:
        print("Error: Label encoder not available. Cannot determine number of classes. Exiting.")
        return

    model = VAR_CNN(
        n_channels=N_CHANNELS,
        n_sources=N_SOURCES,
        n_classes=num_classes,
        filter_len=7,  # Default from VAR_CNN
        dropout=DROPOUT,
        # l1_penalty=3e-4  # Default from VAR_CNN
    ).to(device)


    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if train_loader_cross and val_loader_cross:
        model, history = train_model(
            model=model,
            train_loader=train_loader_cross, 
            val_loader=val_loader_cross,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            lr_patience=LR_PATIENCE,
            lr_factor=LR_FACTOR,
            weight_decay=WEIGHT_DECAY
        )
        
        print("\n--- Plotting Training History ---")
        plot_training_history(history)
    else:
        print("Skipping model training as training or validation data loader is missing.")
        return

    # Evaluate on all test subjects individually
    print("\n--- Evaluating Model on All Test Subjects ---")
    if test_loaders:
        model.eval()
        with torch.no_grad():
            for subject_name, test_loader in test_loaders.items():
                print(f"\nEvaluating on {subject_name}...")
                correct = 0
                total = 0
                all_predictions = []
                all_labels = []
                
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                accuracy = 100 * correct / total
                print(f'{subject_name} Accuracy: {accuracy:.2f}% ({correct}/{total})')
                
        
        # Calculate overall accuracy across all test subjects
        print("\n--- Overall Cross-Subject Performance ---")
        total_correct = 0
        total_samples = 0
        for subject_name, test_loader in test_loaders.items():
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
        
        overall_accuracy = 100 * total_correct / total_samples
        print(f'Overall Cross-Subject Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})')

    if test_loaders:
        print(f"Model evaluated on {len(test_loaders)} test subjects with overall accuracy: {overall_accuracy:.2f}%")

if __name__ == '__main__':
    main() 

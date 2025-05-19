import os
import h5py
import numpy as np
import pandas as pd # Keep for now, might be useful for other utilities
from sklearn.preprocessing import LabelEncoder # For converting string labels to integers
from sklearn.model_selection import train_test_split # Added for validation split
from collections import defaultdict
import scipy.signal # Added for decimation

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    def __init__(self, data_matrices, labels, transform=None, downsample_factor=1, normalize=False,
                 augment_noise_std=None, augment_scale_range=None):
        """
        Args:
            data_matrices (list of np.array): List of time series data matrices.
            labels (list of int): List of corresponding integer labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            downsample_factor (int, optional): Factor by which to downsample the time series data. Defaults to 1 (no downsampling).
            normalize (bool, optional): Whether to apply MinMax normalization per channel. Defaults to False.
            augment_noise_std (float, optional): If not None, adds Gaussian noise with this standard deviation.
            augment_scale_range (tuple, optional): If not None, a tuple (min_scale, max_scale) to randomly scale the data.
        """
        self.data_matrices = data_matrices
        self.labels = labels
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.augment_noise_std = augment_noise_std
        self.augment_scale_range = augment_scale_range

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data_matrices[idx].copy()
        label = self.labels[idx]

        # rn this is not used, we do downsampling in dataloder function
        if self.downsample_factor > 1:
            downsampled_channels = []
            for i in range(sample.shape[0]):
                channel_data = sample[i, :]
                downsampled_channel = scipy.signal.decimate(channel_data, q=self.downsample_factor, ftype='fir', axis=-1, zero_phase=True)
                downsampled_channels.append(downsampled_channel)
            sample = np.array(downsampled_channels)

        if self.normalize:
            for i in range(sample.shape[0]):
                channel_data = sample[i, :]
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                sample[i, :] = (channel_data - min_val) / (max_val - min_val + 1e-6)


        if self.augment_noise_std is not None:
            noise = np.random.normal(0, self.augment_noise_std, sample.shape)
            sample = sample + noise

        if self.augment_scale_range is not None:
            scale_factor = np.random.uniform(self.augment_scale_range[0], self.augment_scale_range[1])
            sample = sample * scale_factor

        if self.transform:
            sample = self.transform(sample)
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



def _filter_and_transform_labels(data_matrices, string_labels, label_encoder, dataset_name="data"):
    """
    Filters data based on known labels in a LabelEncoder and transforms labels to integers.
    Optionally fits the encoder or updates it with new labels if fit_encoder is True (for training data).
    """
    if not data_matrices or not string_labels:
        print(f"No data or labels provided to _filter_and_transform_labels for {dataset_name} set.")
        return [], None, label_encoder

    try:
        integer_labels = label_encoder.transform(string_labels)
        return data_matrices, integer_labels, label_encoder
    except ValueError as ve:
        print(f"Warning: Error transforming {dataset_name} labels: {ve}. Unknown labels found when 'fit_encoder' was False. Returning empty for this dataset.")
        return [], None, label_encoder



def prepare_pytorch_dataloader(data_dir, batch_size=32, shuffle_train=True, label_encoder=None, num_workers=0, 
                               downsample_factor=1, chunk_length=None, hop_length=None, normalize=False,
                               validation_split: float = None, random_state: int = None,
                               augment_noise_std_train=None, augment_scale_range_train=None):
    """
    Loads H5 data, applies downsampling, optionally splits sequences into chunks, 
    preprocesses labels, optionally splits data into training and validation sets, 
    and creates PyTorch DataLoader(s).

    Args:
        data_dir (str): Path to the directory containing H5 data.
        batch_size (int): Batch size for the DataLoader.
        shuffle_train (bool): Whether to shuffle the training data. Validation data is never shuffled.
        label_encoder (LabelEncoder, optional): Pre-fitted LabelEncoder. If None, a new one is fitted 
                                                (on training data if validation_split is used).
        num_workers (int): Number of worker processes for data loading.
        downsample_factor (int): Factor by which to downsample the time series data.
        chunk_length (int, optional): If provided, splits each time series (after downsampling) 
                                      into chunks of this length.
        hop_length (int, optional): If chunk_length is provided, this is the step/stride between chunks.
                                   Defaults to chunk_length // 2 if not specified and chunk_length is provided.
        normalize (bool, optional): Whether to apply MinMax normalization to the data. Defaults to False.
        validation_split (float, optional): Percentage of data to use for validation (e.g., 0.2 for 20%).
                                            If None, no validation set is created.
        random_state (int, optional): Random seed for train-validation split for reproducibility.
                                      Used only if validation_split is not None.
        augment_noise_std_train (float, optional): If not None, adds Gaussian noise with this standard deviation to training data.
        augment_scale_range_train (tuple, optional): If not None, a tuple (min_scale, max_scale) to randomly scale training data.

    Returns:
        tuple: (train_loader, val_loader, label_encoder)
               train_loader: DataLoader for the training set.
               val_loader: DataLoader for the validation set (None if validation_split is None).
               label_encoder: The LabelEncoder used.
    """
    print(f"Loading H5 data from {data_dir}...")
    raw_data_with_labels = load_h5_data(data_dir) 
    
    if not raw_data_with_labels:
        print(f"No data loaded from {data_dir}.")
        return None, None, label_encoder

    data_matrices, string_labels_original = zip(*raw_data_with_labels)
    
    processed_data_matrices = []
    if downsample_factor > 1:
        print(f"Applying decimation (factor {downsample_factor}) to {len(data_matrices)} loaded samples...")
        for idx, matrix in enumerate(data_matrices):
            try:
                downsampled_channels = []
                for i in range(matrix.shape[0]):
                    channel_data = matrix[i, :]
                    downsampled_channel = scipy.signal.decimate(channel_data, q=downsample_factor, ftype='fir', axis=-1, zero_phase=True)
                    downsampled_channels.append(downsampled_channel)
                downsampled_matrix = np.array(downsampled_channels)
            except ValueError as e:
                downsampled_matrix = matrix[:, ::downsample_factor]
        
        processed_data_matrices.append(downsampled_matrix)
    else:
        processed_data_matrices = list(data_matrices)

    final_data_matrices = []
    final_string_labels = []

    if chunk_length and chunk_length > 0:
        actual_hop_length = hop_length if hop_length is not None else chunk_length // 2
        if actual_hop_length <= 0: 
            print(f"Warning: hop_length ({actual_hop_length}) must be positive. Defaulting to 1.")
            actual_hop_length = 1

        print(f"Applying chunking with chunk_length={chunk_length}, hop_length={actual_hop_length}...")
        
        for i, matrix in enumerate(processed_data_matrices):
            original_label = string_labels_original[i]
            seq_len = matrix.shape[1]
            
            if seq_len == 0:
                print(f"  Warning: Sample {i} (original index) has zero length after downsampling. Skipping.")
                continue

            if seq_len < chunk_length:
                # Simplified padding assuming 2D data (channels, timepoints)
                pad_width = ((0,0), (0, chunk_length - seq_len)) 
                padded_matrix = np.pad(matrix, pad_width, 'constant', constant_values=0)
                final_data_matrices.append(padded_matrix)
                final_string_labels.append(original_label)
                continue 

            for start_idx in range(0, seq_len - chunk_length + 1, actual_hop_length):
                end_idx = start_idx + chunk_length
                chunk = matrix[:, start_idx:end_idx]
                final_data_matrices.append(chunk)
                final_string_labels.append(original_label)
        
        if not final_data_matrices and data_matrices:
            print("Warning: No data generated after chunking.")
            if processed_data_matrices and any(m.shape[1] > 0 for m in processed_data_matrices):
                 print("Returning None for DataLoader as chunking resulted in no usable data.")
                 return None, None, label_encoder
            elif not processed_data_matrices: 
                 return None, None, label_encoder

        if len(final_data_matrices) > len(processed_data_matrices) or (processed_data_matrices and not final_data_matrices and chunk_length):
             print(f"Chunking complete: {len(processed_data_matrices)} original samples expanded/processed into {len(final_data_matrices)} chunks.")
        elif chunk_length :
            print(f"Chunking processed {len(processed_data_matrices)} samples into {len(final_data_matrices)} segments (no net increase in sample count, possibly due to sequence lengths).")

    else: 
        final_data_matrices = processed_data_matrices
        final_string_labels = list(string_labels_original)


    train_data_matrices = final_data_matrices
    train_string_labels = final_string_labels
    val_data_matrices = None
    val_string_labels = None # Explicitly initialize
    integer_labels_train = None
    integer_labels_val = None

    if validation_split is not None and 0 < validation_split < 1:
        print(f"Splitting data into training and validation sets (validation_split={validation_split}, random_state={random_state})...")
        try:
            train_data_matrices, val_data_matrices, train_string_labels, val_string_labels = train_test_split(
                final_data_matrices, final_string_labels, 
                test_size=validation_split, 
                random_state=random_state, 
                stratify=final_string_labels 
            )
        except ValueError as e: # Catch error if stratification fails (e.g. too few samples for a class)
            print(f"Warning: Could not stratify data during train-validation split: {e}. Splitting without stratification.")
            train_data_matrices, val_data_matrices, train_string_labels, val_string_labels = train_test_split(
                final_data_matrices, final_string_labels,
                test_size=validation_split,
                random_state=random_state,
                stratify=None 
            )
        print(f"Training set size: {len(train_data_matrices)}, Validation set size: {len(val_data_matrices)}")
        
    # Process training labels
    if train_data_matrices and train_string_labels:
        train_data_matrices, integer_labels_train, label_encoder = _filter_and_transform_labels(
            train_data_matrices, train_string_labels, label_encoder, 
            dataset_name="training", fit_encoder=True # Always fit/update on training data
        )
        if not train_data_matrices or integer_labels_train is None: # integer_labels_train could be an empty array from helper
            print("Training data or labels became empty after encoding/filtering. Cannot create train DataLoader.")
            return None, None, label_encoder
    else:
        print("No training data/labels to process for encoder. Cannot create train DataLoader.")
        return None, None, label_encoder # label_encoder might be None here

    # Process validation labels (if validation set exists)
    if val_data_matrices and val_string_labels and label_encoder:
        val_data_matrices, integer_labels_val, _ = _filter_and_transform_labels(
            val_data_matrices, val_string_labels, label_encoder, 
            dataset_name="validation", fit_encoder=False # Never fit/update encoder on validation data
        )
        if not val_data_matrices or integer_labels_val is None: # integer_labels_val could be an empty array
            print("Validation data or labels became empty after encoding/filtering. Validation loader will be None.")
            val_data_matrices = None # Ensure it's None if effectively empty
            integer_labels_val = None
    elif validation_split is not None and not label_encoder:
        print("Warning: Validation split requested, but label encoder was not created (e.g. training data was empty). Cannot process validation labels.")
        val_data_matrices = None
        integer_labels_val = None
    
    if not train_data_matrices or not isinstance(integer_labels_train, (list, np.ndarray)) or len(integer_labels_train) == 0: # Final check
        print(f"No training data or labels remaining after all processing steps from {data_dir}. Cannot create train DataLoader.")
        return None, None, label_encoder

    train_dataset = TimeSeriesDataset(train_data_matrices, integer_labels_train, downsample_factor=1, normalize=normalize,
                                      augment_noise_std=augment_noise_std_train, augment_scale_range=augment_scale_range_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    print(f"Loaded {len(train_dataset)} samples/chunks into Training DataLoader from {data_dir}.")
    if label_encoder:
        print(f"LabelEncoder classes used for training: {list(label_encoder.classes_)}")


    val_loader = None
    if val_data_matrices and integer_labels_val is not None and len(integer_labels_val) > 0:
        # Augmentations are typically not applied to validation data
        val_dataset = TimeSeriesDataset(val_data_matrices, integer_labels_val, downsample_factor=1, normalize=normalize)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
        print(f"Loaded {len(val_dataset)} samples/chunks into Validation DataLoader from {data_dir}.")
    elif validation_split is not None:
        print("Validation set was requested but is empty or has no valid labels after processing. Validation DataLoader will be None.")

    print(f"Effective params: (original_downsample_factor={downsample_factor}, chunk_length={chunk_length}, hop_length={hop_length if chunk_length else None}, normalize={normalize}, validation_split={validation_split}).")
    return train_loader, val_loader, label_encoder


# h5 file parsing ------------------------------------------------------------

def parse_h5_filename(filename_with_dir):
    """
    Parses an H5 filename to extract label, base identifier, chunk index, and H5 dataset key.
    Assumes filename format like: [labelpart1]_[labelpart2]_[subjectID]_[chunkID].h5
    or [labelpart]_[subjectID]_[chunkID].h5
    Example: "task_motor_12345_1.h5"
        -> label: "task_motor"
        -> base_identifier: "task_motor_12345" (used for grouping chunks)
        -> chunk_index: 1
        -> h5_dataset_key: "task_motor_12345" (key for H5 file's dataset)
    Example: "rest_105923_1.h5"
        -> label: "rest"
        -> base_identifier: "rest_105923"
        -> chunk_index: 1
        -> h5_dataset_key: "rest_105923"
    """
    filename_without_dir = os.path.basename(filename_with_dir)
    filename_stem = filename_without_dir.replace('.h5', '') # e.g., "task_motor_12345_1"
    
    parts = filename_stem.split('_') # e.g., ['task', 'motor', '12345', '1']

    if len(parts) < 2: # Not enough parts for subjectID and chunkID
        print(f"Warning: Filename {filename_without_dir} does not fit expected format. Skipping.")
        return None

    try:
        chunk_index = int(parts[-1])
        base_parts = parts[:-1] # e.g., ['task', 'motor', '12345'] for grouping and H5 key
        
        h5_dataset_key = "_".join(base_parts) # This is the key for H5, e.g. "task_motor_12345"
        base_identifier_for_grouping = h5_dataset_key # Group by this

        label_candidate_parts = []
        for part in h5_dataset_key.split('_'):
            if part.isdigit():
                break
            label_candidate_parts.append(part)
        
        label = "_".join(label_candidate_parts)
        if not label: # Fallback if all parts were digits or label_candidate_parts is empty
            label = h5_dataset_key # Or a more specific part like h5_dataset_key.split('_')[0]

        return label, base_identifier_for_grouping, chunk_index, h5_dataset_key

    except ValueError: 
        print(f"Warning: Could not parse chunk index from {filename_without_dir}. Assuming not a chunked file or malformed.")
        h5_dataset_key = filename_stem
        base_identifier_for_grouping = filename_stem
        label = parts[0]
        return label, base_identifier_for_grouping, 0, h5_dataset_key

def load_h5_data(data_dir):
    """
    Loads H5 data files from a given directory.
    Handles chunked files (e.g., _1, _2) by concatenating them.
    Extracts data matrix and label for each full sample.
    """
    file_info_list = []
    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        return []

    for filename in os.listdir(data_dir):
        if filename.endswith(".h5"):
            file_path = os.path.join(data_dir, filename)
            parsed_info = parse_h5_filename(file_path)
            if parsed_info:
                label, base_id, chunk_idx, h5_key = parsed_info
                file_info_list.append({
                    "path": file_path,
                    "label": label,
                    "base_id": base_id,
                    "chunk_idx": chunk_idx,
                    "h5_key": h5_key 
                })

    grouped_files = defaultdict(list)
    for info in file_info_list:
        grouped_files[info["base_id"]].append(info)

    all_data = []
    for base_id, infos in grouped_files.items():
        infos.sort(key=lambda x: x["chunk_idx"]) 

        concatenated_matrix = None
        current_label = infos[0]["label"] 

        for info in infos:
            if info["label"] != current_label:
                print(f"Warning: Inconsistent labels for base_id {base_id}. Expected {current_label}, got {info['label']} in {info['path']}. Skipping this file.")
                continue
            
            try:
                with h5py.File(info["path"], 'r') as f:
                    h5_key_to_use = info["h5_key"]
                    if h5_key_to_use not in f:
                        print(f"Warning: Dataset key '{h5_key_to_use}' not found in {info['path']}. Available keys: {list(f.keys())}. Skipping chunk.")
                        continue
                    
                    matrix_chunk = f[h5_key_to_use][()]

                    if concatenated_matrix is None:
                        concatenated_matrix = matrix_chunk
                    else:
                        if concatenated_matrix.shape[0] == matrix_chunk.shape[0]: 
                            concatenated_matrix = np.concatenate((concatenated_matrix, matrix_chunk), axis=1)
                        else:
                            print(f"Error: Channel mismatch for {base_id} in {info['path']}. Expected {concatenated_matrix.shape[0]} channels, got {matrix_chunk.shape[0]}. Skipping chunk.")
                            continue
            except Exception as e:
                print(f"Error reading or processing {info['path']} (dataset: {info['h5_key']}): {e}")
                concatenated_matrix = None 
                break 
        
        if concatenated_matrix is not None:
            all_data.append((concatenated_matrix, current_label))
        elif infos: 
             print(f"Warning: Failed to load and concatenate all chunks for base_id {base_id}. Sample discarded.")

    return all_data



if __name__ == '__main__':
    # Example usage
    
    # For Intra-subject data
    intra_train_path = 'data/Intra/train'
    intra_test_path = 'data/Intra/test'
    
    print("--- Processing Intra-subject Training Data (with Validation Split) ---")
    train_loader_intra, val_loader_intra, le_intra = prepare_pytorch_dataloader(
        intra_train_path, batch_size=4, shuffle_train=True, num_workers=0, 
        downsample_factor=10, chunk_length=1500, hop_length=750, normalize=True,
        validation_split=0.2, random_state=42,
        augment_noise_std_train=0.01, augment_scale_range_train=(0.9, 1.1)
    )
    
    if train_loader_intra and le_intra: 
        print(f"Intra-subject training DataLoader created. Label mapping: {dict(zip(le_intra.classes_, le_intra.transform(le_intra.classes_)))}")
        if val_loader_intra:
            print("Intra-subject validation DataLoader created.")
        else:
            print("Intra-subject validation DataLoader was NOT created (check split or data).")
    elif not train_loader_intra:
        print("Failed to create Intra-subject training DataLoader.")

    print("\n--- Processing Intra-subject Test Data ---")
    if le_intra: 
        # Test data loader does not need validation split.
        test_loader_intra, _, _ = prepare_pytorch_dataloader( # Test loader does not produce a val_loader
            intra_test_path, batch_size=4, shuffle_train=False, 
            label_encoder=le_intra, num_workers=0, 
            downsample_factor=10, chunk_length=1500, hop_length=750, normalize=True,
            validation_split=None # Explicitly None for test set
        )
        if test_loader_intra:
            print("Intra-subject test DataLoader created.")
        else:
            print("Failed to create Intra-subject test DataLoader (possibly due to label issues or no data).")
    else:
        print("Skipping Intra-subject test data processing as training label encoder is missing.")


    # For Cross-subject data
    cross_train_path = 'data/Cross/train'
    cross_test1_path = 'data/Cross/test1' 
    
    print("\n--- Processing Cross-subject Training Data (with Validation Split) ---")
    train_loader_cross, val_loader_cross, le_cross = prepare_pytorch_dataloader(
        cross_train_path, batch_size=4, shuffle_train=True, num_workers=0,
        downsample_factor=10, chunk_length=1500, hop_length=750, normalize=True,
        validation_split=0.2, random_state=42,
        augment_noise_std_train=0.01, augment_scale_range_train=(0.9, 1.1)
    )
    if train_loader_cross and le_cross:
        print(f"Cross-subject training DataLoader created. Label mapping: {dict(zip(le_cross.classes_, le_cross.transform(le_cross.classes_)))}")
        if val_loader_cross:
            print("Cross-subject validation DataLoader created.")
        else:
            print("Cross-subject validation DataLoader was NOT created.")
    elif not train_loader_cross:
        print("Failed to create Cross-subject training DataLoader.")


    print("\n--- Processing Cross-subject Test Data (test1) ---")
    if le_cross: 
        test_loader_cross1, _, _ = prepare_pytorch_dataloader(
            cross_test1_path, batch_size=4, shuffle_train=False, 
            label_encoder=le_cross, num_workers=0, 
            downsample_factor=10, chunk_length=1500, hop_length=750, normalize=True,
            validation_split=None # Explicitly None for test set
        )
        if test_loader_cross1:
            print("Cross-subject test1 DataLoader created.")
        else:
            print("Failed to create Cross-subject test1 DataLoader (possibly due to label issues or no data).")

    else:
        print("Skipping Cross-subject test1 data processing as training label encoder is missing.")



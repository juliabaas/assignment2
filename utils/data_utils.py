import os
import h5py
import numpy as np
import time  # added for timing measurements
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler # for converting string labels to integers and standard scaling
from sklearn.exceptions import NotFittedError # added for robust scaler handling
import scipy.signal # added for decimation
# pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    pytorch dataset for time series data (with optional downsampling and normalization)
    """
    def __init__(self, data_matrices, labels, downsample_factor=1, normalize=False, scalers=None, 
                 sample_length=None, stride=None):
        """
        Args:
            data_matrices (list of np.array): List of time series data matrices.
            labels (list of int): List of corresponding integer labels.
            downsample_factor (int, optional): Factor by which to downsample the time series data. Defaults to 1 (no downsampling).
            normalize (bool, optional): whether to apply minmax normalization per channel. defaults to false.
            scalers (list of StandardScaler, optional): Pre-fitted scalers for each channel. Defaults to None.
            sample_length (int, optional): Length of each sample after splitting. If None, no splitting is performed.
            stride (int, optional): Stride for sliding window. If None, defaults to sample_length (no overlap).
        """
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.scalers = scalers
        self.sample_length = sample_length
        self.stride = stride if stride is not None else sample_length
        
        # process and split the data into samples
        #print(f"    Starting data processing and splitting...")
        start_time = time.time()
        self.samples, self.sample_labels = self._process_and_split_data(data_matrices, labels)
        #print(f"    Data processing and splitting completed in {time.time() - start_time:.2f} seconds")

    def _process_and_split_data(self, data_matrices, labels):
        """process data matrices (downsample and split into samples)."""
        all_samples, all_labels = [], []
        
        for matrix, label in zip(data_matrices, labels):
            # apply downsampling first
            downsampled_matrix = scipy.signal.decimate(matrix, self.downsample_factor, axis=1) if self.downsample_factor > 1 else matrix
            
            # split into samples if sample_length is specified
            if self.sample_length is not None:
                samples_from_matrix = self._split_matrix_into_samples(downsampled_matrix)
                all_samples.extend(samples_from_matrix)
                all_labels.extend([label] * len(samples_from_matrix))
            else:
                all_samples.append(downsampled_matrix)
                all_labels.append(label)
                
        return all_samples, all_labels
    
    def _split_matrix_into_samples(self, matrix):
        """split a single matrix into samples using sliding window."""
        samples = []
        _, time_length = matrix.shape
        
        if time_length < self.sample_length:
           # print(f"Warning: Matrix with time length {time_length} is shorter than sample_length {self.sample_length}. Skipping.")
            return []
        
        for start_idx in range(0, time_length - self.sample_length + 1, self.stride):
            samples.append(matrix[:, start_idx:start_idx + self.sample_length])
            
        return samples

    def __len__(self):
        return len(self.sample_labels)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        label = self.sample_labels[idx]

        if self.normalize:
            sample = self._apply_normalization(sample, idx)
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _apply_normalization(self, sample, idx):
        """apply normalization to a sample using the provided scalers."""
        if not self.scalers or sample.shape[1] == 0:
            if not self.scalers:
                print(f"    Warning: Normalization is True, but no scalers were provided to TimeSeriesDataset for sample {idx}. Skipping normalization for this sample.")
            return sample

        if len(self.scalers) != sample.shape[0]:
           # print(f"    Warning: Number of scalers ({len(self.scalers)}) does not match number of channels ({sample.shape[0]}) for sample {idx}. Skipping normalization for this sample.")
            return sample

        return self._normalize_channels(sample, idx)

    def _normalize_channels(self, sample, idx):
        """normalize each channel using its corresponding scaler."""
        for i in range(sample.shape[0]):
            channel_data_reshaped = sample[i, :].reshape(-1, 1)
            if channel_data_reshaped.size > 0:
                try:
                    sample[i, :] = self.scalers[i].transform(channel_data_reshaped).flatten()
                except NotFittedError:
                  #  print(f"    Error: Scaler for channel {i} was not fitted. Setting channel to zeros for sample {idx}.")
                    sample[i, :] = np.zeros_like(sample[i, :])
                except ValueError as e:
                    #print(f"    Warning: ValueError transforming channel {i} for sample {idx}: {e}. Setting channel to zeros.")
                    sample[i, :] = np.zeros_like(sample[i, :])
        return sample



def _prepare_label_encoder(string_labels, label_encoder):
    """prepare and fit label encoder if needed."""
    if label_encoder is None:
       # print("No LabelEncoder provided, fitting a new one.")
        label_encoder = LabelEncoder()
        try:
            label_encoder.fit(string_labels)
        except Exception as e:
        #    print(f"Error fitting LabelEncoder: {e}. Labels (first 10): {string_labels[:10]}.")
            return None
    return label_encoder


def _collect_channel_data_for_fitting(data_matrices, num_channels):
    """collect all channel data for scaler fitting."""
    all_channel_series_data = [[] for _ in range(num_channels)]
    
    for sample_matrix in data_matrices:
        if sample_matrix.ndim > 1 and sample_matrix.shape[0] == num_channels:
            for i in range(num_channels):
                all_channel_series_data[i].append(sample_matrix[i, :].flatten())
        else:
            print(f"Warning: Sample with shape {sample_matrix.shape} does not match expected num_channels {num_channels} or ndim. Skipping for scaler fitting.")
    
    return all_channel_series_data


def _fit_channel_scalers(all_channel_series_data, num_channels):
    """fit scalers for each channel."""
    fitted_scalers_list = [MinMaxScaler() for _ in range(num_channels)]
    
   # print(f"    Starting scaler fitting for {num_channels} channels...")
    start_time = time.time()
    
    for i in range(num_channels):
        if all_channel_series_data[i]:
            concatenated_channel_data = np.concatenate(all_channel_series_data[i]).reshape(-1, 1)
            if concatenated_channel_data.size > 0:
                try:
                    fitted_scalers_list[i].fit(concatenated_channel_data)
                except ValueError as e:
                    print(f"Warning: ValueError fitting scaler for channel {i}: {e}. This channel's scaler may not be effective.")
            else:
                print(f"Warning: No data to fit scaler for channel {i} (all samples might have been empty for this channel or mismatched).")
        else:
            print(f"Warning: No data collected for channel {i} to fit scaler.")
    
   # print(f"    Scaler fitting completed in {time.time() - start_time:.2f} seconds")
    return fitted_scalers_list


def _prepare_scalers_for_training(data_matrices, num_channels, normalize, channel_scalers):
    """prepare scalers for training data."""
    if not normalize:
        return None, channel_scalers
    
    if num_channels == 0:
       # print("Warning: Normalization is True, but number of channels is 0. Cannot fit/use scalers.")
        return None, channel_scalers
    
    if channel_scalers is None:
       # print(f"Fitting new StandardScaler for {num_channels} channels (Training mode)...")
        all_channel_series_data = _collect_channel_data_for_fitting(data_matrices, num_channels)
        fitted_scalers_list = _fit_channel_scalers(all_channel_series_data, num_channels)
        return fitted_scalers_list, fitted_scalers_list
    else:
      #  print("Using pre-provided scalers for training data...")
        return channel_scalers, channel_scalers


def _prepare_scalers_for_validation(num_channels, normalize, channel_scalers):
    """prepare scalers for validation/test data."""
    if not normalize:
        return None
    
    if channel_scalers is not None:
        if len(channel_scalers) == num_channels:
         #   print(f"Using {len(channel_scalers)} provided scalers for validation/test data.")
            return channel_scalers
        else:
          #  print(f"Warning: Number of provided scalers ({len(channel_scalers)}) does not match determined num_channels ({num_channels}) for validation/test. Normalization may be skipped by Dataset.")
            return None
    else:
      #  print("Warning: Normalization is True for validation/test data, but no scalers were provided. Data will NOT be normalized by TimeSeriesDataset.")
        return None


def _transform_labels(string_labels, label_encoder):
    """transform string labels to integers using the label encoder."""
    try:
        return label_encoder.transform(string_labels)
    except Exception as e:
      #  print(f"Error transforming labels: {e}. Check if all labels were seen during fitting if encoder was pre-provided.")
        return None


def _print_dataloader_summary(dataset, data_dir, label_encoder, normalize, current_scalers_for_dataset):
    """print summary information about the created dataloader."""
  #  print(f"Loaded {len(dataset)} samples into DataLoader from {data_dir}.")
    
    if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
        print(f"LabelEncoder classes: {list(label_encoder.classes_)}")
    else:
        print("LabelEncoder has no classes (e.g., fitting failed or no labels provided).")
    
    if normalize and current_scalers_for_dataset:
        print(f"Normalization active with {len(current_scalers_for_dataset)} scalers.")
    elif normalize and not current_scalers_for_dataset:
        print("Normalization was requested but scalers could not be prepared/provided; data may not be scaled.")


def prepare_pytorch_dataloader(data_dir, batch_size=32, shuffle=True, label_encoder=None,
                               channel_scalers=None, # new: to pass/receive pre-fitted scalers
                               num_workers=0, downsample_factor=1, normalize=False,
                               sample_length=None, stride=None):
  #  print(f"Loading H5 data from {data_dir}...")
    
    start_time = time.time()
    raw_data_with_labels = load_h5_data(data_dir) 
   # print(f"H5 data loading completed in {time.time() - start_time:.2f} seconds")
    
    if not raw_data_with_labels:
   #     print(f"No data loaded from {data_dir}.")
        return None, label_encoder, channel_scalers # return provided scalers

    data_matrices, string_labels = zip(*raw_data_with_labels)
    data_matrices, string_labels = list(data_matrices), list(string_labels)

    if not data_matrices or not string_labels:
     #   print("No data or labels after loading. Cannot create DataLoader.")
        return None, label_encoder, channel_scalers

    num_channels = 248

    # scaler preparation/handling
    current_scalers_for_dataset = None
    returned_scalers = channel_scalers # default to returning what was passed

    if label_encoder is None: # this implies it's a training run or the first run for a dataset block
     #   print("    Starting label encoder preparation...")
        start_time = time.time()
        label_encoder = _prepare_label_encoder(string_labels, label_encoder)
        if label_encoder is None:
            return None, label_encoder, returned_scalers
      #  print(f"    Label encoder preparation completed in {time.time() - start_time:.2f} seconds")

        # fit scalers only in this block (training / first run) if normalize is true and no scalers provided
        current_scalers_for_dataset, returned_scalers = _prepare_scalers_for_training(
            data_matrices, num_channels, normalize, channel_scalers)
    
    else: # label_encoder is provided (typically validation/test)
        current_scalers_for_dataset = _prepare_scalers_for_validation(
            num_channels, normalize, channel_scalers)

   # print("    Starting label transformation...")
    start_time = time.time()
    integer_labels = _transform_labels(string_labels, label_encoder)
   # print(f"    Label transformation completed in {time.time() - start_time:.2f} seconds")
    if integer_labels is None:
        return None, label_encoder, returned_scalers

  #  print("    Starting TimeSeriesDataset creation...")
    start_time = time.time()
    dataset = TimeSeriesDataset(data_matrices, integer_labels,
                                downsample_factor=downsample_factor,
                                normalize=normalize, # pass the overall normalize flag
                                scalers=current_scalers_for_dataset, # pass the determined scalers
                                sample_length=sample_length,
                                stride=stride)
   # print(f"    TimeSeriesDataset creation completed in {time.time() - start_time:.2f} seconds")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    _print_dataloader_summary(dataset, data_dir, label_encoder, normalize, current_scalers_for_dataset)

    return loader, label_encoder, returned_scalers


# h5 file parsing ------------------------------------------------------------

def _extract_label_from_filename(filename_with_dir):
    """
    parses an h5 filename to extract a label.
    the label is the part of the filename stem before the first numeric part (assumed to be a subject id).
    example: "task_motor_105923_1.h5" -> "task_motor"
             "rest_105923.h5" -> "rest"
    """
    filename_without_dir = os.path.basename(filename_with_dir)
    filename_stem = filename_without_dir.replace('.h5', '') # e.g., "task_motor_105923_1" or "rest_105923"
    
    parts = filename_stem.split('_')
    label_parts = []
    for part in parts:
        if part.isdigit(): # stop when the first numeric part (like subject id) is found
            break
        label_parts.append(part)
    
    if not label_parts: # fallback if no non-numeric parts found (e.g. "123_1.h5") or stem is empty
        if parts: # e.g. "123_1.h5" -> label "123" (first part)
            return parts[0]
        else: # e.g. empty stem from ".h5"
            return filename_stem # or a default error label
            
    return "_".join(label_parts)


def _load_single_h5_file(file_path, label):
    """load a single h5 file and return its data and label."""
    try:
        with h5py.File(file_path, 'r') as f:
            data_key = list(f.keys())[0]
            return (f[data_key][()], label)
    except Exception as e:
  #      print(f"Error reading or processing {file_path}: {e}")
        return None


def load_h5_data(data_dir):
    all_data = []
    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        return []

  #  print(f"    Starting file discovery in {data_dir}...")
    start_time = time.time()
    
    # group files by base name (everything before final _x suffix)
    file_groups = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".h5"):
            # extract base name by removing final _x suffix if it exists
            stem = filename.replace('.h5', '')
            parts = stem.split('_')
            base_name = '_'.join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() else stem
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(filename)
    
  #  print(f"    File discovery completed in {time.time() - start_time:.2f} seconds. Found {len(file_groups)} file groups.")
    
    # process each group by concatenating the parts
 #   print(f"    Starting file loading and concatenation...")
    start_time = time.time()
    
    for base_name, filenames in file_groups.items():
        # sort filenames to ensure proper order (_1, _2, _3, etc.)
        filenames.sort(key=lambda x: int(x.replace('.h5', '').split('_')[-1]) if x.replace('.h5', '').split('_')[-1].isdigit() else 0)
        
        matrices = []
        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    data_key = list(f.keys())[0]
                    matrices.append(f[data_key][()])
            except Exception as e:
      #          print(f"Error reading {file_path}: {e}")
                continue
        
        if matrices:
            # concatenate along time axis (assuming shape is [channels, time])
            concatenated_data = np.concatenate(matrices, axis=1)
            label = _extract_label_from_filename(filenames[0])  # use first file for label extraction
            all_data.append((concatenated_data, label))
        
 #   print(f"    File loading and concatenation completed in {time.time() - start_time:.2f} seconds")
    
    if not all_data:
        print(f"No H5 files successfully processed in {data_dir}.")
    else:
        print(f"    Successfully loaded {len(all_data)} data samples")
    return all_data



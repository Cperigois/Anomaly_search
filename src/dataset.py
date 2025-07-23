import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Custom time series dataset inheriting from torch.utils.data.Dataset

    Args:
        data: Input time series data (numpy array or torch tensor)
        dataset_size: Total number of samples in dataset
        signal_length: Length of each time series sample
        split_ratio: Ratio for train/validation split (0-1)
        dataset_name: Name identifier for the dataset
    """

    def __init__(self, data, dataset_size, signal_length, split_ratio=0.8, dataset_name="unnamed"):
        super().__init__()

        data = np.array(data)
        # Convert and validate input data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        elif not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be either numpy array or torch tensor")

        # Ensure correct shape [N_samples, 1, signal_length]
        if data.dim() == 2:
            data = data.unsqueeze(1)  # Add channel dimension
        elif data.dim() != 3:
            raise ValueError("Input data must have shape [N, C, L] or [N, L]")

        self.data = data
        self.dataset_size = dataset_size
        self.signal_length = signal_length
        self.split_ratio = split_ratio
        self.name = dataset_name

        # Validate dataset size matches actual data
        if len(data) != dataset_size:
            raise ValueError(f"Specified dataset_size ({dataset_size}) doesn't match actual data size ({len(data)})")

        # Create train/validation split
        self._create_splits()

    def _create_splits(self):
        """Internal method to create train/validation splits"""
        self.train_size = int(self.split_ratio * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size

        indices = torch.randperm(self.dataset_size)  # Random permutation for shuffling
        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:]

        # Store original full dataset
        self.full_dataset = torch.utils.data.TensorDataset(self.data)

    def __len__(self):
        """Returns total number of samples in dataset"""
        return self.dataset_size

    def __getitem__(self, idx):
        """Returns a single sample at given index"""
        return self.data[idx]

    def get_train_loader(self, batch_size=4, shuffle=True):
        """Create DataLoader for training data"""
        train_subset = torch.utils.data.Subset(self.full_dataset, self.train_indices)
        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def get_val_loader(self, batch_size=4, shuffle=False):
        """Create DataLoader for validation data"""
        val_subset = torch.utils.data.Subset(self.full_dataset, self.val_indices)
        return DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def get_loaders(self, batch_size=4, train_shuffle=True):
        """
        Convenience method to get both train and validation loaders

        Returns:
            Tuple of (train_loader, val_loader)
        """
        return (
            self.get_train_loader(batch_size, train_shuffle),
            self.get_val_loader(batch_size)
        )
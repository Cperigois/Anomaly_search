import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

def relative_mae_loss(output, target):
    # Ajustement si les tailles ne correspondent pas
    if output.shape[-1] > target.shape[-1]:
        output = output[..., :target.shape[-1]]
    elif output.shape[-1] < target.shape[-1]:
        target = target[..., :output.shape[-1]]
    return torch.mean(torch.abs(output - target) / (torch.abs(target) + 1e-8)) * 100

def mse_loss(output, target):
    if output.shape[-1] > target.shape[-1]:
        output = output[..., :target.shape[-1]]
    elif output.shape[-1] < target.shape[-1]:
        target = target[..., :output.shape[-1]]
    return torch.mean((output - target) ** 2)

import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

from IAmodel.loss import relative_mae_loss, mse_loss
from IAmodel.models import CNNAutoencoder

def train(epochs, encoded_size_ratio, dataset,
          learning_rate_params={'init_learning_rate': 0.1, 'learning_rate_factor': 0.5,'learning_rate_patience': 10, 'Evolution': True},
          model_name="model",batch_size = 32):
    """
    Train an autoencoder model with resume capability and automatic saving

    Args:
        epochs: Number of training epochs
        encoded_size_ratio: Ratio of latent space size to input size
        dataset: Dataset object containing train/val loaders
        dataset_size: Size of the dataset
        dataset_name: Name of the dataset for saving results
        learning_rate_params: Dictionary with LR scheduling parameters
        model_name: Name of the model for saving/loading
    """

    train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)
    # Setup device and paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(f"results/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "final_model.pth"

    # Initialize or load model
    encoded_size = int(dataset.signal_length * encoded_size_ratio)
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        model = CNNAutoencoder.load(model_path, device=device)
        # Verify latent size matches
        if model.encoded_size != encoded_size:
            raise ValueError(f"Loaded model has encoded_size={model.encoded_size} but requested {encoded_size}")
    else:
        print("Initializing new model")
        model = CNNAutoencoder(encoded_size).to(device)

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_params['init_learning_rate'])
    if learning_rate_params['Evolution'] == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=learning_rate_params['learning_rate_factor'],
            patience=learning_rate_params['learning_rate_patience']
        )

    # Prepare result directory
    result_dir = Path(
        f"results/{model_name}/{dataset.name}/"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training tracking
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    learning_rate = np.zeros(epochs)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = relative_mae_loss(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        train_loss[epoch] = epoch_train_loss / dataset.train_size

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = mse_loss(outputs, inputs)
                epoch_val_loss += loss.item() * inputs.size(0)
        val_loss[epoch] = epoch_val_loss / dataset.val_size

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if learning_rate_params.get('Evolution', True):
            scheduler.step(val_loss[epoch])
        learning_rate[epoch] = current_lr


        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss[epoch]:.4f}, "
              f"Val Loss: {val_loss[epoch]:.4f}, "
              f"LR: {current_lr:.2e}")

        # Save visualization and periodic checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Generate sample visualization
            visualize_sample(model, val_loader, device, result_dir, epoch)

    # Final model save
    model.save(model_dir, "final_model.pth")

    # Save training metrics
    save_training_metrics(train_loss, val_loss, learning_rate, result_dir)

    return result_dir


def visualize_sample(model, val_loader, device, result_dir, epoch):
    """Helper function to visualize a validation sample"""
    idx = random.randint(0, len(val_loader.dataset) - 1)
    val_sample = val_loader.dataset[idx][0].unsqueeze(0).to(device)

    with torch.no_grad():
        output_sample = model(val_sample).cpu().squeeze().numpy()
        input_sample = val_sample.cpu().squeeze().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(input_sample, label="Original")
    plt.plot(output_sample, label="Reconstruction", linestyle='--')
    plt.title(f"Epoch {epoch + 1:02d} - Validation Sample")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(result_dir / f"epoch_{epoch + 1}.png")
    plt.close()


def save_training_metrics(train_loss, val_loss, learning_rate, result_dir):
    """Helper function to save training metrics"""
    df = pd.DataFrame({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate
    })
    df.to_csv(result_dir / "training_metrics.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss", linestyle='--')
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_dir / "loss_curves.png")
    plt.close()

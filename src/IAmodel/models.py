import torch
import torch.nn as nn
import os
from pathlib import Path


class CNNEncoder(nn.Module):
    def __init__(self, encoded_size):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, 250]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 125]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, ~63]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 63, encoded_size)
        )

    def forward(self, x):
        return self.encoder(x)


class CNNDecoder(nn.Module):
    def __init__(self, encoded_size):
        super(CNNDecoder, self).__init__()
        self.decoder_input = nn.Linear(encoded_size, 64 * 63)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 63)),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 32, ~125]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 16, ~250]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),   # [B, 1, ~500]
        )

    def forward(self, x):
        x = self.decoder_input(x)
        x = self.decoder(x)
        return x


class CNNAutoencoder(nn.Module):
    def __init__(self, encoded_size):
        super(CNNAutoencoder, self).__init__()
        self.encoder = CNNEncoder(encoded_size)
        self.decoder = CNNDecoder(encoded_size)
        self.encoded_size = encoded_size

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def save(self, directory="models", filename="cnn_autoencoder.pth"):
        """
        Save the model to specified directory

        Args:
            directory (str): Directory to save the model
            filename (str): Name of the model file

        Returns:
            str: Path where the model was saved
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Prepare save path
        save_path = os.path.join(directory, filename)
        self.get_config()

        # Save model state dict and metadata
        torch.save({
            'state_dict': self.state_dict(),
            'encoded_size': self.encoded_size,
            'encoder_config': self.encoder.config if hasattr(self.encoder, 'config') else None,
            'decoder_config': self.decoder.config if hasattr(self.decoder, 'config') else None
        }, save_path)

        return save_path

    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Load model from file

        Args:
            filepath (str): Path to the saved model
            device (str): Device to load the model onto ('cpu' or 'cuda')

        Returns:
            CNNAutoencoder: Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If there's an error during loading
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=device)

            # Create model instance
            model = cls(checkpoint['encoded_size'])

            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])

            # Transfer to appropriate device
            model = model.to(device)

            # Set to evaluation mode
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {str(e)}")

    def get_config(self):
        """
        Get model configuration

        Returns:
            dict: Model configuration dictionary
        """
        return {
            'encoded_size': self.encoded_size,
            'encoder_config': self.encoder.config if hasattr(self.encoder, 'config') else None,
            'decoder_config': self.decoder.config if hasattr(self.decoder, 'config') else None
        }
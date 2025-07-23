import pandas as pd
from src.signal.signal import Signal
import torch
import numpy as np
from pathlib import Path
from IAmodel.loss import relative_mae_loss, mse_loss
from IAmodel.models import CNNAutoencoder


def test_model_individual_loss(test_name, model_path, signal_length=500, test_size=10000, num_examples = 10):
    """
    Teste le modèle sur 10 000 signaux et sauvegarde la perte MSE pour CHAQUE signal individuellement.
    Args:
        model_path: Chemin vers le modèle (.pth)
        signal_length: Longueur des signaux (doit matcher le modèle)
        test_size: Nombre de signaux à tester (ici 10 000)
    """
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder.load(model_path, device=device)
    model.eval()

    # 2. Génération des 10 000 signaux (un par un)
    all_signals = []
    for _ in range(test_size):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.025+0.2 * np.random.random(), 2 + 4*np.random.random(), np.pi * np.random.random())
        signal.insert_gaussian_noise(0, 0.25)
        signal.insert_drift(0.25, 'min')
        all_signals.append(signal.values)  # shape (500,)

    # 3. Conversion en tenseur PyTorch
    test_data = torch.tensor(np.array(all_signals), dtype=torch.float32).unsqueeze(1)  # shape [10000, 1, 500]

    # 4. Calcul des pertes individuelles
    individual_losses = np.zeros(test_size)
    example_data = {i: {'time_step': [], 'original': [], 'reconstructed': []}
                    for i in range(num_examples)}

    with torch.no_grad():
        for i in range(test_size):
            # Traitement signal par signal
            input_signal = test_data[i].unsqueeze(0).to(device)  # shape [1, 1, 500]
            output = model(input_signal)

            # MSE pour ce signal uniquement
            loss = mse_loss(output = output, target= input_signal)
            individual_losses[i] = loss

            # Stockage des exemples
            if i < num_examples:
                # Extraction et conversion des signaux
                original_signal = input_signal.squeeze().cpu().numpy()[:signal_length]  # Troncature à 500 points
                reconstructed_signal = output.squeeze().cpu().numpy()[:signal_length]  # Troncature à 500 points

                # Stockage avec vérification de taille
                if len(original_signal) != signal_length or len(reconstructed_signal) != signal_length:
                    print(f"Avertissement: Signal {i} tronqué à {signal_length} points")

                example_data[i]['time_step'] = np.arange(signal_length)
                example_data[i]['original'] = original_signal
                example_data[i]['reconstructed'] = reconstructed_signal

    df_losses = pd.DataFrame({'signal' : np.arange(test_size), 'loss' : np.array(individual_losses)})
    example_dfs = {f"signal_{i}": pd.DataFrame(data) for i, data in example_data.items()}

    # 5. Sauvegarde des résultats
    results_dir = Path(f"tests/{test_name}")
    results_dir.mkdir(parents = True, exist_ok=True)

    df_losses.to_csv(f'{results_dir}/losses.csv')

    for signal_id, df in example_dfs.items():
        df.to_csv(results_dir / f"{signal_id}.csv", index=False)




# Exemple d'utilisation
if __name__ == "__main__":
    losses = test_model_individual_loss(
        model_path="results/sinusoïd recognition last run/final_model.pth",
        signal_length=500,
        test_size=1000, test_name = 'sinusoid_recognition drift 8'
    )

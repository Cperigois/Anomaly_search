import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration des drifts à afficher
selected_drifts = [1, 0.05, 0.025, 0.005]
drift_indices = [0, 3, 6, 5]  # Indices correspondants dans la liste originale

# Création de la figure avec 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Signal reconstruction at different drift rates', fontsize=16, y=1.02)

# Parcours des 4 drifts sélectionnés
for i, (drift_idx, ax) in enumerate(zip(drift_indices, axs.flat)):
    drift_rate = selected_drifts[i]
    path = f"tests/sinusoid_recognition drift {drift_idx + 1}"
    filename = "signal_0"

    df = pd.read_csv(f"{path}/{filename}.csv", index_col=None)

    # Plot des signaux
    ax.plot(df['time_step'], df['original'], label='Original')
    ax.plot(df['time_step'], df['reconstructed'], label='Reconstructed')

    # Configuration des subplots
    ax.set_xlim(0, 500)
    ax.set_title(f"Drift rate = {drift_rate} min$^{-1}$", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Ajout de la légende seulement sur le premier subplot
    if i == 1:  # Subplot en haut à droite
        ax.legend(fontsize=12, loc='upper right')

# Ajustement de l'espacement
plt.tight_layout()
plt.savefig("combined_drift_comparison.png", dpi=300, bbox_inches='tight')
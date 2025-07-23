import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

steps = {'pure_fixed_sinusoïd' : 'step 1',
    'small_amplitude_variation' : 'step 2',
    'small_amplitude_and_phase_variation' : 'step 3',
    'small_amplitude_phase_frequency_variation' : 'step 4',
    'small_variation_and_small_gaussian_noise' : 'step 5',
    'variation_and_small_gaussian_noise' : 'step 6',
    'variation_and_gaussian_noise' : 'step 7'
         }

steps_epoch = [10, 40, 70, 170, 270, 520, 920]

val_loss = np.array([]) #step by step
learning_rates = np.array([])

for step in steps.keys():
    df = pd.read_csv(f"results/sinusoïd recognition last run/{step}/training_metrics.csv", index_col=None)
    val_loss = np.concatenate((val_loss, df['val_loss']))
    learning_rates = np.concatenate((learning_rates, df['learning_rate']))


# Clamp les valeurs très petites
min_display_value = 0.0008
val_loss_display = np.clip(val_loss, min_display_value, None)
val_loss_mask = val_loss < min_display_value

# Création du graphique
fig, ax1 = plt.subplots(figsize=(8, 6))

# Couleurs harmonisées
loss_color = "tab:blue"
lr_color = "crimson"

# Plot de la loss avec couleur harmonisée
ax1.plot(np.arange(len(val_loss)), val_loss_display, label="Validation Loss", color=loss_color)
ax1.scatter(np.arange(len(val_loss))[val_loss_mask], val_loss_display[val_loss_mask],
            color='red', label='Values <0.001', zorder=5, s=8)

# Configuration de l'axe gauche (loss)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Validation Loss (MSE)", fontsize=12, color=loss_color)
ax1.tick_params(axis='x', which='major', labelsize=14, colors='black')
ax1.tick_params(axis='y', which='major', labelsize=14, colors=loss_color)
ax1.spines['left'].set_color(loss_color)  # Axe gauche en bleu

# Zone grise = bruit résiduel
#ax1.axhspan(0, 0.0625, color='gray', alpha=0.2, label='Residual final noise level')

# Zone grise = bruit résiduel variable par phase
#ax1.axhspan(0, 0, xmin=0/920, xmax=70/920, color='gray', alpha=0.2)  # Phase 1-3 (0-69)
ax1.axhspan(0, 0.0025, xmin=70/920, xmax=520/920, color='gray', alpha=0.2)  # Phase 4-6 (70-519)
ax1.axhspan(0, 0.0625, xmin=520/920, xmax=1, color='gray', alpha=0.2, label='Residual noise level')  # Phase 7 (520-919)


# Axe secondaire pour le learning rate (couleur harmonisée)
ax2 = ax1.twinx()
ax2.plot(np.arange(len(val_loss)), learning_rates, color=lr_color, label="Learning Rate")
ax2.set_ylabel("Learning Rate", fontsize=12, color=lr_color)  # Label en vert
ax2.tick_params(axis='y', which='both',  labelsize=14,colors=lr_color)  # Ticks en vert
ax2.spines['right'].set_color(lr_color)  # Axe droit en vert
ax2.set_yscale("log")

# Labels pour les étapes
y_max = np.max(val_loss_display) * 1.2
for i, epoch in enumerate(steps_epoch):
    ax1.axvline(x=epoch, color='gray', linestyle='--', linewidth=1)
    ax1.text(epoch - 2, 1e-3, f"step {i+1}",
             rotation=90, va='bottom', ha='right', fontsize=10, color='black')

# Configuration commune
ax1.set_yscale("log")
ax1.set_title("Training Progress with Curriculum Steps (Log Scale)", fontsize=16)
ax1.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
ax1.set_xlim(0, 920)
ax1.set_ylim(0.0007, 20)

# Combiner les légendes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig("sinusoïd_recognition.png", dpi=300, bbox_inches='tight')

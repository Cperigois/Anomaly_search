import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap


# Configuration des chemins et labels
data_paths = [
    "tests/sinusoid_recognition last run/losses.csv",
    "tests/sinusoid_recognition drift 1/losses.csv",
    "tests/sinusoid_recognition drift 2/losses.csv",
    "tests/sinusoid_recognition drift 3/losses.csv",
    "tests/sinusoid_recognition drift 4/losses.csv",
    "tests/sinusoid_recognition drift 5/losses.csv",
    "tests/sinusoid_recognition drift 6/losses.csv"
]

labels = [
    "Evaluation",
    "1 min⁻¹",
    "0.5 min⁻¹",
    "0.1 min⁻¹",
    "0.05 min⁻¹",
    "0.01 min⁻¹",
    "0.005 min⁻¹"
]

# Couleurs personnalisées
COLORS = {
    'green': '#8ac929',
    'orange': '#ffca3a',
    'red': '#ff595e',
    'eval': '#000000'
}

# Palette bleue avec 6 nuances exactement (du clair au foncé)
blue_palette = [
    '#6baed6',  # drift 1 min⁻¹ (le plus clair)
    '#4292c6',  # drift 0.5 min⁻¹
    '#2171b5',  # drift 0.1 min⁻¹
    '#08519c',  # drift 0.05 min⁻¹
    '#08306b',  # drift 0.01 min⁻¹
    '#00204a'   # drift 0.005 min⁻¹ (le plus foncé)
]

# Chargement des données
data = []
for path in data_paths:
    df = pd.read_csv(path)
    data.append(df['loss'].values)

# Calcul des valeurs importantes
max_eval = np.max(data[0])
zone_green_end = np.percentile(data[0], 95)  # 95ème percentile
zone_orange_end = max_eval
all_values = np.concatenate(data)
min_val = np.min(all_values[all_values > 0])  # Éviter 0 pour le log
max_val = np.max(all_values)
print(max_val)

# Création des bins logarithmiques
num_bins = 100
log_bins = np.logspace(np.log10(min_val), np.log10(30), num_bins + 1)

# Création du graphique
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Palette de couleurs
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)-1))  # Du clair au foncé

# Tracé des histogrammes en densité avec les mêmes bins
# D'abord les drifts (du plus clair au plus foncé)
for i in range(1, len(data)):
    hist, _ = np.histogram(data[i], bins=log_bins)
    plt.step(log_bins[:-1], hist, where='post',
             color=blue_palette[i-1], label=labels[i], linewidth=3)

# Puis l'évaluation (noire en pointillé épais)
hist_eval, _ = np.histogram(data[0], bins=log_bins, weights= 0.1*np.ones(10000) )
plt.step(log_bins[:-1], hist_eval, where='post',
         color=COLORS['eval'], label=labels[0], linewidth=4, linestyle='--')

# Zones colorées verticales (sans légende)
plt.axvspan(min_val, zone_green_end, color=COLORS['green'], alpha=0.4)
plt.axvspan(zone_green_end, zone_orange_end, color=COLORS['orange'], alpha=0.4)
plt.axvspan(zone_orange_end, 30, color=COLORS['red'], alpha=0.4)

# Lignes de délimitation des zones
plt.axvline(zone_green_end, color=COLORS['orange'], linestyle=':', alpha=0.7, linewidth=1)
plt.axvline(zone_orange_end, color=COLORS['orange'], linestyle=':', alpha=0.7, linewidth=1)

# Configuration des axes en échelle log
plt.xscale('log')
plt.yscale('log')


# Format des ticks
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# Labels et titres
plt.xlabel('Loss Value (log scale)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Loss distributions with drift anomalies', fontsize=14, fontweight='bold')

# Grille
ax.grid(True, which='both', linestyle='--', alpha=0.3)

# Légende
plt.legend(fontsize=14, framealpha=1, edgecolor='black')

# Ajustement des limites
plt.xlim(min_val, 20)
plt.ylim(bottom=1)  # Ajuster selon vos données
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)


plt.tight_layout()
plt.savefig('loss_distributions_with_zones.png', dpi=300)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# Configuration des chemins et données
drift_rates = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.025, 0.25]  # en min⁻¹
data_paths = [f"tests/sinusoid_recognition drift {i+1}/losses.csv" for i in range(8)]
eval_path = "tests/sinusoid_recognition last run/losses.csv"

# Chargement des données
def load_data(path):
    df = pd.read_csv(path)
    return df['loss'].values[df['loss'].values > 0]  # Exclure les valeurs ≤0

eval_data = load_data(eval_path)
drift_data = [load_data(path) for path in data_paths]

# Calcul des statistiques
eval_median = np.median(eval_data)
eval_max = np.max(eval_data)

eval_lower = np.percentile(eval_data, 2.5) # 2.5% percentile
eval_upper = np.percentile(eval_data, 97.5)  # 97.5% percentile

print(eval_median)
print(eval_lower - eval_median)
print(eval_upper - eval_median)


drift_medians = [np.median(data) for data in drift_data]
drift_lower = [np.percentile(data, 2.5) for data in drift_data]  # 2.5% percentile
drift_upper = [np.percentile(data, 97.5) for data in drift_data]  # 97.5% percentile



# Zones de seuil (verticales maintenant)
zone_green_end = np.percentile(eval_data, 95)  # 95ème percentile
zone_orange_end = eval_max
green_zone = (0, zone_green_end)
orange_zone = (zone_green_end, eval_max)
red_zone = (eval_max, np.max([np.max(data) for data in drift_data]))

# Création du graphique
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Lignes de séparation des zones
plt.axhline(green_zone[1], color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.axhline(orange_zone[1], color='gray', linestyle=':', alpha=0.7, linewidth=1)

# Plot des données de drift
plt.errorbar(drift_rates, drift_medians,
             yerr=[np.array(drift_medians)-np.array(drift_lower),
                   np.array(drift_upper)-np.array(drift_medians)],
             fmt='o', markersize=10, capsize=5, capthick=2,
             color='#08306b', label='Drift (median ± 95%)')


# Plot de la référence d'évaluation
plt.axhline(y=eval_median, color='darkgreen', linestyle='--', linewidth=3,
            label=f'Evaluation (median: {eval_median:.2e})')

# Mise en forme des axes
plt.xscale('log')
plt.yscale('log')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks(drift_rates)
ax.set_xticklabels([f"{rate}\n{rate*500/60/0.25:.2f}" for rate in drift_rates])

# Labels et titres
plt.xlabel('Drift rate (min⁻¹)\nNoise ratio', fontsize=14, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=14, fontweight='bold')
plt.title('Drift impact on the loss', fontsize=16, fontweight='bold')

# Légende et grille
plt.legend(fontsize=14, framealpha=1, loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
# Zones colorées horizontales
plt.axhspan(green_zone[0], green_zone[1], color='#8ac929', alpha=0.35, label='Zone verte')
plt.axhspan(orange_zone[0], orange_zone[1], color='#ffca3a', alpha=0.35, label='Zone orange')
plt.axhspan(red_zone[0], red_zone[1], color='#ff595e', alpha=0.35, label='Zone rouge')
# Ajustement des ticks
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14
               )

# Ajustement des limites
plt.xlim(min(drift_rates)*0.8, max(drift_rates)*1.2)
plt.ylim(min(drift_lower)*0.9, max(drift_upper)*1.1)

# Ajout des mentions verticales à droite
ax = plt.gca()
x_pos = ax.get_xlim()[1] * 1.2  # Position juste à droite de l'axe

# Positionnement vertical des mentions (milieu de chaque zone)
plt.text(x_pos, 0.08, 'OK',
         color='#8ac929', alpha=1, fontsize=14, fontweight='bold',
         ha='left', va='center', rotation=90)
plt.text(x_pos, 0.23, 'WARNING',
         color='#ffca3a', alpha=1, fontsize=14, fontweight='bold',
         ha='left', va='center', rotation=90)
plt.text(x_pos, 1.2, 'ANOMALY',
         color='#ff595e', alpha=1, fontsize=14, fontweight='bold',
         ha='left', va='center', rotation=90)

# Ajustement des limites pour faire de la place aux mentions
plt.xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.15)

plt.tight_layout()
plt.savefig('drift_impact_analysis.png', dpi=300, bbox_inches='tight')
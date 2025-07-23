import numpy as np
import scipy.stats as stats
from pathlib import Path


def calculate_and_save_stats(data_series: list[float], filename: str) -> None:
    data_array = np.array(data_series, dtype=np.float64)

    # ✅ Nettoyage : enlever NaN et infinities
    data_array = data_array[np.isfinite(data_array)]

    n = len(data_array)

    if n == 0:
        print("⚠️ Aucune donnée valide fournie pour l'analyse statistique.")
        return

    mean_value = np.mean(data_array)
    std_dev = np.std(data_array, ddof=1) if n > 1 else 0.0
    median_value = np.median(data_array)

    # ✅ Calcul de l'intervalle avec sécurité
    if n > 1 and std_dev > 0:
        sem = stats.sem(data_array)
        if n < 30:
            interval = stats.t.interval(0.95, df=n - 1, loc=mean_value, scale=sem)
        else:
            interval = stats.norm.interval(0.95, loc=mean_value, scale=sem)
    else:
        interval = (mean_value, mean_value)

    # Affichage
    print("\nStatistical Analysis Results:")
    print(f"- Mean: {mean_value:.6f}")
    print(f"- Standard deviation: {std_dev:.6f}")
    print(f"- 95% Confidence Interval: ({interval[0]:.6f}, {interval[1]:.6f})")
    print(f"- Median: {median_value:.6f}")

    stats_content = f"""Statistical Analysis Results:
Mean: {mean_value:.6f}
Standard deviation: {std_dev:.6f}
95% Confidence Interval: ({interval[0]:.6f}, {interval[1]:.6f})
Median: {median_value:.6f}
Sample size: {n}
"""

    output_file = Path(f"{filename}.stats")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(stats_content)
        print(f"\nStatistics saved successfully to {output_file}")
    except IOError as e:
        print(f"\nError saving file: {e}")
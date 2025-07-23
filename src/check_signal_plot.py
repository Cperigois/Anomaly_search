import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

drift = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.025, 0.25]

for i in range(8) :

    path = f"tests/sinusoid_recognition drift {i+1}"
    filename = "signal_0"

    df = pd.read_csv(f"{path}/{filename}.csv", index_col=None)

    plt.plot(df['time_step'], df['original'], label = "Original")
    plt.plot(df['time_step'],  df['reconstructed'], label = "Reconstructed")
    plt.legend(fontsize = 12)
    plt.title(f"Drift rate {drift[i]} min^-1")

    plt.tight_layout()
    plt.savefig(f"{path}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
"""Plot g(r) of the working state point and of the NIST-condition run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    runs = {
        "ρ*=0.8442, T*=0.722, rc=2.5σ (N=864)": "results/raw/working/s1_rdf.csv",
        "ρ*=0.84, T*≈0.85, rc=3σ (N=500)": "results/raw/nist/rho0.84_s1_rdf.csv",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, path in runs.items():
        data = np.genfromtxt(path, delimiter=",", names=True)
        ax.plot(data["r"], data["g"], label=label)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("r / σ")
    ax.set_ylabel("g(r)")
    ax.legend(fontsize=8)
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/rdf.png", dpi=200)


if __name__ == "__main__":
    main()

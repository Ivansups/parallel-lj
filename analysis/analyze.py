"""Post-processing of md_serial output: D from MSD, block errors, LRC-corrected averages."""
from __future__ import annotations

import sys

import numpy as np


def block_mean_error(x: np.ndarray, nblocks: int = 10) -> tuple[float, float]:
    """Mean and standard error from block averaging."""
    size = len(x) // nblocks
    blocks = np.asarray(x[: size * nblocks], dtype=float).reshape(nblocks, size).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(nblocks))


def fit_diffusion(t: np.ndarray, msd: np.ndarray, t_min: float, t_max: float) -> float:
    """Self-diffusion D = slope / 6 of MSD(t) fitted on [t_min, t_max]."""
    mask = (t >= t_min) & (t <= t_max)
    slope = np.polyfit(t[mask], msd[mask], 1)[0]
    return float(slope / 6.0)


def interpolate_to_temperature(
    temps: np.ndarray, values: np.ndarray, target: float
) -> tuple[float, float]:
    """Linear fit of values(T) across runs, evaluated at target T with its standard error."""
    coef, cov = np.polyfit(temps, values, 1, cov=True)
    x = np.array([target, 1.0])
    return float(np.polyval(coef, target)), float(np.sqrt(x @ cov @ x))


def lrc_energy_per_particle(rho: float, rc: float) -> float:
    """Standard long-range correction to U/N (g(r) = 1 beyond rc)."""
    return float(8.0 / 3.0 * np.pi * rho * (rc**-9 / 3.0 - rc**-3))


def lrc_pressure(rho: float, rc: float) -> float:
    """Standard long-range correction to the pressure."""
    return float(16.0 / 3.0 * np.pi * rho**2 * (2.0 * rc**-9 / 3.0 - rc**-3))


def summarize_thermo(
    path: str, n: int, rho: float, rc: float, nblocks: int = 10
) -> dict[str, tuple[float, float]]:
    """NVE averages (value, error): T, P with LRC, U per particle with LRC."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    nve = data[data["phase"] == "nve"]
    temperature = block_mean_error(nve["temperature"], nblocks)
    pressure = block_mean_error(nve["pressure"] + lrc_pressure(rho, rc), nblocks)
    energy = block_mean_error(
        nve["potential_plain"] / n + lrc_energy_per_particle(rho, rc), nblocks
    )
    return {"T": temperature, "P": pressure, "U": energy}


def main(prefix: str, n: int, rho: float, rc: float) -> None:
    """Print the NVE averages and D for one run (prefix as given to md_serial)."""
    summary = summarize_thermo(f"{prefix}_thermo.csv", n, rho, rc)
    msd = np.genfromtxt(f"{prefix}_msd.csv", delimiter=",", names=True)
    t_end = float(msd["time"][-1])
    diffusion = fit_diffusion(msd["time"], msd["msd"], 0.2 * t_end, t_end)
    for key, (value, err) in summary.items():
        print(f"{key} = {value:.4f} +/- {err:.4f}")
    print(f"D = {diffusion:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

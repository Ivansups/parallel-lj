"""Compare md_serial runs with the NIST NVE MD benchmark (T* ~ 0.85, N = 500, rc = 3).

Source: https://mmlapps.nist.gov/srs/LJ_PURE/md.htm
Runs are brought to the NIST temperature by a linear fit across seeds.
"""
from __future__ import annotations

import glob

import numpy as np

import analyze

N, RC = 500, 3.0
# rho: (T, U, p, D) from the NIST table
NIST = {
    "0.776": (0.851, -5.517, 0.030, 0.061),
    "0.82": (0.852, -5.803, 0.573, 0.048),
    "0.84": (0.851, -5.909, 0.910, 0.042),
}


def collect(rho: str) -> np.ndarray:
    """Per seed: T, P, U, D for one density."""
    rows = []
    for thermo in sorted(glob.glob(f"results/raw/nist/rho{rho}_s*_thermo.csv")):
        prefix = thermo.removesuffix("_thermo.csv")
        summary = analyze.summarize_thermo(thermo, N, float(rho), RC)
        msd = np.genfromtxt(f"{prefix}_msd.csv", delimiter=",", names=True)
        t_end = float(msd["time"][-1])
        d = analyze.fit_diffusion(msd["time"], msd["msd"], 0.2 * t_end, t_end)
        rows.append((summary["T"][0], summary["P"][0], summary["U"][0], d))
    return np.array(rows)


def main() -> None:
    print("| rho | quantity | this work | NIST | difference |")
    print("|---|---|---|---|---|")
    for rho, (t0, u_ref, p_ref, d_ref) in NIST.items():
        runs = collect(rho)
        for name, col, ref in (("U", 2, u_ref), ("P", 1, p_ref), ("D", 3, d_ref)):
            value, err = analyze.interpolate_to_temperature(runs[:, 0], runs[:, col], t0)
            print(f"| {rho} | {name} | {value:.4f} +/- {err:.4f} | {ref} | {value - ref:+.4f} |")


if __name__ == "__main__":
    main()

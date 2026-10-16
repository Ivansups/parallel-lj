#!/usr/bin/env bash
# NIST-condition NVE runs (N=500, rc=3, T=0.85): 8 seeds per density, 4 in parallel.
# Run inside the container: docker run --rm -v "$PWD":/work parallel-lj scripts/run_nist.sh
set -euo pipefail
out=results/raw/nist
mkdir -p "$out"
for rho in 0.776 0.82 0.84; do
  for batch in "1 2 3 4" "5 6 7 8"; do
    for s in $batch; do
      ./build/md_serial 5 10000 20000 "$s" "$rho" 0.85 3.0 "$out/rho${rho}_s$s" \
        > "$out/rho${rho}_s${s}_thermo.csv" &
    done
    wait
  done
done

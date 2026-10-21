#!/usr/bin/env bash
# Working state point (rho=0.8442, T=0.722, rc=2.5, N=864): 4 seeds in parallel.
# Run inside the container: docker run --rm -v "$PWD":/work parallel-lj scripts/run_working.sh
set -euo pipefail
out=results/raw/working
mkdir -p "$out"
for s in 1 2 3 4; do
  ./build/md_serial 6 10000 20000 "$s" 0.8442 0.722 2.5 "$out/s$s" > "$out/s${s}_thermo.csv" &
done
wait

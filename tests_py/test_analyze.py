import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import analyze  # noqa: E402


def test_block_error_of_constant_series_is_zero() -> None:
    mean, err = analyze.block_mean_error(np.full(100, 3.0))
    assert mean == pytest.approx(3.0)
    assert err == pytest.approx(0.0)


def test_block_error_known_series() -> None:
    mean, err = analyze.block_mean_error(np.arange(100, dtype=float), nblocks=10)
    assert mean == pytest.approx(49.5)
    # block means are 4.5, 14.5, ..., 94.5: sample std 30.2765, error = std / sqrt(10)
    assert err == pytest.approx(30.276503540974915 / np.sqrt(10))


def test_fit_diffusion_recovers_slope_and_ignores_offset() -> None:
    t = np.linspace(0.1, 10.0, 200)
    msd = 6 * 0.042 * t + 0.5
    assert analyze.fit_diffusion(t, msd, 2.0, 8.0) == pytest.approx(0.042)


def test_fit_diffusion_uses_only_the_window() -> None:
    t = np.linspace(0.1, 10.0, 200)
    msd = np.where(t < 2.0, 100.0 * t, 6 * 0.05 * t)  # ballistic-like junk early on
    assert analyze.fit_diffusion(t, msd, 3.0, 9.0) == pytest.approx(0.05)


def test_long_range_corrections_match_closed_forms() -> None:
    assert analyze.lrc_energy_per_particle(0.84, 3.0) == pytest.approx(-0.2605166599, abs=1e-9)
    assert analyze.lrc_pressure(0.84, 3.0) == pytest.approx(-0.4374677746, abs=1e-9)
    assert analyze.lrc_energy_per_particle(0.8442, 2.5) == pytest.approx(-0.4520126248, abs=1e-9)
    assert analyze.lrc_pressure(0.8442, 2.5) == pytest.approx(-0.7621346985, abs=1e-9)


def test_summarize_thermo_adds_corrections_and_skips_equilibration(tmp_path: Path) -> None:
    n, rho, rc = 500, 0.84, 3.0
    rows = ["phase,step,time,kinetic,potential_shifted,potential_plain,total,temperature,pressure"]
    rows += [f"equil,{i},0,0,0,0,0,9.0,9.0" for i in range(1, 21)]  # must be ignored
    rows += [f"nve,{i},0,0,0,{-5.0 * n},0,0.85,1.0" for i in range(1, 21)]
    path = tmp_path / "thermo.csv"
    path.write_text("\n".join(rows) + "\n")
    out = analyze.summarize_thermo(str(path), n, rho, rc)
    assert out["T"][0] == pytest.approx(0.85)
    assert out["P"][0] == pytest.approx(1.0 + analyze.lrc_pressure(rho, rc))
    assert out["U"][0] == pytest.approx(-5.0 + analyze.lrc_energy_per_particle(rho, rc))


def test_interpolate_to_temperature_is_exact_for_a_line() -> None:
    temps = np.array([0.84, 0.845, 0.85, 0.855, 0.86])
    value, err = analyze.interpolate_to_temperature(temps, 2.0 * temps + 1.0, 0.851)
    assert value == pytest.approx(2.0 * 0.851 + 1.0)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_interpolate_to_temperature_error_grows_with_scatter() -> None:
    rng = np.random.default_rng(0)
    temps = np.linspace(0.84, 0.86, 8)
    _, err_small = analyze.interpolate_to_temperature(temps, temps + rng.normal(0, 0.001, 8), 0.851)
    _, err_large = analyze.interpolate_to_temperature(temps, temps + rng.normal(0, 0.01, 8), 0.851)
    assert err_large > err_small > 0.0

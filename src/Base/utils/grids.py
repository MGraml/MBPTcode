"""Quadrature grids: Gauss-Legendre, gap-scaled, and minimax frequency/time.

The minimax grid tables in minimax_omega_data.json and minimax_tau_data.json
are derived from the GreenX library (https://github.com/nomad-coe/greenX),
which is distributed under the Apache License 2.0.
"""
import json
import os
import numpy as np

_MINIMAX_DATA_PATH = os.path.join(os.path.dirname(__file__), 'minimax_omega_data.json')
_minimax_data_cache = None

_MINIMAX_TAU_DATA_PATH = os.path.join(os.path.dirname(__file__), 'minimax_tau_data.json')
_minimax_tau_data_cache = None


def _load_minimax_data():
    global _minimax_data_cache
    if _minimax_data_cache is None:
        with open(_MINIMAX_DATA_PATH) as f:
            _minimax_data_cache = json.load(f)
    return _minimax_data_cache


def _load_minimax_tau_data():
    global _minimax_tau_data_cache
    if _minimax_tau_data_cache is None:
        with open(_MINIMAX_TAU_DATA_PATH) as f:
            _minimax_tau_data_cache = json.load(f)
    return _minimax_tau_data_cache


def minimax_supported_sizes():
    """Grid sizes with tabulated GreenX minimax frequency-grid data."""
    return sorted(int(k) for k in _load_minimax_data())


def minimax_tau_supported_sizes():
    """Grid sizes with tabulated GreenX minimax imaginary-time-grid data."""
    return sorted(int(k) for k in _load_minimax_tau_data())


def minimax_frequency_grid(nfreq, e_min, e_max):
    """Minimax imaginary-frequency grid, from tabulated GreenX coefficients (greenX/GX-TimeFrequency/src/minimax_omega.F90).

    A lookup (Remez-exchange-optimized table per grid_size/energy_range bin),
    not an on-the-fly optimization; dimensionless table entries rescaled by e_min.
    e_min/e_max: smallest/largest orbital-energy transition. Prefer this over
    gauss_legendre_grid where nfreq is in minimax_supported_sizes().
    """
    data = _load_minimax_data()
    key = str(nfreq)
    if key not in data:
        raise ValueError(
            f"nfreq={nfreq} has no tabulated GreenX minimax grid; supported sizes: {minimax_supported_sizes()}"
        )
    energy_range = data[key]['energy_range']
    matrix = data[key]['aw_erange_matrix']

    e_range = e_max / e_min
    ien = None  # 1-indexed, matches minimax_utils.F90::find_erange
    for i, er in enumerate(energy_range):
        if e_range < er:
            ien = i + 1
            break
    if ien is None:
        ien = len(energy_range) + 1  # fallback column for out-of-table e_range

    row = np.array(matrix[ien - 1])

    e_ratio = 1.0
    if ien == 1 and nfreq > 20:
        e_ratio = energy_range[0] / e_range
        if e_ratio > 1.5:
            e_ratio /= 1.5

    freq_point = row[:nfreq] / e_ratio * e_min
    freq_weight = row[nfreq:] / e_ratio * e_min
    return freq_point, freq_weight


def minimax_time_grid(ntau, e_min, e_max):
    """Minimax imaginary-time grid (tau_k, w_k) s.t. 1/x ~= sum_k w_k*exp(-x*tau_k)
    for x in [e_min, e_max], from tabulated GreenX coefficients
    (greenX/GX-TimeFrequency/src/minimax_tau.F90, "Laplace-transformed direct
    MP2" grids -- same table layout/lookup as minimax_frequency_grid, parsed
    by tools/parse_minimax_tau.py into minimax_tau_data.json).

    Rescaling is the INVERSE of minimax_frequency_grid's: the tabulated data is
    for a dimensionless problem with e_min=1 (energies rescaled by e_min), but
    tau/weight here have units of 1/energy (not energy, as omega/its weights
    do), so recovering the physical grid divides by e_min instead of
    multiplying -- verified numerically in tests/test_minimax_tau_grid.py by
    checking sum(w_k*exp(-x*tau_k)) reproduces 1/x directly.
    """
    data = _load_minimax_tau_data()
    key = str(ntau)
    if key not in data:
        raise ValueError(
            f"ntau={ntau} has no tabulated GreenX minimax tau grid; supported sizes: {minimax_tau_supported_sizes()}"
        )
    energy_range = data[key]['energy_range']
    matrix = data[key]['aw_erange_matrix']

    e_range = e_max / e_min
    ien = None  # 1-indexed, matches minimax_utils.F90::find_erange
    for i, er in enumerate(energy_range):
        if e_range < er:
            ien = i + 1
            break
    if ien is None:
        ien = len(energy_range) + 1  # fallback column for out-of-table e_range

    row = np.array(matrix[ien - 1])

    e_ratio = 1.0
    if ien == 1 and ntau > 20:
        e_ratio = energy_range[0] / e_range
        if e_ratio > 1.5:
            e_ratio /= 1.5

    tau_point = row[:ntau] / e_ratio / e_min
    tau_weight = row[ntau:] / e_ratio / e_min
    return tau_point, tau_weight


def gauss_legendre_grid(nfreq, w0=0.5):
    """Modified Gauss-Legendre grid on [0,infty) (eqs 85-87, Erhard, Fauser, Tuchin, Goerling, JCP 157, 114105 (2022)).

    w0 sets the grid's characteristic energy scale; see gap_scaled_w0 for a system-specific choice.
    """
    x, w = np.polynomial.legendre.leggauss(nfreq)
    freq_weight = w0 * 2.0 * w / (1.0 - x)**2
    freq_point = w0 * (1.0 + x) / (1.0 - x)
    return freq_point, freq_weight


def gap_scaled_w0(eps, nocc, scale=0.5):
    """w0 = scale * (eps_LUMO - eps_HOMO): scales the grid to the gap instead of using one fixed w0 for every system."""
    return scale * (eps[nocc] - eps[nocc - 1])

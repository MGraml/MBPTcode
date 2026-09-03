"""Imaginary-axis quadrature grids: minimax (tabulated) and Gauss-Legendre.

The minimax grids are lookups, not on-the-fly optimizations: a dimensionless
table entry per (grid size, energy range) bin, rescaled by e_min. That is why
only certain sizes exist -- see `minimax_supported_sizes()`.

References
----------
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) -- the
minimax imaginary-time and imaginary-frequency quadratures themselves.
The coefficient tables in minimax_omega_data.json / minimax_tau_data.json are
derived from the GreenX library (Apache-2.0; see NOTICE), whose
GX-TimeFrequency/src/minimax_omega.F90 and minimax_tau.F90 generate them.
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


def _require_positive_e_min(e_min, e_max, who):
    """Reject a non-positive smallest transition energy.

    Every grid in this module is a T = 0 construction scaled by e_min, the
    HOMO-LUMO gap. A metal has no gap, and the failures are silent: a tiny gap
    collapses the grid toward omega = 0, a numerically inverted (degenerate)
    Fermi level gives NEGATIVE frequency points and negative weights, and an
    exactly zero gap divides by zero. None of those announce themselves.

    At finite temperature the low-energy scale is the temperature, not the gap:
    pass e_min = utils.matsubara.thermal_e_min(beta, gap) instead, or use the
    intermediate representation there, which has no gap in its construction.
    """
    if not e_min > 0:
        raise ValueError(
            f"{who}: e_min = {e_min:.6g} is not positive, so this T=0 grid is "
            f"undefined. It is scaled by the smallest transition energy (the "
            f"gap), which vanishes for a metal and can come out negative for a "
            f"degenerate Fermi level. Use "
            f"src.Base.utils.matsubara.thermal_e_min(beta, gap) to set e_min "
            f"from the temperature, or the IRBasis there, which needs no gap.")
    if e_max is not None and not e_max > e_min:
        raise ValueError(
            f"{who}: e_max = {e_max:.6g} must exceed e_min = {e_min:.6g}")


def minimax_frequency_grid(nfreq, e_min, e_max):
    """Minimax imaginary-frequency grid, from GreenX coefficients tabulated in minimax_omega_data.json (GreenX: GX-TimeFrequency/src/minimax_omega.F90).

    A lookup (Remez-exchange-optimized table per grid_size/energy_range bin),
    not an on-the-fly optimization; dimensionless table entries rescaled by e_min.
    e_min/e_max: smallest/largest orbital-energy transition. Prefer this over
    gauss_legendre_grid where nfreq is in minimax_supported_sizes().
    """
    _require_positive_e_min(e_min, e_max, 'minimax_frequency_grid')
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
    tabulated in minimax_tau_data.json (GreenX: GX-TimeFrequency/src/minimax_tau.F90,
    the "Laplace-transformed direct MP2" grids -- same table layout and lookup
    as minimax_frequency_grid).

    Rescaling is the INVERSE of minimax_frequency_grid's: the tabulated data is
    for a dimensionless problem with e_min=1 (energies rescaled by e_min), but
    tau/weight here have units of 1/energy (not energy, as omega/its weights
    do), so recovering the physical grid divides by e_min instead of
    multiplying -- verified numerically in tests/test_minimax_tau_grid.py by
    checking sum(w_k*exp(-x*tau_k)) reproduces 1/x directly.
    """
    _require_positive_e_min(e_min, e_max, 'minimax_time_grid')
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
    _require_positive_e_min(w0, None, 'gauss_legendre_grid (w0)')
    x, w = np.polynomial.legendre.leggauss(nfreq)
    freq_weight = w0 * 2.0 * w / (1.0 - x)**2
    freq_point = w0 * (1.0 + x) / (1.0 - x)
    return freq_point, freq_weight


def gap_scaled_w0(eps, nocc, scale=0.5):
    """w0 = scale * (eps_LUMO - eps_HOMO): scales the grid to the gap instead of using one fixed w0 for every system.

    Raises for a vanishing or inverted gap rather than returning a w0 that
    would silently collapse the grid -- see _require_positive_e_min, and
    utils.matsubara for the finite-temperature replacement.
    """
    w0 = scale * (eps[nocc] - eps[nocc - 1])
    _require_positive_e_min(w0, None, 'gap_scaled_w0')
    return w0

"""
Validates src/Base/utils/grids.py::minimax_time_grid (GreenX minimax imaginary-time
grid, parsed by tools/parse_minimax_tau.py from
greenX/GX-TimeFrequency/src/minimax_tau.F90) by checking the Laplace-transform
identity it exists to provide:

    1/x ~= sum_k w_k * exp(-x * tau_k)      for x in [e_min, e_max]

This is the standard quadrature used in Laplace-transformed MP2 to factorize
an energy denominator into separable per-particle exponentials; the same
trick generalizes directly to MP3's six-index (three-particle) denominator
D_ijk^abc = (e_a+e_b+e_c) - (e_i+e_j+e_k), since
exp(-D*tau) = exp(-e_a*tau)*exp(-e_b*tau)*exp(-e_c*tau)*exp(e_i*tau)*exp(e_j*tau)*exp(e_k*tau)
is fully separable -- the actual target this grid is being built for
(see the "Cholesky/Laplace-type denominator factorization" discussion).

minimax_time_grid's rescaling is the INVERSE of minimax_frequency_grid's
(divide by e_min instead of multiply), since tau/its weights have units of
1/energy rather than energy -- this is the one place a naive copy-paste of
minimax_frequency_grid would have silently gotten the units backwards, so it
is the main thing this test locks in.

Usage: python tests/test_minimax_tau_grid.py (no pytest, per project convention).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np

from src.Base.utils.grids import minimax_time_grid, minimax_tau_supported_sizes

# (grid_size, e_min, e_max, tolerance) -- e_min/e_max chosen within (or very
# close to) the energy_range bin each grid_size is actually tabulated for
# (small grids for narrow ratios, large grids for wide ratios); picking a
# grid size far outside its tabulated range is a user error, not something
# this test needs to cover (confirmed separately: ntau=30 misused on a
# ratio-100 problem gives ~40% error, vs <1e-10 on its native ratio-4000
# range).
CASES = [
    (6, 1.0, 3.0, 1e-6),
    (8, 1.0, 12.0, 1e-5),
    (14, 0.3, 30.0, 1e-6),
    (20, 0.3, 30.0, 1e-9),
    (30, 1.0, 4000.0, 1e-8),
]


def check_case(ntau, e_min, e_max, tol):
    tau, w = minimax_time_grid(ntau, e_min, e_max)
    assert tau.shape == (ntau,) and w.shape == (ntau,)
    xs = np.geomspace(e_min, e_max, 25)
    approx = np.array([np.sum(w * np.exp(-x * tau)) for x in xs])
    exact = 1.0 / xs
    relerr = np.max(np.abs(approx - exact) / exact)
    status = "OK" if relerr < tol else "FAIL"
    print(f"  ntau={ntau:3d}  range=({e_min},{e_max})  max relerr={relerr:.3e}  (tol={tol:.0e})  [{status}]")
    assert relerr < tol, f"ntau={ntau}: relerr {relerr:.3e} exceeds tol {tol:.0e}"


def main():
    print(f"Supported minimax tau grid sizes: {minimax_tau_supported_sizes()}")
    print("Checking 1/x ~= sum_k w_k*exp(-x*tau_k):")
    for ntau, e_min, e_max, tol in CASES:
        check_case(ntau, e_min, e_max, tol)
    print("\nAll minimax_time_grid checks passed.")


if __name__ == '__main__':
    main()

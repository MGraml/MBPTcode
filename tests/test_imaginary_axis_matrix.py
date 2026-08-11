import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.utils.grids import minimax_frequency_grid
from src.SingleReference.GW.imaginary_axis import (
    solve_screening_imaginary_axis,
    self_energy_imaginary_axis,
    self_energy_matrix_imaginary_axis,
)

if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    norb = len(eps)
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    eta = 1e-3

    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=eta)

    nfreq = 8
    occ, virt = np.arange(nocc), np.arange(nocc, norb)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    freq_points, freq_weights = minimax_frequency_grid(nfreq, e_min, e_max)
    W_grid = solve_screening_imaginary_axis(lr, nocc, freq_points)

    sigma_matrix = self_energy_matrix_imaginary_axis(df_coeff, eps, freq_points, freq_weights, W_grid)
    print(f"sigma_matrix shape: {sigma_matrix.shape}")

    all_ok = True

    # Direct check: query at exactly freq_points (matching the matrix builder's default)
    # against the existing (already-validated) scalar function evaluated at those same
    # points, for every orbital p.
    max_diag_diff2 = 0.0
    for p in range(norb):
        sigma_scalar_unsigned = self_energy_imaginary_axis(df_coeff, eps, nocc, p, freq_points,
                                                             freq_weights, W_grid, freq_points)
        d = np.max(np.abs(sigma_matrix[:, p, p] - sigma_scalar_unsigned))
        max_diag_diff2 = max(max_diag_diff2, d)

    ok = max_diag_diff2 < 1e-10
    all_ok &= ok
    print(f"self_energy_matrix_imaginary_axis diagonal vs scalar self_energy_imaginary_axis "
          f"(all p, query=freq_points): maxdiff={max_diag_diff2:.2e}  {'OK' if ok else 'FAIL'}")

    # Symmetry sanity check: with real, symmetric Wc/C and G_mm(iw) carrying no p/q
    # dependence, Sigma_c,pq(iw) = Sigma_c,qp(iw) (plain transpose symmetry -- NOT
    # Hermitian, since G(iw) is genuinely complex, not real).
    sym_diff = np.max(np.abs(sigma_matrix - sigma_matrix.transpose(0, 2, 1)))
    ok_sym = sym_diff < 1e-10
    all_ok &= ok_sym
    print(f"sigma_matrix p<->q symmetry: maxdiff={sym_diff:.2e}  {'OK' if ok_sym else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")

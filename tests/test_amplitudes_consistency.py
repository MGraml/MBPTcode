import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients, get_two_electron_integrals_chemist
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver

if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    nocc = mol.nelectron // 2
    p_state = nocc - 1

    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    lr_df = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=1e-3)
    A, B = lr_df.build_casida_matrices(nocc, lBSE=False)
    omega, X, Y = CasidaSolver(A, B).solve()
    w_aux = lr_df.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    # get_chi_a: the p_state-sliced blocked code path (the one the amplitudes.py
    # contiguity fix touched) must exactly match slicing the full (p_state=None)
    # matrix -- these are the same mathematical quantity computed two ways.
    se_df = SelfEnergySolver(eps, df_coeff=df_coeff, spin_mode='restricted', eta=1e-3)
    chi_a_full = se_df.get_chi_a(nocc, X, Y, p_state=None)
    chi_a_sliced = se_df.get_chi_a(nocc, X, Y, p_state=p_state)
    ok = np.allclose(chi_a_sliced, chi_a_full[:, :, p_state], atol=1e-10)
    all_ok &= ok
    print(f"get_chi_a_df: p_state-sliced == full[:, :, p_state]: {'OK' if ok else 'FAIL'} "
          f"(max diff={np.max(np.abs(chi_a_sliced - chi_a_full[:, :, p_state])):.2e})")

    chi_b_full = se_df.get_chi_b_vertex(nocc, X, Y, eri_w=w_aux, p_state=None)
    chi_b_sliced = se_df.get_chi_b_vertex(nocc, X, Y, eri_w=w_aux, p_state=p_state)
    ok = np.allclose(chi_b_sliced, chi_b_full[:, :, p_state], atol=1e-10)
    all_ok &= ok
    print(f"get_chi_b_vertex_df: p_state-sliced == full[:, :, p_state]: {'OK' if ok else 'FAIL'} "
          f"(max diff={np.max(np.abs(chi_b_sliced - chi_b_full[:, :, p_state])):.2e})")

    # Same checks on the full-ERI (non-DF) code path.
    se_full = SelfEnergySolver(eps, eri_chemist=eri, spin_mode='restricted', eta=1e-3)
    lr_full = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted', eta=1e-3)
    A_f, B_f = lr_full.build_casida_matrices(nocc, lBSE=False)
    omega_f, X_f, Y_f = CasidaSolver(A_f, B_f).solve()

    chi_a_full_e = se_full.get_chi_a(nocc, X_f, Y_f, p_state=None)
    chi_a_sliced_e = se_full.get_chi_a(nocc, X_f, Y_f, p_state=p_state)
    ok = np.allclose(chi_a_sliced_e, chi_a_full_e[:, :, p_state], atol=1e-10)
    all_ok &= ok
    print(f"get_chi_a_full: p_state-sliced == full[:, :, p_state]: {'OK' if ok else 'FAIL'} "
          f"(max diff={np.max(np.abs(chi_a_sliced_e - chi_a_full_e[:, :, p_state])):.2e})")

    chi_b_full_e = se_full.get_chi_b_vertex(nocc, X_f, Y_f, eri_w=eri, p_state=None)
    chi_b_sliced_e = se_full.get_chi_b_vertex(nocc, X_f, Y_f, eri_w=eri, p_state=p_state)
    ok = np.allclose(chi_b_sliced_e, chi_b_full_e[:, :, p_state], atol=1e-10)
    all_ok &= ok
    print(f"get_chi_b_vertex_full: p_state-sliced == full[:, :, p_state]: {'OK' if ok else 'FAIL'} "
          f"(max diff={np.max(np.abs(chi_b_sliced_e - chi_b_full_e[:, :, p_state])):.2e})")

    # DF and full-ERI should agree closely for this tiny basis (small fitting error).
    ok = np.max(np.abs(chi_a_sliced - chi_a_sliced_e)) < 1e-3
    all_ok &= ok
    print(f"DF vs full-ERI chi_a agree to <1e-3: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")

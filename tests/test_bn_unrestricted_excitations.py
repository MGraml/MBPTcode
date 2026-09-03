import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver

if __name__ == '__main__':
    all_ok = True

    # BN, triplet ground state (2 unpaired electrons) -- forces a genuine
    # unrestricted (UHF) reference, needed to exercise the spin-flip
    # (Sz=+-1, "triplet" in the open-shell sense) Casida machinery.
    mol = gto.M(atom='B 0 0 0; N 0 0 1.28', basis='cc-pvdz', spin=2)
    mf = scf.UHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    ok = mf.converged
    all_ok &= ok
    print(f"BN/cc-pvdz UHF (spin=2) converged: {'OK' if ok else 'FAIL'}  E={mf.e_tot:.6f}")

    nocc = mf.nelec  # (nocc_a, nocc_b)
    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    mo_a, mo_b = mf.mo_coeff
    S_AO = mol.intor_symmetric('int1e_ovlp')
    S_ab = mo_a.T @ S_AO @ mo_b
    df_a, df_b = df_coeff
    df_ab = np.einsum('ia, pik -> pka', S_ab, df_a)
    df_coeff = (df_a, df_b, df_ab)

    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='unrestricted', eta=1e-3)
    w_aux = lr.solve_rpa_screening(np.array([0.0]), nocc, is_imaginary=True)[0]

    def check(label, omega):
        lowest = np.min(omega)
        ok = np.all(np.isfinite(omega)) and lowest > 0
        print(f"{label:32s}: lowest excitation = {lowest:10.6f} Ha  (n_states={len(omega)})  {'OK' if ok else 'FAIL'}")
        return ok

    # Spin-conserving (Sz=0) channel -- for an unrestricted reference this is
    # a single combined alpha/beta problem (no separate singlet/triplet split
    # the way a closed-shell restricted reference has; that spin-adaptation
    # only exists for RHF references).
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    omega, _, _ = CasidaSolver(A, B).solve()
    all_ok &= check('RPA (Sz=0)', omega)

    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=None)
    omega, _, _ = CasidaSolver(A, B).solve()
    all_ok &= check('TDHF (Sz=0, bare exchange)', omega)

    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux)
    omega, _, _ = CasidaSolver(A, B).solve()
    all_ok &= check('BSE (Sz=0, screened exchange)', omega)

    # Spin-flip (Sz=+-1, the open-shell analog of a "triplet" channel):
    # ba = occ_beta -> virt_alpha, ab = occ_alpha -> virt_beta.
    for channel in ['ba', 'ab']:
        A, B = lr.build_spin_flip_casida_matrices(nocc, lBSE=False, channel=channel)
        omega, _, _ = CasidaSolver(A, B).solve()
        all_ok &= check(f'RPA spin-flip ({channel})', omega)

        A, B = lr.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=None, channel=channel)
        omega, _, _ = CasidaSolver(A, B).solve()
        all_ok &= check(f'TDHF spin-flip ({channel})', omega)

        A, B = lr.build_spin_flip_casida_matrices(nocc, lBSE=True, W_aux=w_aux, channel=channel)
        omega, _, _ = CasidaSolver(A, B).solve()
        all_ok &= check(f'BSE spin-flip ({channel})', omega)

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)

"""Upfolded (frequency-free) dynamical BSE -- Bintrim & Berkelbach 2022.

The validation chain here is unusually strong because the paper hands us an
exact algebraic identity to test against. Downfolding the doubles out of the
upfolded matrix (Eq. 9) must return the frequency-dependent BSE matrix
(Eq. 1), and that matrix can be built a completely different way -- by
diagonalizing the RPA problem and summing poles (Eq. 6). Two independent
routes to the same object, exact at every frequency, is what pins the
construction before any solver is involved.

Run: python tests/test_bse_upfolded.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_two_electron_integrals_chemist,
                                      DFIntegrals)
from src.SingleReference.BSE import bse_upfolded as B
from src.SingleReference.BSE.bse_upfolded import solve_bse_upfolded, qp_energies
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Base.constants import HARTREE_TO_EV

SPINS = ('singlet', 'triplet')
WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def _case(basis='sto-3g', atom=WATER):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'))
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    no, norb = mol.nelectron // 2, mol.nao
    gb = B.eri_blocks(eri, no, norb)
    # quasiparticle energies deliberately DIFFERENT from eps: the two enter
    # different blocks (E in A and the outer doubles pair, eps in S), and a
    # mixup is invisible if they are equal
    rng = np.random.default_rng(0)
    e_qp = eps + 0.05 * rng.standard_normal(norb)
    return mol, mf, eps, e_qp, gb, no, norb - no


def test_downfolding_equals_sum_over_states():
    """Eq. (9) == Eq. (1) with the kernel (6). The decisive test."""
    _, _, eps, e_qp, gb, no, nv = _case()
    ok = True
    for spin in SPINS:
        for omega in (0.3, 0.7, -0.2):
            a = B.downfold(eps, e_qp, gb, no, nv, omega, spin)
            b = B.dynamical_bse_matrix(eps, e_qp, gb, no, nv, omega, spin)
            d = np.abs(a - b).max()
            ok &= check(d < 1e-10, f'{spin}, omega={omega:+.1f}: downfold == '
                        'sum over states', f'{d:.1e}')
    return ok


def test_eigenvalues_solve_the_frequency_dependent_problem():
    """Every eigenvalue Omega of the upfolded H must be an eigenvalue of
    A(Omega) -- the self-consistency the upfolding is supposed to deliver."""
    _, _, eps, e_qp, gb, no, nv = _case()
    ok = True
    for spin in SPINS:
        w, _ = B.solve_dense(eps, e_qp, gb, no, nv, spin, nroots=5)
        d = max(np.abs(np.linalg.eigvals(
            B.downfold(eps, e_qp, gb, no, nv, om, spin)) - om).min() for om in w)
        ok &= check(d < 1e-9, f'{spin}: every Omega is an eigenvalue of A(Omega)',
                    f'{d:.1e}')
    return ok


def test_sigma_matches_dense_hamiltonian():
    _, _, eps, e_qp, gb, no, nv = _case()
    ok = True
    for spin in SPINS:
        n = B.dimensions(no, nv)['nH']
        H = B.build_hamiltonian(eps, e_qp, gb, no, nv, spin)
        Hs = np.column_stack([B.sigma(np.eye(n)[:, k], eps, e_qp, gb, no, nv, spin)
                              for k in range(n)])
        d = np.abs(H - Hs).max()
        ok &= check(d < 1e-10, f'{spin}: matrix-free sigma == dense H', f'{d:.1e}')
        d = np.abs(np.diag(H) - B.diagonal(eps, e_qp, gb, no, nv, spin)).max()
        ok &= check(d < 1e-10, f'{spin}: diagonal == diag(H)', f'{d:.1e}')
    return ok


def test_df_sigma_matches_exact_integrals():
    mol, mf, eps, e_qp, gb, no, nv = _case()
    Bf = DFIntegrals.from_scf(mol, mf, exact=True).B_aa
    ok = True
    for spin in SPINS:
        n = B.dimensions(no, nv)['nH']
        d = 0.0
        for k in range(0, n, max(1, n // 12)):
            v = np.eye(n)[:, k]
            d = max(d, np.abs(B.sigma(v, eps, e_qp, gb, no, nv, spin)
                              - B.sigma_df(v, eps, e_qp, gb, Bf, no, nv, spin)).max())
        ok &= check(d < 1e-10, f'{spin}: DF sigma == exact-integral sigma', f'{d:.1e}')
    return ok


def test_davidson_matches_dense():
    _, _, eps, e_qp, gb, no, nv = _case()
    ok = True
    for spin in SPINS:
        w_ref, _ = B.solve_dense(eps, e_qp, gb, no, nv, spin, nroots=4)
        diag = B.diagonal(eps, e_qp, gb, no, nv, spin)
        w, _ = B.davidson_nonsymmetric(
            lambda v: B.sigma(v, eps, e_qp, gb, no, nv, spin), diag, nroots=4)
        d = np.abs(np.sort(w) - w_ref).max() * HARTREE_TO_EV
        ok &= check(d < 1e-5, f'{spin}: Davidson == dense', f'{d:.1e} eV')
    return ok


def test_hamiltonian_is_asymmetric():
    """Not a detail to be tidied away: the (1,2) blocks are -V^e/-V^h while
    the (2,1) blocks are +(V^h)^T/+(V^e)^T. This is the structural difference
    from ADC, whose ISR matrix is Hermitian by construction, and it is why
    the solver has to be a non-symmetric one."""
    _, _, eps, e_qp, gb, no, nv = _case()
    H = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'singlet')
    d = np.abs(H - H.T).max()
    return check(d > 1e-3, 'H is genuinely non-symmetric', f'max|H - H^T| {d:.1e}')


def test_screening_matrix_is_the_direct_rpa_casida_matrix():
    """S must be the direct (Hartree-only) RPA matrix in the TDA. The repo
    already builds that for the GW module, so compare rather than trust."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'))
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    no, norb = mol.nelectron // 2, mol.nao
    gb = B.eri_blocks(eri, no, norb)
    S = B.block_S(eps, gb, no, norb - no).reshape(no * (norb - no), -1)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    A_rpa, _ = lr.build_casida_matrices(no, lBSE=False)
    d = np.abs(S - np.asarray(A_rpa)).max()
    return check(d < 1e-12, 'S == the direct RPA Casida A matrix', f'{d:.1e}')


def test_triplet_drops_the_coulomb_term_only():
    """kappa multiplies (ia|jb) and nothing else: the doubles blocks and the
    couplings are direct-only screening, identical in both spin channels."""
    _, _, eps, e_qp, gb, no, nv = _case()
    Hs = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'singlet')
    Ht = B.build_hamiltonian(eps, e_qp, gb, no, nv, 'triplet')
    n_s = no * nv
    d_rows = np.abs((Hs - Ht)[n_s:, :]).max()
    d_cols = np.abs((Hs - Ht)[:, n_s:]).max()
    d_ss = np.abs((Hs - Ht)[:n_s, :n_s]
                  - 2.0 * gb['ovov'].reshape(n_s, n_s)).max()
    ok = check(max(d_rows, d_cols) < 1e-12,
               'singlet and triplet share every doubles block',
               f'{max(d_rows, d_cols):.1e}')
    ok &= check(d_ss < 1e-12, 'they differ by exactly 2 (ia|jb) in A', f'{d_ss:.1e}')
    return ok


def test_familiar_form_is_symmetric_and_lies_below():
    """Eq. (12). The paper reports it comes out 2-3 eV below the exact
    dynamical BSE; reproducing the SIGN and rough size of that gap is a check
    on both constructions at once."""
    _, _, eps, e_qp, gb, no, nv = _case()
    Hf = B.build_hamiltonian_familiar(eps, gb, no, nv, 'singlet')
    d = np.abs(Hf - Hf.T).max()
    ok = check(d < 1e-12, 'the familiar form is symmetric', f'{d:.1e}')
    lowest_familiar = np.sort(np.linalg.eigvalsh(Hf))
    lowest_familiar = lowest_familiar[lowest_familiar > 0][0]
    w, _ = B.solve_dense(eps, eps, gb, no, nv, 'singlet', nroots=1)
    gap = (w[0] - lowest_familiar) * HARTREE_TO_EV
    ok &= check(lowest_familiar < w[0] and 0.5 < gap < 6.0,
                'and lies below the dynamical BSE by a few eV', f'{gap:.2f} eV')
    return ok


def test_doubles_weight_is_a_fraction():
    _, _, eps, e_qp, gb, no, nv = _case()
    _, X = B.solve_dense(eps, e_qp, gb, no, nv, 'singlet', nroots=3)
    r2 = [B.doubles_weight(X[:, k], no, nv) for k in range(X.shape[1])]
    return check(all(0.0 <= r <= 100.0 for r in r2),
                 'doubles weight is a percentage', ' '.join(f'{r:.2f}' for r in r2))


def test_driver_end_to_end_with_gw_energies():
    """The full path: GW quasiparticle energies -> upfolded BSE."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    e_qp = qp_energies(mf)
    no = mol.nelectron // 2
    ok = check(e_qp[no - 1] > mf.mo_energy[no - 1], 'GW raises the HOMO',
               f'{mf.mo_energy[no - 1] * HARTREE_TO_EV:.3f} -> '
               f'{e_qp[no - 1] * HARTREE_TO_EV:.3f} eV')
    ws, _ = solve_bse_upfolded(mf, mol, spin='singlet', nroots=3, e_qp=e_qp)
    wt, _ = solve_bse_upfolded(mf, mol, spin='triplet', nroots=3, e_qp=e_qp)
    ok &= check(wt[0] < ws[0], 'the triplet lies below the singlet',
                f'{wt[0] * HARTREE_TO_EV:.3f} < {ws[0] * HARTREE_TO_EV:.3f} eV')
    ok &= check(5.0 < ws[0] * HARTREE_TO_EV < 12.0,
                'the lowest singlet is in range',
                f'{ws[0] * HARTREE_TO_EV:.3f} eV')
    return ok


if __name__ == '__main__':
    tests = [test_downfolding_equals_sum_over_states,
             test_eigenvalues_solve_the_frequency_dependent_problem,
             test_sigma_matches_dense_hamiltonian,
             test_df_sigma_matches_exact_integrals,
             test_davidson_matches_dense,
             test_hamiltonian_is_asymmetric,
             test_screening_matrix_is_the_direct_rpa_casida_matrix,
             test_triplet_drops_the_coulomb_term_only,
             test_familiar_form_is_symmetric_and_lies_below,
             test_doubles_weight_is_a_fraction,
             test_driver_end_to_end_with_gw_energies]
    all_ok = True
    for t in tests:
        print(f'\n-- {t.__name__}')
        all_ok &= bool(t())
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

"""Exciton descriptors reduce to orbital moments for hand-built eigenvectors.

`exciton_descriptors` contracts (X, Y) with dipole and second-moment matrices in
the orbital basis. For an eigenvector with a single nonzero X_ia every
descriptor is an orbital expectation value, and with one X_ia and one Y_jb the
Y term and the X-Y cross term of <r_e r_h> appear with known weights. Both are
checked against the AO integrals directly, so an index or block mix-up shows as
a mismatch rather than as a plausible number. A centrosymmetric molecule then
checks d_eh = 0 on real Casida vectors, and the ABBA norm c_n = 1 + 2 Y^T Y; the
matrix-free Davidson vectors must give the dense descriptors (same pair order),
and an unrestricted mean field is refused.

Run: python tests/test_exciton_descriptors.py
"""
import os
import sys

import numpy as np
from pyscf import df, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.exciton_descriptors import exciton_descriptors
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

TOL = 1e-10


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def orbital_moments(mf, mol):
    """(mu, M): dipole <p|r|q> (3, nmo, nmo) and diagonal second moments
    <p|r_x^2|q> (3, nmo, nmo) in the orbital basis, origin at zero."""
    mo = mf.mo_coeff
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)
        ao_rr = mol.intor_symmetric('int1e_rr', comp=9)[[0, 4, 8]]
    mu = np.einsum('xmn,mp,nq->xpq', ao_r, mo, mo)
    big_m = np.einsum('xmn,mp,nq->xpq', ao_rr, mo, mo)
    return mu, big_m


def main():
    ok = True
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvirt = nmo - nocc
    mu, big_m = orbital_moments(mf, mol)

    print('\n=== 1. one X_ia: descriptors are moments of orbitals i and a ===')
    i, a = nocc - 1, nocc + 1                      # HOMO -> LUMO+1
    X = np.zeros((nocc * nvirt, 1))
    X[i * nvirt + (a - nocc), 0] = 1.0
    Y = np.zeros_like(X)
    d = exciton_descriptors(mf, mol, nocc, X, Y)
    r_h, r_e = mu[:, i, i], mu[:, a, a]
    sig_h = np.sqrt(big_m[:, i, i].sum() - r_h @ r_h)
    sig_e = np.sqrt(big_m[:, a, a].sum() - r_e @ r_e)
    d_eh = np.linalg.norm(r_h - r_e)
    ok &= check(abs(d['c_n'][0] - 1.0) < TOL, 'c_n = 1')
    ok &= check(np.abs(d['r_h'][0] - r_h).max() < TOL, '<r_h> = <i|r|i>')
    ok &= check(np.abs(d['r_e'][0] - r_e).max() < TOL, '<r_e> = <a|r|a>')
    ok &= check(abs(d['sigma_h'][0] - sig_h) < TOL, 'sigma_h from <i|r^2|i>')
    ok &= check(abs(d['sigma_e'][0] - sig_e) < TOL, 'sigma_e from <a|r^2|a>')
    ok &= check(abs(d['cov_eh'][0]) < TOL and abs(d['R_eh'][0]) < TOL,
                'COV_eh = R_eh = 0 for a product state')
    ok &= check(abs(d['d_eh'][0] - d_eh) < TOL, 'd_eh = |<a|r|a> - <i|r|i>|')
    ok &= check(abs(d['d_exc'][0] - np.sqrt(d_eh**2 + sig_e**2 + sig_h**2)) < TOL,
                'd_exc^2 = d_eh^2 + sigma_e^2 + sigma_h^2')
    sig_e_dir = np.sqrt(big_m[:, a, a] - r_e**2)
    sig_h_dir = np.sqrt(big_m[:, i, i] - r_h**2)
    ok &= check(np.abs(d['d_eh_dir'][0] - np.abs(r_h - r_e)).max() < TOL
                and np.abs(d['sigma_e_dir'][0] - sig_e_dir).max() < TOL
                and np.abs(d['sigma_h_dir'][0] - sig_h_dir).max() < TOL,
                'directional d_eh, sigma_e, sigma_h are the per-component moments')
    ok &= check(np.abs(d['cov_eh_mat'][0]).max() < TOL
                and np.abs(d['R_eh_mat'][0]).max() < TOL,
                'covariance and correlation matrices vanish for a product state')
    d_exc_dir = np.sqrt((r_h - r_e)**2 + sig_e_dir**2 + sig_h_dir**2)
    ok &= check(np.abs(d['d_exc_dir'][0] - d_exc_dir).max() < TOL,
                'directional d_exc from the per-component pieces')

    print('\n=== 2. one X_ia and one Y_jb: Y term and X-Y cross term ===')
    j, b = nocc - 2, nocc
    x, y = 0.9, 0.4
    X = np.zeros((nocc * nvirt, 1))
    Y = np.zeros_like(X)
    X[i * nvirt + (a - nocc), 0] = x
    Y[j * nvirt + (b - nocc), 0] = y
    d = exciton_descriptors(mf, mol, nocc, X, Y)
    c = x * x + y * y
    # hole on i (X term) and on b (Y term); electron on a and on j
    r_h = (x * x * mu[:, i, i] + y * y * mu[:, b, b]) / c
    r_e = (x * x * mu[:, a, a] + y * y * mu[:, j, j]) / c
    r_h2 = (x * x * big_m[:, i, i] + y * y * big_m[:, b, b]) / c
    r_e2 = (x * x * big_m[:, a, a] + y * y * big_m[:, j, j]) / c
    # <r_e^x r_h^x>: X-X, Y-Y and the cross term through <a|r|j> and <i|r|b>
    r_eh = (x * x * mu[:, a, a] * mu[:, i, i] + y * y * mu[:, j, j] * mu[:, b, b]
            + 2 * x * y * mu[:, a, j] * mu[:, i, b]) / c
    cov = (r_eh - r_e * r_h).sum()
    sig_h = np.sqrt(r_h2.sum() - r_h @ r_h)
    sig_e = np.sqrt(r_e2.sum() - r_e @ r_e)
    d_eh = np.linalg.norm(r_h - r_e)
    ok &= check(abs(d['c_n'][0] - c) < TOL, 'c_n = x^2 + y^2')
    ok &= check(np.abs(d['r_h'][0] - r_h).max() < TOL,
                '<r_h> mixes <i|r|i> and <b|r|b>')
    ok &= check(np.abs(d['r_e'][0] - r_e).max() < TOL,
                '<r_e> mixes <a|r|a> and <j|r|j>')
    ok &= check(abs(d['sigma_h'][0] - sig_h) < TOL
                and abs(d['sigma_e'][0] - sig_e) < TOL,
                'sigma_e, sigma_h with both terms')
    ok &= check(abs(d['cov_eh'][0] - cov) < TOL,
                'COV_eh carries the X-Y cross term 2xy <a|r|j><i|r|b>')
    ok &= check(abs(d['R_eh'][0] - cov / (sig_e * sig_h)) < TOL,
                'R_eh = COV/(sigma_e sigma_h)')
    d_exc = np.sqrt(d_eh**2 + sig_e**2 + sig_h**2 - 2 * cov)
    ok &= check(abs(d['d_exc'][0] - d_exc) < TOL, 'd_exc from the four pieces')
    # full <r_e^x r_h^y>: electron component on <a|.|a>, <j|.|j>, <a|.|j>; hole
    # component on <i|.|i>, <b|.|b>, <i|.|b>
    r_eh_mat = (x * x * np.outer(mu[:, a, a], mu[:, i, i])
                + y * y * np.outer(mu[:, j, j], mu[:, b, b])
                + x * y * (np.outer(mu[:, a, j], mu[:, i, b])
                           + np.outer(mu[:, i, b], mu[:, a, j]))) / c
    cov_mat = r_eh_mat - np.outer(r_e, r_h)
    sig_e_dir = np.sqrt(r_e2 - r_e**2)
    sig_h_dir = np.sqrt(r_h2 - r_h**2)
    ok &= check(np.abs(d['cov_eh_mat'][0] - cov_mat).max() < TOL,
                'covariance matrix, electron component first, with the cross term')
    r_mat = cov_mat / np.outer(sig_e_dir, sig_h_dir)
    ok &= check(np.abs(d['R_eh_mat'][0] - r_mat).max() < TOL,
                'correlation matrix = cov / (sigma_e^x sigma_h^y)')
    ok &= check(abs(np.trace(d['cov_eh_mat'][0]) - d['cov_eh'][0]) < TOL
                and abs((d['sigma_e_dir'][0]**2).sum() - d['sigma_e'][0]**2) < TOL
                and abs((d['d_exc_dir'][0]**2).sum() - d['d_exc'][0]**2) < TOL
                and abs(np.linalg.norm(d['d_eh_dir'][0]) - d['d_eh'][0]) < TOL,
                'directional pieces sum back to the scalar descriptors')

    print('\n=== 3. centrosymmetric molecule, real Casida vectors ===')
    mol2 = gto.M(atom='N 0 0 -0.549; N 0 0 0.549', basis='6-31g', verbose=0)
    mf2 = scf.RHF(mol2).run()
    nocc2 = mol2.nelectron // 2
    eps2 = get_orbital_energies(mf2, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol2, mf2, representation='spatial')
    lr = LinearResponseSolver(eps2, eri_chemist=eri, spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc2, lBSE=False)
    omega, X2, Y2 = CasidaSolver(A, B).solve()
    order = np.argsort(omega)[:6]
    X2, Y2 = X2[:, order], Y2[:, order]
    d = exciton_descriptors(mf2, mol2, nocc2, X2, Y2)
    ok &= check(np.abs(d['d_eh']).max() < 1e-8, 'd_eh = 0 for every root of N2',
                f'max {np.abs(d["d_eh"]).max():.1e} bohr')
    yy = np.einsum('pn,pn->n', Y2, Y2)
    ok &= check(np.abs(d['c_n'] - 1.0 - 2.0 * yy).max() < TOL,
                'c_n = 1 + 2 Y^T Y with the X^T X - Y^T Y = 1 normalization')
    d_tda = exciton_descriptors(mf2, mol2, nocc2, X2, np.zeros_like(Y2))
    ok &= check(np.all(d_tda['sigma_e'] > 0) and np.all(d['d_exc'] > 0),
                'sizes are positive', f'd_exc = {np.round(d["d_exc"], 3)} bohr')

    print('\n=== 4. matrix-free Davidson vectors give the dense descriptors ===')
    # The Davidson action needs the DF factor; the dense build takes the same one.
    mf_df = scf.RHF(mol).density_fit()
    mf_df.with_df.auxbasis = df.make_auxbasis(mol)
    mf_df.run()
    eps_df = get_orbital_energies(mf_df, representation='spatial')
    coeff = get_density_fitting_coefficients(mol, mf_df, representation='spatial')
    lr = LinearResponseSolver(eps_df, coeff_df=coeff, spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    omega, Xd, Yd = CasidaSolver(A, B).solve()
    order = np.argsort(omega)[:3]
    dense = exciton_descriptors(mf_df, mol, nocc, Xd[:, order], Yd[:, order])
    om_dav, X_dav, Y_dav = solve_casida_davidson(lr, nocc, nroots=3,
                                                 polarizability='RPA', conv_tol=1e-8)
    order = np.argsort(om_dav)
    dav = exciton_descriptors(mf_df, mol, nocc, X_dav[:, order], Y_dav[:, order])
    worst = max(np.abs(dav[k] - dense[k]).max() for k in ('d_eh', 'sigma_e', 'sigma_h',
                                                          'd_exc', 'R_eh', 'c_n',
                                                          'd_exc_dir', 'cov_eh_mat'))
    ok &= check(worst < 1e-5,
                'Davidson and dense (X, Y) agree on every descriptor, 3 roots',
                f'max |d| {worst:.1e}')

    print('\n=== 5. unrestricted reference is refused ===')
    mf_u = scf.UHF(mol).run()
    try:
        exciton_descriptors(mf_u, mol, nocc, X, Y)
        raised = False
    except NotImplementedError:
        raised = True
    ok &= check(raised, 'UHF mean field raises NotImplementedError')

    print('\n' + ('All exciton descriptor checks passed.' if ok
                  else 'FAILURES DETECTED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

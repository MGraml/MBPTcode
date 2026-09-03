"""The exactness claims of the blocked, screened AO self-energy.

`self_energy_matrix_imaginary_time` holds (M, M) intermediates. Unblocked that
is 159 GB at the 476-atom hexamer/cc-pVDZ, which the OOM killer ended. Blocking
the interpolation index is a pure regrouping of a sum and must not move the
answer; the geometric screen drops only block pairs whose bounding spheres are
further apart than the cutoff, and on separated fragments must not move it
either. X_ao is carried from the fit rather than inverted out of X_mo, which
has to agree wherever the inversion is legal.

Checks 2-4 compare the kernel against ITSELF under a regrouping, so a defect
sitting on both sides survives all of them. Check 5 is the one that does not:
`self_energy_imaginary_time` is the same physics written in four lines -- no
blocking, no Morton order, no screening, no tau streaming -- and the optimized
kernel has to reproduce it. It stops in the time domain and works in the basis
its collocation is in, so the comparison applies the tau -> omega
recombination by hand and rotates the production result AO -> MO.

Checks 6 and 7 are structural: Sigma is symmetric in its two AO indices, and
Sigma(mu - i.w) is the conjugate of Sigma(mu + i.w). Note that 7 is weak --
it holds whenever the fitted cosine weights are even in omega and the sine
weights odd, which is a property of the transform rather than of the
self-energy assembly, and it comes out at exactly zero.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto

from src.Base.separable_ri import molecular_points_covariant, optimize_atomic_radii
from src.Base.utils.grids import minimax_frequency_grid, minimax_time_grid
from src.Base.utils.time_frequency import (minimax_transform_weights,
                                           COSINE_TW, COSINE_WT, SINE_TW)
from src.SingleReference.GW.imaginary_time import (
    _morton_order, _transform_screened, self_energy_fit_ranges,
    self_energy_imaginary_time, self_energy_matrix_imaginary_time,
    sigma_ao_to_mo)
from src.SingleReference.GW.space_time import (_ao_collocation, separable_factors,
                                               solve_qp_energy_space_time)
from src.Base.constants import HARTREE_TO_EV



def _rpa_w(X_mo, D, eps, nocc, omega):
    """W(i.omega) = [I - Pi]^-1 in the auxiliary gauge D was whitened in."""
    naux, norb = D.shape[1], len(eps)
    occ, virt = np.arange(nocc), np.arange(nocc, norb)
    Bov = np.einsum('Pi,Pa,PA->Aia', X_mo[:, occ], X_mo[:, virt], D, optimize=True)
    de = eps[virt][None, :] - eps[occ][:, None]
    out = np.empty((len(omega), naux, naux))
    for i, w in enumerate(omega):
        Pi = np.einsum('Aia,ia,Bia->AB', Bov, 4.0 * de / (de**2 + w**2), Bov,
                       optimize=True)
        out[i] = np.linalg.inv(np.eye(naux) - Pi)
    return out


if __name__ == '__main__':
    # Two fragments far enough apart that cross-fragment blocks are genuinely
    # negligible -- a compact molecule has no distance in it and screens nothing.
    R = 12.0
    mol = gto.M(atom=f'O 0 0 .117; H 0 .757 -.47; H 0 -.757 -.47; '
                     f'O {R} 0 .117; H {R} .757 -.47; H {R} -.757 -.47',
                basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = mf.mo_energy
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])

    X_mo, D, X_ao, coords = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    M, naux = X_ao.shape[0], D.shape[1]
    all_ok = True

    # 1. X_ao comes from the fit; the inversion is only a fallback. They must
    #    agree wherever mo_coeff is square, which is the only case it is legal.
    d_ao = np.max(np.abs(X_ao - _ao_collocation(X_mo, mf)))
    ok = d_ao < 1e-12
    all_ok &= ok
    print(f'1. carried X_ao vs inverted: max|d|={d_ao:.2e}  {"OK" if ok else "FAIL"}')

    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    n = 18
    om = minimax_frequency_grid(n, *rW)[0]
    tau = 0.5 * minimax_time_grid(n, *rS)[0]
    W = _rpa_w(X_mo, D, eps, nocc, om)
    out = np.array([0.05, 0.2])

    # 2. Blocking regroups a sum over P; only the summation order changes.
    ref = self_energy_matrix_imaginary_time(X_ao, D, W, mf.mo_coeff, eps, nocc,
                                            tau, om, out, mu=mu,
                                            block_memory_gb=1e6)
    print('2. blocked == unblocked:')
    ok2 = True
    for gb in (1e-3, 1e-4):
        sig = self_energy_matrix_imaginary_time(X_ao, D, W, mf.mo_coeff, eps,
                                                nocc, tau, om, out, mu=mu,
                                                block_memory_gb=gb)
        rel = np.max(np.abs(sig - ref)) / np.max(np.abs(ref))
        ok = rel < 1e-13
        ok2 &= ok
        print(f'   block_memory_gb={gb:<8g} rel={rel:.2e}  {"OK" if ok else "FAIL"}')
    all_ok &= ok2

    # 3. The screen must actually skip something here, and must not move Sigma.
    #    Both halves matter: a cutoff that skips nothing would pass the error
    #    check while proving nothing.
    idx = _morton_order(coords)
    P = coords[idx]
    b = max(1, min(M, int((-naux + np.sqrt(naux**2 + 8 * 1e-4 * 1e9 / 8)) / 4)))
    edges = list(range(0, M, b)) + [M]
    pr = list(zip(edges[:-1], edges[1:]))
    cen = np.array([P[a:c].mean(axis=0) for a, c in pr])
    rad = np.array([np.linalg.norm(P[a:c] - q, axis=1).max()
                    for (a, c), q in zip(pr, cen)])
    sep = np.linalg.norm(cen[:, None] - cen[None, :], axis=2)
    print('3. geometric screen on separated fragments:')
    ok3 = True
    for rc in (10.0, 6.0):
        sig = self_energy_matrix_imaginary_time(X_ao, D, W, mf.mo_coeff, eps,
                                                nocc, tau, om, out, mu=mu,
                                                block_memory_gb=1e-4,
                                                coords=coords, screen_r_cut=rc)
        rel = np.max(np.abs(sig - ref)) / np.max(np.abs(ref))
        frac = (sep - rad[:, None] - rad[None, :] > rc).mean()
        ok = rel < 1e-13 and frac > 0.2
        ok3 &= ok
        print(f'   r_cut={rc:5.1f} Bohr  skipped {100 * frac:5.1f}%  rel={rel:.2e}'
              f'  {"OK" if ok else "FAIL"}')
    all_ok &= ok3

    # 4. And it has to survive the trip from the driver, not just the kernel.
    base = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1,
                                      factors=(X_mo, D, X_ao, coords)) * HARTREE_TO_EV
    scr = solve_qp_energy_space_time(mf, mol, nocc, nocc - 1,
                                     factors=(X_mo, D, X_ao, coords),
                                     screen_r_cut=6.0) * HARTREE_TO_EV
    ok4 = abs(base - scr) * 1e3 < 1e-3
    all_ok &= ok4
    print(f'4. driver passes screen_r_cut through: HOMO {base:.6f} vs {scr:.6f} eV'
          f'  d={abs(base - scr) * 1e3:.5f} meV  {"OK" if ok4 else "FAIL"}')

    # 5. The whole blocked, streamed, Morton-ordered kernel against the four-line
    #    statement of the same physics: Zt = D Wt D^T, Hadamard against G^{<,>},
    #    contract with the collocation. `self_energy_imaginary_time` is that
    #    form. It stops in the time domain and works in whatever basis X is in,
    #    so the comparison applies the tau -> omega recombination by hand and
    #    rotates the production result AO -> MO. Checks 2 and 3 only compare the
    #    kernel against ITSELF under a regrouping; a sign or an index error
    #    common to both sides survives them and dies here.
    Ctw, _ = minimax_transform_weights(COSINE_WT, tau, om, *rW)
    sig_l, sig_g = self_energy_imaginary_time(X_mo, D, _transform_screened(Ctw, W),
                                              eps, nocc, tau, mu=mu)
    Cs, _ = minimax_transform_weights(COSINE_TW, tau, out, *rS)
    Ss, _ = minimax_transform_weights(SINE_TW, tau, out, *rS)
    naive = -0.5 * (np.tensordot(Cs, sig_g + sig_l, axes=(1, 0))
                    + 1j * np.tensordot(Ss, sig_g - sig_l, axes=(1, 0)))
    rel = np.max(np.abs(sigma_ao_to_mo(ref, mf.mo_coeff) - naive)) / np.max(np.abs(naive))
    ok5 = rel < 1e-12
    all_ok &= ok5
    print(f'5. production kernel == naive reference form: rel={rel:.2e}  '
          f'{"OK" if ok5 else "FAIL"}')

    # 6. Sigma_mu,nu = Sigma_nu,mu. Structural rather than grid-limited: Zt and
    #    Ghat are both symmetric and the outer contraction uses the same X on
    #    both sides. A transposed operand would leave checks 2 and 3 untouched,
    #    since it would sit on both sides of those comparisons.
    d_sym = np.max(np.abs(ref - ref.transpose(0, 2, 1))) / np.max(np.abs(ref))
    ok6 = d_sym < 1e-13
    all_ok &= ok6
    print(f'6. mu<->nu symmetry: rel={d_sym:.2e}  {"OK" if ok6 else "FAIL"}')

    # 7. Sigma(mu - i.w) = conj Sigma(mu + i.w). A caller can take the occupied
    #    half of the axis by conjugation instead of running a second tau sweep,
    #    so the symmetry is pinned here rather than left implicit there.
    neg = self_energy_matrix_imaginary_time(X_ao, D, W, mf.mo_coeff, eps, nocc,
                                            tau, om, -out, mu=mu,
                                            block_memory_gb=1e6)
    d_conj = np.max(np.abs(neg - np.conj(ref))) / np.max(np.abs(ref))
    ok7 = d_conj < 1e-13
    all_ok &= ok7
    print(f'7. Sigma(-i.w) == conj Sigma(+i.w): rel={d_conj:.2e}  '
          f'{"OK" if ok7 else "FAIL"}')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

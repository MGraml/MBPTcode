"""Folded EE-ADC solver (ee_fold) and the GF2 block order.

The doubles block of the EE-ADC operator is exactly diagonal at block order o_dd = 0
(levels adc2 and gf2), so the doubles can be eliminated without approximation and the
singles problem A_eff(w) y = w y, A_eff(w) = M - V^T (D - w)^-1 V, solved at each
root's own w. Checks, on water / cc-pVDZ (RHF, DF factors) unless stated:

  1. build_operator(pieces=True) hands out M, V, Vt, D that rebuild aop(vec) on a
     random vector to 1e-12, at adc2; pieces=True raises ValueError at adc1, adc2x,
     adc3.
  2. the gf2 block order (1, 1, 0): its M equals adc1's singles block, its V and Vt
     equal adc2's on a random vector, its D equals d_ijab, its dimensions equal
     adc2's.
  3. folded_operator at the full solve's root w reproduces that root on the singles part
     of the full eigenvector (residual < 1e-8); the dense path equals the matrix-free
     one; wrong length and complex dtype raise ValueError.
  4. solve_folded at adc2 reproduces the full spin-free ADC(2) Davidson roots, three
     singlets and three triplets, to 1e-5 eV, with T1 against the full eigenvector's
     singles weight to 1e-4, every root under eight Newton steps; t_min=1.0 forces the
     fixed-point loop to the same roots; H2/STO-3G triplets warn on a short channel
     and the one root equals the full solve.
  5. CH4 / cc-pVDZ (not NH3: the rounded C3v geometry splits the E pair by 2e-4 eV):
     the degenerate T2 singlet triple is followed without a skip, its partners
     orthonormal, on the dense and on the Davidson branch.
  6. the Davidson branch equals the dense one on water; spin=None equals the full
     solve over both channels.
  7. gf2: the fold equals the full gf2 channel solve; the dense integral route
     equals the DF route with exact factors.
  8. pieces=True with parity and en_dress at gf2 raise ValueError; an exhausted
     step budget reports converged False with a warning; no input is mutated.

Run: python tests/test_ee_adc_fold.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_r_sigma, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

HARTREE_TO_EV = 27.211386245988
WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water_df(basis='cc-pvdz'):
    """(mf, eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return mf, eps, B, mol.nelectron // 2


def check_pieces(eps, B, no):
    ok = True
    aop, diag, dims, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2',
                                                      pieces=True)
    nv = len(eps) - no
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(dims['nH'])
    y1, Y = ee_r_sigma.to_blocks(vec, no, nv, 'adc2')
    be = P['be']
    # the fold's reading of the operator: [M y + Vt Y ; V y + D * Y]
    w1 = be.ein('iajb,jb->ia', P['M'], y1) + P['Vt'](Y)
    W = P['V'](y1) + be.scale(Y, P['D'])
    rebuilt = ee_r_sigma.from_blocks(w1, W, no, nv, 'adc2')
    err = float(np.max(np.abs(rebuilt - aop(vec))))
    ok &= check(err < 1e-12, 'pieces rebuild aop(vec) at adc2', f'max |diff| {err:.1e}')
    ok &= check(P['D'].shape == (no, no, nv, nv), 'D has shape (no, no, nv, nv)')
    for level in ('adc1', 'adc2x', 'adc3'):
        try:
            ee_r_sigma_df.build_operator(eps, B, no, level=level, pieces=True)
            ok &= check(False, f'pieces=True raises at {level}')
        except ValueError as exc:
            ok &= check(level in str(exc), f'pieces=True raises at {level}',
                        str(exc)[:60])
    return ok


def check_gf2(eps, B, no):
    """eq 53 of Monino-Loos 2023: ADC(2) with the singles block cut back to A^HF."""
    ok = True
    nv = len(eps) - no
    _, _, d_gf2, P2 = ee_r_sigma_df.build_operator(eps, B, no, level='gf2',
                                                   pieces=True)
    _, _, d_adc2, P1 = ee_r_sigma_df.build_operator(eps, B, no, level='adc2',
                                                    pieces=True)
    aop1, _, _ = ee_r_sigma_df.build_operator(eps, B, no, level='adc1')
    ok &= check(d_gf2 == d_adc2, 'gf2 dimensions equal adc2')
    ok &= check(ee_r_sigma._BLOCK_ORDERS['gf2'] == (1, 1, 0),
                'gf2 block order (1, 1, 0)')
    be = P2['be']
    rng = np.random.default_rng(11)
    vec = rng.standard_normal(d_adc2['nH'])
    y1, Y = ee_r_sigma.to_blocks(vec, no, nv, 'adc2')
    # singles block: gf2's M on y1 equals the adc1 operator on the singles part
    w_gf2 = ee_r_sigma.from_blocks(be.ein('iajb,jb->ia', P2['M'], y1), SB(), no, nv,
                                   'adc1')
    w_adc1 = aop1(vec[:2 * no * nv])
    err = float(np.max(np.abs(w_gf2 - w_adc1)))
    ok &= check(err < 1e-12, 'gf2 singles block equals adc1 (A^HF)', f'{err:.1e}')
    # couplings and doubles diagonal: identical to adc2's
    ok &= check(_sb_close(P2['V'](y1), P1['V'](y1)), 'gf2 coupling V equals adc2')
    ok &= check(_sb_close(P2['Vt'](Y), P1['Vt'](Y)), 'gf2 coupling Vt equals adc2')
    ok &= check(np.allclose(P2['D'], P1['D'], atol=0, rtol=0), 'gf2 D equals d_ijab')
    return ok


def check_folded_operator(mf, eps, B, no):
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    nv = len(eps) - no
    n_s = 2 * no * nv
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    # the full solve's lowest singlet: Davidson, since the channel (6365) is far
    # above DENSE_LIMIT; its residual 1e-10 sits well under every tolerance here
    e_full, Z = solve_ee_adc(mf, level='adc2', nroots=1, df=True, spin='singlet',
                             conv_tol=1e-10)
    w = float(e_full[0])
    matvec, dmatvec, diag_s, embed, restrict = ee_fold.folded_operator(P, w,
                                                                       spin='singlet')
    # singles part of the full eigenvector (solve_ee_adc returns it embedded),
    # in the channel basis
    z = np.asarray(Z[:, 0])
    y = restrict(z[:n_s])
    y = y / np.linalg.norm(y)
    res = float(np.linalg.norm(matvec(y) - w * y))
    ok &= check(res < 1e-8, 'A_eff(w) y = w y on the full root', f'residual {res:.1e}')
    # dense build equals the matrix-free action
    n = diag_s.size
    A = ee_fold.dense_effective(matvec, n)
    err = float(np.max(np.abs(A @ y - matvec(y))))
    ok &= check(err < 1e-12, 'dense_effective equals matvec', f'{err:.1e}')
    ok &= check(np.allclose(A, A.T, atol=1e-10), 'A_eff(w) is symmetric')
    # omega=None is the bare singles block
    m0, _, _, _, _ = ee_fold.folded_operator(P, None, spin='singlet')
    err0 = float(np.linalg.norm(m0(y) - restrict(
        ee_fold.singles_sb_to_flat(P['be'].ein('iajb,jb->ia', P['M'],
                                               ee_fold.singles_flat_to_sb(
                                                   embed(y), no, nv)), no, nv))))
    ok &= check(err0 < 1e-12, 'omega=None gives M alone', f'{err0:.1e}')
    # boundary
    for bad, label in ((np.ones(n + 1), 'wrong length'),
                       (np.ones(n, dtype=complex), 'complex dtype')):
        try:
            matvec(bad)
            ok &= check(False, f'{label} raises ValueError')
        except ValueError:
            ok &= check(True, f'{label} raises ValueError')
    return ok


def check_solve_folded(mf, eps, B, no):
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    nv = len(eps) - no
    n_s = 2 * no * nv
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    for spin in ('singlet', 'triplet'):
        e_full, Z = solve_ee_adc(mf, level='adc2', nroots=3, df=True, spin=spin,
                                 conv_tol=1e-10)
        res = ee_fold.solve_folded(P, 3, spin=spin)
        for r in range(3):
            d = abs(res.omega[r] - e_full[r]) * ev
            ok &= check(d < 1e-5, f'{spin} root {r}: folded equals full ADC(2)',
                        f'{res.omega[r] * ev:.6f} vs {e_full[r] * ev:.6f} eV, '
                        f'|d| {d:.1e} eV')
            z = np.asarray(Z[:, r])
            t1_full = float(z[:n_s] @ z[:n_s]) / float(z @ z)
            ok &= check(abs(res.t1[r] - t1_full) < 1e-4, f'{spin} root {r}: T1',
                        f'{res.t1[r]:.5f} vs {t1_full:.5f}')
            newton = (res.converged[r] and res.steps[r] <= 8
                      and res.loop[r] == 'newton')
            ok &= check(newton, f'{spin} root {r}: Newton under eight steps',
                        f'{res.steps[r]} steps, loop {res.loop[r]}')
        # the fixed-point loop reaches the same roots
        res_fp = ee_fold.solve_folded(P, 3, spin=spin, t_min=1.0)
        d = float(np.max(np.abs(res_fp.omega - e_full[:3]))) * ev
        ok &= check(d < 1e-5 and all(lp == 'fixed' for lp in res_fp.loop),
                    f'{spin}: fixed-point loop reaches the same roots',
                    f'|d| {d:.1e} eV')
    # short channel: H2/STO-3G has one triplet single
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf2 = scf.RHF(mol).density_fit()
    mf2.conv_tol = 1e-12
    mf2.kernel()
    eps2 = np.asarray(get_orbital_energies(mf2, representation='spatial'), float)
    B2 = DFIntegrals.from_scf(mol, mf2).B_aa
    _, _, _, P2 = ee_r_sigma_df.build_operator(eps2, B2, 1, level='adc2',
                                               pieces=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res2 = ee_fold.solve_folded(P2, 3, spin='triplet')
    fewer = any('fewer' in str(w.message) for w in caught)
    ok &= check(res2.omega.size == 1 and fewer,
                'short channel warns and returns what it has',
                f'{res2.omega.size} root(s), {len(caught)} warning(s)')
    # nocc = nvirt = 1: every reshape of the layout is degenerate. HeH+, not H2,
    # whose g/u symmetry cuts the one single from the one double
    mol3 = gto.M(atom='He 0 0 0; H 0 0 0.77', basis='sto-3g', charge=1, verbose=0)
    mf3 = scf.RHF(mol3).density_fit()
    mf3.conv_tol = 1e-12
    mf3.kernel()
    eps3 = np.asarray(get_orbital_energies(mf3, representation='spatial'), float)
    B3 = DFIntegrals.from_scf(mol3, mf3).B_aa
    _, _, _, P3 = ee_r_sigma_df.build_operator(eps3, B3, 1, level='adc2',
                                               pieces=True)
    res3 = ee_fold.solve_folded(P3, 1, spin='singlet')
    e3, _ = solve_ee_adc(mf3, level='adc2', nroots=1, df=True, spin='singlet',
                         conv_tol=1e-10)
    d = abs(float(res3.omega[0]) - float(e3[0])) * ev
    ok &= check(d < 1e-5 and res3.t1[0] < 1.0,
                'HeH+ singlet (nocc = nvirt = 1): folded equals the full solve',
                f'|d| {d:.1e} eV, T1 {res3.t1[0]:.6f}')
    return ok


# exact Td: every coordinate is +-0.6276, so the T2 triple is degenerate to
# machine precision (the NH3 of the plan, rounded, splits its E pair by 2e-4 eV)
CH4 = ('C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; '
       'H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276')


def check_degenerate_set():
    """CH4 / cc-pVDZ: the degenerate T2 singlet triple is followed without a
    skip, each partner its own vector."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    mol = gto.M(atom=CH4, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    no = mol.nelectron // 2
    e_full, _ = solve_ee_adc(mf, level='adc2', nroots=4, df=True, spin='singlet',
                             conv_tol=1e-10)
    e_full = np.asarray(e_full)
    gaps = np.abs(np.diff(e_full)) * ev
    ok &= check(gaps[0] < 1e-6 and gaps[1] < 1e-6,
                'a degenerate singlet triple among the four lowest',
                f'gaps {gaps[0]:.1e}, {gaps[1]:.1e} eV')
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    res = ee_fold.solve_folded(P, 4, spin='singlet')
    d = np.abs(res.omega - e_full) * ev
    ok &= check(float(d.max()) < 1e-5, 'all four folded roots equal the full solve',
                f'max |d| {d.max():.1e} eV')
    for dense_limit, route in ((2000, 'dense'), (0, 'Davidson')):
        res = ee_fold.solve_folded(P, 4, spin='singlet', dense_limit=dense_limit)
        d = np.abs(res.omega - e_full) * ev
        ok &= check(float(d.max()) < 1e-5,
                    f'{route}: all four folded roots equal the full solve',
                    f'max |d| {d.max():.1e} eV')
        # within the triple; distinct roots sit at different w, so their singles
        # parts need not be orthogonal
        S = res.y[:, :3].T @ res.y[:, :3]
        ov = float(np.max(np.abs(S - np.eye(3))))
        ok &= check(ov < 1e-10, f'{route}: the triple partners are orthonormal',
                    f'max |y^T y - 1| {ov:.1e}')
    return ok


def check_routes_and_channels(mf, eps, B, no):
    """The Davidson branch of the fold (dense_limit=0) against the dense one, and
    spin=None against the full solve over both channels."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    for spin in ('singlet', 'triplet'):
        a = ee_fold.solve_folded(P, 3, spin=spin)
        b = ee_fold.solve_folded(P, 3, spin=spin, dense_limit=0)
        d = float(np.max(np.abs(a.omega - b.omega))) * ev
        ok &= check(d < 1e-7 and b.converged.all(),
                    f'{spin}: Davidson branch equals the dense branch',
                    f'|d| {d:.1e} eV')
    e_full, _ = solve_ee_adc(mf, level='adc2', nroots=6, df=True, conv_tol=1e-10)
    res = ee_fold.solve_folded(P, 6, spin=None)
    d = float(np.max(np.abs(res.omega - np.asarray(e_full)))) * ev
    ok &= check(d < 1e-5, 'spin=None: six folded roots equal the full solve',
                f'|d| {d:.1e} eV')
    return ok


def check_gf2_solves(mf, eps, B, no):
    """gf2 end to end: the fold against the full gf2 channel solve, and the dense
    integral route against the DF route with exact factors."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    ev = HARTREE_TO_EV
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)
    for spin in ('singlet', 'triplet'):
        e_full, _ = solve_ee_adc(mf, level='gf2', nroots=3, df=True, spin=spin,
                                 conv_tol=1e-10)
        res = ee_fold.solve_folded(P, 3, spin=spin)
        d = float(np.max(np.abs(res.omega - np.asarray(e_full)))) * ev
        ok &= check(d < 1e-5 and res.converged.all(),
                    f'gf2 {spin}: folded equals the full solve', f'|d| {d:.1e} eV')
    mol = mf.mol
    mf0 = scf.RHF(mol)
    mf0.conv_tol = 1e-12
    mf0.kernel()
    e_dense, _ = solve_ee_adc(mf0, level='gf2', nroots=3, df=False, conv_tol=1e-10)
    e_exact, _ = solve_ee_adc(mf0, level='gf2', nroots=3, df=True, auxbasis='exact',
                              conv_tol=1e-10)
    d = float(np.max(np.abs(np.asarray(e_dense) - np.asarray(e_exact)))) * ev
    ok &= check(d < 1e-6, 'gf2: dense integral route equals DF with exact factors',
                f'|d| {d:.1e} eV')
    return ok


def check_boundaries(mf, eps, B, no):
    """The refusals and the step budget; inputs left as they were handed in."""
    from src.SingleReference.ADC.eeADC import ee_fold
    from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
    ok = True
    try:
        ee_r_sigma_df.build_operator(eps, B, no, level='adc2', parity=1.0,
                                     pieces=True)
        ok &= check(False, 'pieces=True with parity raises ValueError')
    except ValueError as exc:
        ok &= check('parity' in str(exc), 'pieces=True with parity raises ValueError')
    try:
        solve_ee_adc(mf, level='gf2', nroots=1, df=True, en_dress=True)
        ok &= check(False, 'en_dress at gf2 raises ValueError')
    except ValueError as exc:
        ok &= check('en_dress' in str(exc), 'en_dress at gf2 raises ValueError',
                    str(exc)[:60])
    _, _, _, P = ee_r_sigma_df.build_operator(eps, B, no, level='adc2', pieces=True)
    D0, M0 = P['D'].copy(), P['M'].get('aaaa').copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = ee_fold.solve_folded(P, 1, spin='singlet', max_newton=1)
    budget = any('not converged' in str(w.message) for w in caught)
    ok &= check(not res.converged[0] and budget,
                'an exhausted step budget reports converged False and warns',
                f'converged {res.converged[0]}, steps {res.steps[0]}')
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    same = (np.array_equal(u, u0) and np.array_equal(P['D'], D0)
            and np.array_equal(P['M'].get('aaaa'), M0))
    ok &= check(same, 'matvec and solve_folded leave their inputs untouched')
    return ok


def check_diis_step():
    """The fixed-point extrapolation is well posed at every history length: a
    1e-14 relative change of the residuals moves it by rounding only, and on a
    residual linear in w it lands on the root."""
    from src.SingleReference.ADC.eeADC.ee_fold import _diis_step
    ok = True
    w_star = 0.3
    om = np.array([0.40, 0.36, 0.33, 0.315, 0.308, 0.304])
    for m in range(2, 7):
        o = om[:m]
        e = (w_star - o) + 0.3 * (w_star - o) ** 2        # a curved residual
        a = _diis_step(list(o), list(e))
        b = _diis_step(list(o), list(e * (1.0 + 1e-14 * np.arange(1, m + 1))))
        lin = _diis_step(list(o), list(0.7 * (w_star - o)))
        ok &= check(abs(a - b) < 1e-10 and abs(lin - w_star) < 1e-12,
                    f'fixed-point step well posed at history {m}',
                    f'|d| perturbed {abs(a - b):.1e}, linear {abs(lin - w_star):.1e}')
    return ok


def _sb_close(A, Bk, tol=1e-12):
    keys = set(A.keys()) | set(Bk.keys())
    return all(np.allclose(A.get(k) if A.get(k) is not None else 0.0,
                           Bk.get(k) if Bk.get(k) is not None else 0.0,
                           atol=tol, rtol=0) for k in keys)


def main():
    all_ok = True
    mf, eps, B, no = water_df()
    all_ok &= check_pieces(eps, B, no)
    all_ok &= check_gf2(eps, B, no)
    all_ok &= check_folded_operator(mf, eps, B, no)
    all_ok &= check_solve_folded(mf, eps, B, no)
    all_ok &= check_degenerate_set()
    all_ok &= check_routes_and_channels(mf, eps, B, no)
    all_ok &= check_gf2_solves(mf, eps, B, no)
    all_ok &= check_boundaries(mf, eps, B, no)
    all_ok &= check_diis_step()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

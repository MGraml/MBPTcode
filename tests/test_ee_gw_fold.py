"""Eq 66 of Monino and Loos 2023 (BSE@GW, one doubles set) as fold level 'gw'.

Checks, water (RHF, DF factors), basis per check:

  1. build_pieces_gw: M is the gf2 builder's A^HF plus eq 70 on the aaaa and bbbb
     blocks only; D has shape (no, nv, nm); Vt is V's transpose.
  2. fold against unfolded (6-31G, TDA and RPA screening): every folded root, singlet,
     triplet and spin=None, is an eigenvalue of the dense spin-orbital supermatrix
     [[M, V^T], [V, D]] assembled from the same pieces, to 3e-10 Ha (1e-8 eV), and its
     T1 is that eigenvector's singles weight to 1e-8.
  3. the fold and the builder leave D, W and the input vector untouched; an unknown
     screening raises ValueError.
  4. eq 66 transcribed from QuAcK's loops (RGW_phBSE_upfolded_sym.f90 86-216) on
     CasidaSolver's modes, both screenings: M per channel and the full spectrum to
     1e-10 Ha; this pins sqrt(2), 1/2, the signs, the spin, and QPqb's normalisation.

Run: python tests/test_ee_gw_fold.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_fold, ee_gw_pieces, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water(basis='6-31g', exact=False):
    """(mf, eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf, exact=exact).B_aa
    return mf, eps, B, mol.nelectron // 2


def dense_supermatrix(P):
    """[[M, V^T], [V, D]] over the full spin-orbital singles (aa then bb) and the
    doubles (the aa then the bb block of (k, c, m)), from the pieces by unit
    vectors. Returns (H, ns), ns the number of singles."""
    no, nv = P['no'], P['nv']
    m0, _, diag_s, _, _ = ee_fold.folded_operator(P, None, spin=None)
    ns = diag_s.size
    M = ee_fold.dense_effective(m0, ns)
    cols = []
    for k in range(ns):
        Y = P['V'](ee_fold.singles_flat_to_sb(np.eye(ns)[:, k], no, nv))
        cols.append(np.concatenate([Y.get('aa').ravel(), Y.get('bb').ravel()]))
    V = np.column_stack(cols)
    D = np.concatenate([P['D'].ravel(), P['D'].ravel()])
    return np.block([[M, V.T], [V, np.diag(D)]]), ns


def check_pieces(eps, B, no):
    """M = A^HF + eq 70 on aaaa and bbbb only; D's shape; Vt is V's transpose."""
    ok = True
    nv = len(eps) - no
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    _, _, _, gf2 = ee_r_sigma_df.build_operator(eps, B, no, level='gf2', pieces=True)
    Abar = ee_gw_pieces.static_term(eps, P['W'], P['omega'], no)
    d_same = max(float(np.max(np.abs(P['M'].get(k) - gf2['M'].get(k) - Abar)))
                 for k in ('aaaa', 'bbbb'))
    d_cross = max(float(np.max(np.abs(P['M'].get(k) - gf2['M'].get(k))))
                  for k in ('aabb', 'bbaa'))
    ok &= check(d_same < 1e-14 and d_cross < 1e-14,
                'M = A^HF + eq 70 on the same-spin blocks, A^HF across',
                f'{d_same:.1e}, {d_cross:.1e}')
    nm = P['omega'].size
    ok &= check(P['D'].shape == (no, nv, nm) and P['level'] == 'gw',
                "D has shape (no, nv, nm); level 'gw'", str(P['D'].shape))
    H, ns = dense_supermatrix(P)
    rng = np.random.default_rng(7)
    Yr = SB({s: rng.standard_normal(P['D'].shape) for s in ('aa', 'bb')})
    wt = ee_fold.singles_sb_to_flat(P['Vt'](Yr), no, nv)
    yf = np.concatenate([Yr.get('aa').ravel(), Yr.get('bb').ravel()])
    d = float(np.max(np.abs(wt - H[:ns, ns:] @ yf)))
    ok &= check(d < 1e-12, 'Vt is the transpose of V', f'{d:.1e}')
    return ok


def check_fold_vs_unfolded(eps, B, no):
    """Every folded root is an eigenpair of the dense supermatrix of its pieces."""
    ok = True
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        H, ns = dense_supermatrix(P)
        w, v = np.linalg.eigh(H)
        for spin in ('singlet', 'triplet', None):
            res = ee_fold.solve_folded(P, 3, spin=spin, tol_omega=1e-12,
                                       tol_residual=1e-10)
            dw, dt = [], []
            for r in range(res.omega.size):
                k = int(np.argmin(np.abs(w - res.omega[r])))
                dw.append(abs(w[k] - res.omega[r]))
                dt.append(abs(float(v[:ns, k] @ v[:ns, k]) - res.t1[r]))
            ok &= check(max(dw) < 3e-10 and max(dt) < 1e-8
                        and bool(res.converged.all()),
                        f'{screening} {spin}: fold roots and T1 are unfolded '
                        'eigenpairs', f'max |dw| {max(dw):.1e} Ha, '
                        f'max |dT1| {max(dt):.1e}')
    return ok


def check_boundaries(eps, B, no):
    """Inputs untouched; an unknown screening refused."""
    ok = True
    P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening='tda')
    D0, W0 = P['D'].copy(), P['W'].copy()
    matvec, _, diag_s, _, _ = ee_fold.folded_operator(P, 0.3, spin='singlet')
    u = np.linspace(-1.0, 1.0, diag_s.size)
    u0 = u.copy()
    matvec(u)
    ee_fold.solve_folded(P, 1, spin='singlet')
    ok &= check(np.array_equal(u, u0) and np.array_equal(P['D'], D0)
                and np.array_equal(P['W'], W0),
                'matvec and solve_folded leave u, D and W untouched')
    try:
        ee_gw_pieces.build_pieces_gw(eps, B, no, screening='bogus')
        ok &= check(False, 'an unknown screening raises ValueError')
    except ValueError:
        ok &= check(True, 'an unknown screening raises ValueError')
    return ok


def quack_supermatrix(eps, B, no, spin, screening):
    """Eq 66 spin-adapted, transcribed from QuAcK's RGW_phBSE_upfolded_sym.f90
    (lines 86-216) and RGW_excitation_density.f90 (lines 40-50), on CasidaSolver's
    modes renormalised to (X - Y)^T (X + Y) = 1. QuAcK's ERI(p,q,r,s) is
    <pq|rs> = (pr|qs); eri here is (pq|rs)."""
    from src.SingleReference.LinearResponse.casida import CasidaSolver
    norb = len(eps)
    nv = norb - no
    o, v = slice(0, no), slice(no, norb)
    eri = np.einsum('Qpq,Qrs->pqrs', B, B, optimize=True)
    eo, ev = eps[o], eps[v]
    ns = no * nv
    # phRLR on eHF, singlet direct RPA: A = de + 2 (ia|jb), B = 2 (ia|jb) or 0
    iajb = eri[o, v, o, v].reshape(ns, ns)
    A = np.diag((ev[None, :] - eo[:, None]).ravel()) + 2.0 * iajb
    Bw = np.zeros_like(A) if screening == 'tda' else 2.0 * iajb
    Om, X, Y = CasidaSolver(A, Bw).solve(tda=(screening == 'tda'))
    XpY = (X + Y) / np.sqrt(np.einsum('Im,Im->m', X - Y, X + Y))[None, :]
    nm = Om.size
    # rho(p,q,m) = sum_jb ERI(p,j,q,b) XpY(jb,m) = sum_jb (pq|jb) XpY(jb,m)
    rho = np.einsum('pqI,Im->pqm', eri[:, :, o, v].reshape(norb, norb, ns), XpY,
                    optimize=True)
    H = np.zeros((ns + ns * nm, ns + ns * nm))
    kappa = 2.0 if spin == 'singlet' else 0.0
    for i in range(no):
        for a in range(nv):
            ia = i * nv + a
            for j in range(no):
                for b in range(nv):
                    jb = j * nv + b
                    # (eHF(a) - eHF(i)) d_ij d_ab + kappa ERI(i,b,a,j) - ERI(i,b,j,a)
                    h = kappa * eri[i, no + a, j, no + b] - eri[i, j, no + a, no + b]
                    if i == j and a == b:
                        h += ev[a] - eo[i]
                    if i == j:
                        # + rho(a,k,m) rho(b,k,m) [1/(e_a - e_k + Om) + 1/(e_b - ...)]
                        rr = rho[no + a, o, :] * rho[no + b, o, :]
                        h += np.sum(rr / (ev[a] - eo[:, None] + Om[None, :])
                                    + rr / (ev[b] - eo[:, None] + Om[None, :]))
                    if a == b:
                        # - rho(i,c,m) rho(j,c,m) [1/(e_i - e_c - Om) + 1/(e_j - ...)]
                        rr = rho[i, v, :] * rho[j, v, :]
                        h -= np.sum(rr / (eo[i] - ev[:, None] - Om[None, :])
                                    + rr / (eo[j] - ev[:, None] - Om[None, :]))
                    H[ia, jb] = h
            # Jph + Kph = - sqrt(2) d_ac rho(i,k,m) + sqrt(2) d_ik rho(a,c,m)
            blk = np.zeros((no, nv, nm))
            blk[:, a, :] -= np.sqrt(2.0) * rho[i, o, :]
            blk[i, :, :] += np.sqrt(2.0) * rho[no + a, v, :]
            H[ia, ns:] = blk.ravel()
            H[ns:, ia] = blk.ravel()
    # C2h2p(iam) = Om(m) + eHF(a) - eHF(i)
    H[ns:, ns:] = np.diag((Om[None, None, :] + ev[None, :, None]
                           - eo[:, None, None]).ravel())
    return H


def check_eq66_transcription(eps, B, no):
    """The pieces against QuAcK's loops, with modes from CasidaSolver: the singles
    block element by element per channel, and the full spectrum."""
    ok = True
    nv = len(eps) - no
    ns = no * nv
    for screening in ('tda', 'rpa'):
        P = ee_gw_pieces.build_pieces_gw(eps, B, no, screening=screening)
        Hq = {s: quack_supermatrix(eps, B, no, s, screening)
              for s in ('singlet', 'triplet')}
        Ma, Mx = P['M'].get('aaaa'), P['M'].get('aabb')
        for spin, sgn in (('singlet', 1.0), ('triplet', -1.0)):
            Mch = (Ma + sgn * Mx).reshape(ns, ns)
            d = float(np.max(np.abs(Mch - Hq[spin][:ns, :ns])))
            ok &= check(d < 1e-10, f'{screening} {spin}: M equals QuAcK singles block',
                        f'max |d| {d:.1e} Ha')
        H, _ = dense_supermatrix(P)
        ref = np.sort(np.concatenate([np.linalg.eigvalsh(Hq[s]) for s in Hq]))
        got = np.linalg.eigvalsh(H)
        d = float(np.max(np.abs(got - ref))) if got.size == ref.size else np.inf
        ok &= check(d < 1e-10, f'{screening}: the full spectrum equals QuAcK form',
                    f'max |d| {d:.1e} Ha over {ref.size}')
    return ok


def main():
    all_ok = True
    mf, eps, B, no = water('6-31g')
    all_ok &= check_pieces(eps, B, no)
    all_ok &= check_fold_vs_unfolded(eps, B, no)
    all_ok &= check_eq66_transcription(eps, B, no)
    all_ok &= check_boundaries(eps, B, no)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

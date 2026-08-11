import sys
import numpy as np
from pyscf.lib import logger
from pyscf.tdscf._lr_eig import real_eig

from src.SingleReference.base import get_occ_virt_indices


def solve_casida_davidson(lr_solver, nocc, nroots=3, polarizability='RPA',
                           W_aux=None, conv_tol=1e-5, max_cycle=100, orbsym=None):
    """Matrix-free Davidson solver for the `nroots` lowest Casida excitation energies (never forms dense A/B).

    For a handful of low-lying states; vertex-correction sums still need the
    full spectrum (use build_casida_matrices + CasidaSolver for that).
    Iteration via pyscf's real_eig; the A/B matrix-free action (apply_V/
    apply_exchange_*) is ours, validated against build_casida_matrices to
    machine precision.

    polarizability: 'RPA' (Hartree-only), 'TDHF' (bare exchange), or 'BSE'
    (screened exchange, needs W_aux from solve_rpa_screening).
    orbsym: optional pyscf orbital irrep IDs; enables symmetry-block Davidson
    (validated correct, but measured no speedup on benzene).
    Returns (omega, X, Y) normalized <X|X>-<Y|Y>=1.
    """
    mode = polarizability.upper()
    if mode == 'RPA':
        lBSE, w = False, None
    elif mode == 'TDHF':
        lBSE, w = True, None
    elif mode == 'BSE':
        if W_aux is None:
            raise ValueError("polarizability='BSE' requires W_aux (see LinearResponseSolver.solve_rpa_screening).")
        lBSE, w = True, W_aux
    else:
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose 'RPA', 'TDHF', or 'BSE'.")

    return _solve_df_davidson(lr_solver, nocc, nroots, lBSE, w, conv_tol, max_cycle, orbsym)


def _solve_df_davidson(lr_solver, nocc, nroots, lBSE, W_aux, conv_tol, max_cycle, orbsym=None):
    occ, virt = get_occ_virt_indices(lr_solver.eps, nocc)
    no, nv = len(occ), len(virt)
    n_pair = no * nv

    if orbsym is not None:
        orbsym_d2h = np.asarray(orbsym) % 10
        x_sym = (orbsym_d2h[occ][:, None] ^ orbsym_d2h[virt][None, :]).ravel()
    else:
        x_sym = None

    C_ov = lr_solver.df_coeff[:, occ[:, None], virt]
    C_oo = lr_solver.df_coeff[:, occ[:, None], occ]
    C_vv = lr_solver.df_coeff[:, virt[:, None], virt]
    diag_d = (lr_solver.eps[virt][None, :] - lr_solver.eps[occ][:, None])
    factor = 2.0

    def apply_V(z):
        t = np.einsum('Pjb,njb->nP', C_ov, z, optimize=True)
        return np.einsum('nP,Pia->nia', t, C_ov, optimize=True)

    # Precompute the W_aux contraction once; redoing it per apply_exchange_* call
    # made each Davidson iteration ~naux times more expensive.
    if W_aux is not None:
        WC_vv = np.einsum('PQ,Qab->Pab', W_aux, C_vv, optimize=True)
        WC_ov = np.einsum('PQ,Qjb->Pjb', W_aux, C_ov, optimize=True)
    else:
        WC_vv = C_vv
        WC_ov = C_ov

    def apply_exchange_direct(z):
        # Explicit 2-step contraction (a single 3-operand einsum silently costs
        # O(naux*nocc^2*nvirt^2) instead of O(naux*nocc*nvirt*max(nocc,nvirt))).
        # optimize=True matters too: ~13x slower without it for these shapes.
        tmp = np.einsum('Pab,njb->nPja', WC_vv, z, optimize=True)
        return np.einsum('Pij,nPja->nia', C_oo, tmp, optimize=True)

    def apply_exchange_swap(z):
        tmp = np.einsum('Pja,njb->nPab', WC_ov, z, optimize=True)
        return np.einsum('Pib,nPab->nia', C_ov, tmp, optimize=True)

    def apply_A(z):
        val = diag_d[None, :, :] * z + factor * apply_V(z)
        if lBSE:
            val = val - apply_exchange_direct(z)
        return val

    def apply_B(z):
        val = factor * apply_V(z)
        if lBSE:
            val = val - apply_exchange_swap(z)
        return val

    def vind(xys):
        xys = np.asarray(xys).reshape(-1, 2, no, nv)
        xs, ys = xys[:, 0], xys[:, 1]
        top = (apply_A(xs) + apply_B(ys)).reshape(xys.shape[0], -1)
        bot = (apply_B(xs) + apply_A(ys)).reshape(xys.shape[0], -1)
        return np.hstack([top, -bot])

    hdiag = np.hstack([diag_d.ravel(), -diag_d.ravel()])

    def precond(dx, e):
        e = np.atleast_1d(e)
        d = hdiag[None, :] - e[:, None]
        d[np.abs(d) < 1e-8] = 1e-8
        return (dx.reshape(len(e), -1) / d).reshape(dx.shape)

    nstates_guess = min(nroots, n_pair)
    order = np.argsort(diag_d.ravel())[:nstates_guess]
    x0 = np.zeros((nstates_guess, 2 * n_pair))
    for k, idx in enumerate(order):
        x0[k, idx] = 1.0
    x0sym = x_sym[order] if x_sym is not None else None

    converged, e, xy = real_eig(vind, x0, precond, tol_residual=conv_tol,
                                 nroots=nroots, x0sym=x0sym, max_cycle=max_cycle,
                                 verbose=logger.Logger(sys.stdout, 0))

    omega = np.asarray(e)
    X = np.zeros((n_pair, len(omega)))
    Y = np.zeros((n_pair, len(omega)))
    for k, z in enumerate(xy):
        x, y = z.reshape(2, no, nv)
        norm = np.sqrt(abs(np.sum(x**2) - np.sum(y**2)))
        X[:, k] = (x / norm).ravel()
        Y[:, k] = (y / norm).ravel()
    return omega, X, Y

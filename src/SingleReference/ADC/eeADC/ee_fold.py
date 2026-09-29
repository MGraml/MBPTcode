"""Doubles fold of the EE-ADC operator: the singles problem at its own frequency.

At block order o_dd = 0 (levels adc2 and gf2) the doubles block of the EE-ADC
supermatrix is the bare diagonal D_ijab = ε_a + ε_b - ε_i - ε_j, so the doubles
can be eliminated without approximation: the supermatrix eigenproblem

    sum_jb M_ia,jb y_jb + sum_K V_K,ia Y_K = ω y_ia ,
    sum_jb V_K,jb y_jb + D_K Y_K = ω Y_K

gives Y_K = sum_jb V_K,jb y_jb / (ω - D_K) and sum_jb A_eff(ω)_ia,jb y_jb = ω y_ia,

    A_eff(ω)_ia,jb = M_ia,jb - sum_K V_K,ia V_K,jb / (D_K - ω) ,

with ia, jb the singles and K the doubles of the flat adc2 layout, in whose
metric the supermatrix is symmetric (K is a spin-resolved (k, l, c, d)).
For the eigenpair (λ(ω), y) of A_eff(ω) with sum_ia y_ia² = 1,

    dλ/dω = -sum_K (sum_jb V_K,jb y_jb)² / (D_K - ω)²,
    T1 = 1 / (1 - dλ/dω)      (the singles weight of the full eigenvector),

and self-consistency λ(ω) = ω is reached per root by Newton,

    ω_{k+1} = ω_k + (λ(ω_k) - ω_k) T1(ω_k),

with a fixed-point fallback ω_{k+1} = λ(ω_k), DIIS-accelerated, when T1 falls
under t_min (a doubles element near ω makes the derivative unreliable). This
is the shape of Turbomole's RI-CC2 solver on the ADC operator; the doubles
vector exists only inside one matvec.

Flat singles vectors here are the singles segments of ee_r_sigma's layout,
length 2 no nv (aa block then bb block); with a spin channel, the channel basis
of ee_driver._channel_basis at the singles layout (level 'adc1').

References
----------
E. Monino and P.-F. Loos, J. Chem. Phys. 159, 034105 (2023), eq 53 (gf2).
C. Hättig and F. Weigend, J. Chem. Phys. (2000), doi:10.1063/1.1290013 (the
folded-doubles solver of RI-CC2).
"""
import warnings

import numpy as np

from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.Solvers.davidson import overlap_pick, solve_symmetric

FOLD_LEVELS = ('adc2', 'gf2')
_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}
_DEGENERATE = 1e-8      # Hartree: eigenvalues closer than this are one level


def singles_flat_to_sb(u, no, nv):
    """Flat singles vector, shape (2 no nv,), aa then bb -> SB {'aa', 'bb'} of
    shape (no, nv), index order (i, a)."""
    return _r.to_blocks(np.asarray(u, float), no, nv, 'adc1')[0]


def singles_sb_to_flat(w1, no, nv):
    """SB singles {'aa', 'bb'} shape (no, nv) -> flat vector, shape (2 no nv,)."""
    return _r.from_blocks(w1, SB(), no, nv, 'adc1')


def doubles_sb_to_flat(W, no, nv):
    """SB doubles -> the doubles segment of the adc2 layout, shape
    (2 n_same + n_mixed,), in the metric the full eigenvector is normalised in."""
    n_s = no * nv
    return _r.from_blocks(SB(), W, no, nv, 'adc2')[2 * n_s:]


def _check_level(pieces):
    if pieces['level'] not in FOLD_LEVELS:
        raise ValueError(f"the fold needs a bare-diagonal doubles block; level="
                         f"{pieces['level']!r} is not one of {FOLD_LEVELS}")


def _check_vector(u, n):
    u = np.asarray(u)
    if np.iscomplexobj(u):
        raise ValueError('the fold is real symmetric; a complex vector was passed')
    u = u.ravel()
    if u.size != n:
        raise ValueError(f'singles vector of length {u.size}, expected {n}')
    return u.astype(float, copy=False)


def folded_operator(pieces, omega, spin=None):
    """A_eff(ω) on flat singles vectors, one spin channel or both.

    Parameters
    ----------
    pieces : dict
        From ``build_operator(..., pieces=True)``: 'M' (SB, blocks of shape
        (no, nv, no, nv), index order (i, a, j, b)), 'V' (SB singles, blocks
        (no, nv), (i, a) -> SB doubles, blocks (no, no, nv, nv), (i, j, a, b)),
        'Vt' (the reverse map), 'D' (ndarray, shape (no, no, nv, nv), index
        order (i, j, a, b)), 'level', 'no', 'nv', 'be'.
    omega : float or None
        Frequency in Hartree; None gives the bare singles block M.
    spin : {'singlet', 'triplet', None}
        Restrict to one eigenspace of the alpha<->beta flip.

    Returns
    -------
    matvec : callable
        u (n,) -> A_eff(ω) u (n,), n = 2 no nv or the channel's size.
    dmatvec : callable
        u (n,) -> sum_K (sum_jb V_K,jb u_jb)² / (D_K - ω)² (a float), so that
        dλ/dω = -dmatvec(y) for a unit eigenvector y; 0.0 when omega is None.
    diag_s : ndarray, shape (n,)
        Diagonal of M on the channel's representative entries (preconditioner).
    embed, restrict : callables
        Channel basis -> full singles vector and back; identities for spin=None.
    """
    _check_level(pieces)
    no, nv, be = pieces['no'], pieces['nv'], pieces['be']
    M, V, Vt, D = pieces['M'], pieces['V'], pieces['Vt'], pieces['D']
    n_full = 2 * no * nv
    if spin is None:
        def embed(u):
            return np.asarray(u, float)

        def restrict(v):
            return np.asarray(v, float)
        reps = np.arange(n_full)
    else:
        if spin not in _CHANNEL:
            raise ValueError(f"spin={spin!r}; expected 'singlet', 'triplet' or None")
        from src.SingleReference.ADC.eeADC.ee_driver import _channel_basis
        embed, restrict, reps = _channel_basis(n_full, no, nv, 'adc1',
                                               _CHANNEL[spin])
    n = reps.size
    # diag of M on the singles layout: M_ia,ia of the aa and the bb block
    m_diag = np.concatenate([np.einsum('iaia->ia', M.get('aaaa')).ravel(),
                             np.einsum('iaia->ia', M.get('bbbb')).ravel()])
    diag_s = m_diag[reps]
    denom = None if omega is None else D - omega        # (D - ω), sign kept

    def matvec(u):
        u = _check_vector(u, n)
        y1 = singles_flat_to_sb(embed(u), no, nv)
        w1 = be.ein('iajb,jb->ia', M, y1)                 # sum_jb M_ia,jb y_jb
        if denom is not None:
            # - sum_K V_K,ia (sum_jb V_K,jb y_jb) / (D_K - ω) : the folded doubles
            Y = be.divide(V(y1), denom)
            w1 = w1 - Vt(Y)
        return restrict(singles_sb_to_flat(w1, no, nv))

    def dmatvec(u):
        if denom is None:
            return 0.0
        u = _check_vector(u, n)
        y1 = singles_flat_to_sb(embed(u), no, nv)
        # Y_K = sum_jb V_K,jb y_jb / (D_K - ω), flat; then sum_K Y_K²
        Yf = doubles_sb_to_flat(be.divide(V(y1), denom), no, nv)
        return float(Yf @ Yf)

    return matvec, dmatvec, diag_s, embed, restrict


def dense_effective(matvec, n):
    """A_eff as an (n, n) array from n unit-vector matvecs, symmetrised."""
    A = np.column_stack([matvec(np.eye(n)[:, k]) for k in range(n)])
    return 0.5 * (A + A.T)


class FoldResult:
    """Per-root results of solve_folded.

    omega : ndarray, shape (nroots,), Hartree, ascending
    y : ndarray, shape (n, nroots), unit singles vectors in the channel basis
    t1 : ndarray, shape (nroots,), singles weights in (0, 1]
    steps : ndarray, shape (nroots,), outer iterations used
    loop : list of str, 'newton' or 'fixed' per root
    converged : ndarray, shape (nroots,), bool
    embed : callable, channel vector -> full flat singles vector
    """

    def __init__(self, omega, y, t1, steps, loop, converged, embed):
        self.omega, self.y, self.t1 = omega, y, t1
        self.steps, self.loop, self.converged = steps, loop, converged
        self.embed = embed


def _diis_step(hist_omega, hist_err):
    """Pulay step on the scalar history: ω_next = sum_k c_k λ_k, λ_k = ω_k + e_k,
    with the c_k of least norm under sum_k c_k = 1 and sum_k c_k e_k = 0.

    One residual per step makes the Pulay matrix e_k e_l rank one, so from three
    entries on its bordered system is singular. The least-norm weights are
    c_k = a + b e_k, which make ω_next the least-squares line λ = α + β e over
    the history, taken at e = 0: the secant step for two entries, λ for one."""
    e = np.asarray(hist_err, float)
    lam = np.asarray(hist_omega, float) + e
    de = e - e.mean()
    var = float(de @ de)                   # sum_k (e_k - ē)²
    if var == 0.0:                         # one entry, or residuals that stay put
        return float(lam[-1])
    beta = float(de @ (lam - lam.mean())) / var
    return float(lam.mean() - beta * e.mean())


def _eig_at(pieces, omega, spin, ref, nfollow, dense, tol_residual, label):
    """(λ, y, T1) of A_eff(ω): the eigenpair of maximal overlap with ref
    (ref None: the lowest), by eigh below dense_limit, else by Davidson."""
    matvec, dmatvec, diag_s, embed, restrict = folded_operator(pieces, omega, spin)
    n = diag_s.size
    if dense:
        # sum_q A_pq v_qm = w_m v_pm, A built from n matvecs
        A = dense_effective(matvec, n)
        w, v = np.linalg.eigh(A)
        if ref is None:
            lam, y = float(w[0]), v[:, 0]
        else:
            k = int(np.argmax(np.abs(ref @ v)))
            # a degenerate eigenspace has no preferred basis: follow ref's
            # projection onto it, y_p = sum_m v_pm sum_q v_qm ref_q over the m
            # with w_m = w_k, so partners seeded orthogonal stay orthogonal
            cl = np.abs(w - w[k]) < _DEGENERATE
            lam, y = float(w[k]), v[:, cl] @ (v[:, cl].T @ ref)
    else:
        # sum_q A_pq x_q = λ x_p by Davidson on the matvec, A never built
        if ref is None:
            e, X, conv = solve_symmetric(matvec, diag_s, nroots=nfollow,
                                         tol_residual=tol_residual, label=label)
            k = 0
        else:
            # overlap_pick ranks Ritz vectors by |<ref|x>|; past the followed
            # root that ranking is noise and never converges, so follow one
            e, X, conv = solve_symmetric(matvec, diag_s, nroots=1, x0=ref,
                                         pick=overlap_pick(ref),
                                         tol_residual=tol_residual, label=label)
            k = 0
        lam, y = float(e[k]), np.asarray(X[:, k], float)
    y = y / np.linalg.norm(y)
    if ref is not None:
        ov = abs(float(ref @ y))
        if ov < 0.5:
            warnings.warn(f'{label}: root crossing, overlap with the previous vector '
                          f'{ov:.3f} at omega = {omega:.6f} Ha; the picked vector is '
                          'kept', RuntimeWarning, stacklevel=3)
    t1 = 1.0 / (1.0 + dmatvec(y))
    return lam, y, t1, embed


def solve_folded(pieces, nroots, spin=None, tol_omega=1e-6, tol_residual=1e-6,
                 t_min=0.3, max_newton=12, max_fixed=30, dense_limit=2000,
                 verbose=0):
    """The nroots lowest folded roots, each at its own frequency.

    Seeds: the nroots lowest eigenpairs of the bare singles block M on the
    channel. Per root: λ, y at ω_k by eigh (n ≤ dense_limit) or Davidson with
    overlap following; then the Newton step ω_{k+1} = ω_k + (λ - ω_k) T1 while
    T1 ≥ t_min, else the DIIS-accelerated fixed point ω_{k+1} = λ. Stops at
    |λ - ω_k| < tol_omega; a root that exhausts max_newton (or max_fixed) steps
    is returned with converged False and a RuntimeWarning. The roots come back
    sorted by ω, whatever the order of their seeds, and the partners of a
    degenerate level (ω closer than tol_omega) mutually orthonormal.

    Parameters
    ----------
    pieces : dict, from ``build_operator(..., pieces=True)`` at adc2 or gf2.
    nroots : int
    spin : {'singlet', 'triplet', None}
    tol_omega, tol_residual : float, Hartree and residual norm.
    t_min : float, singles weight below which the fixed-point loop takes over.
    max_newton, max_fixed : int, step budgets per root.
    dense_limit : int, channel size at or below which A_eff is built and eigh'd.
    verbose : int, 1 prints one line per outer step.

    Returns
    -------
    FoldResult
    """
    _check_level(pieces)
    m0, _, diag_s, embed, _ = folded_operator(pieces, None, spin)
    n = diag_s.size
    dense = n <= dense_limit
    nroots = int(nroots)
    if n < nroots:
        warnings.warn(f'the {spin or "combined"} channel holds {n} states, fewer '
                      f'than nroots={nroots}; returning {n}', RuntimeWarning,
                      stacklevel=2)
        nroots = n
    nfollow = min(nroots + 2, n)
    label = f'ee-ADC fold ({spin or "both"})'
    if dense:
        # sum_q M_pq v_qr = w_r v_pr, M built from n matvecs
        w0, v0 = np.linalg.eigh(dense_effective(m0, n))
        seeds = [(float(w0[r]), v0[:, r]) for r in range(nroots)]
    else:
        # sum_q M_pq x_qr = e_r x_pr, the nroots lowest by Davidson
        e0, X0, _ = solve_symmetric(m0, diag_s, nroots=nroots,
                                    tol_residual=tol_residual, label=label + ' seeds')
        seeds = [(float(e0[r]), np.asarray(X0[:, r], float)) for r in range(nroots)]

    omega = np.empty(nroots)
    y_out = np.empty((n, nroots))
    t1_out = np.empty(nroots)
    steps = np.zeros(nroots, int)
    loop, converged = [], np.zeros(nroots, bool)
    for r, (om, y) in enumerate(seeds):
        mode, hist_o, hist_e = 'newton', [], []
        k = 0
        while True:
            lam, y, t1, _ = _eig_at(pieces, om, spin, y, nfollow, dense,
                                    tol_residual, label)
            err = lam - om
            k += 1
            if verbose:
                print(f'{label} root {r} step {k} ({mode}): omega = {om:.8f} '
                      f'lambda = {lam:.8f} T1 = {t1:.4f}', flush=True)
            if abs(err) < tol_omega:
                converged[r] = True
                break
            if mode == 'newton' and t1 < t_min:
                mode, hist_o, hist_e = 'fixed', [], []
            if mode == 'newton':
                if k >= max_newton:
                    break
                om = om + err * t1                         # Newton on λ(ω) - ω
            else:
                hist_o.append(om)
                hist_e.append(err)
                hist_o, hist_e = hist_o[-6:], hist_e[-6:]
                if k >= max_newton + max_fixed:
                    break
                om = _diis_step(hist_o, hist_e)
        if not converged[r]:
            warnings.warn(f'{label}: root {r} not converged after {k} steps '
                          f'({mode}), |lambda - omega| = {abs(err):.2e} Ha',
                          RuntimeWarning, stacklevel=2)
        omega[r], y_out[:, r], t1_out[r], steps[r] = lam, y, t1, k
        loop.append(mode)
    # the fold can reorder the seeds: a lower root of M may land above a higher
    order = np.argsort(omega, kind='stable')
    omega, y_out = omega[order], y_out[:, order]
    # a degenerate level has no preferred basis, and the Davidson follow does not
    # keep its partners orthogonal: Löwdin per level, y_pr ← sum_s y_ps (S^-1/2)_sr
    # with S_rs = sum_p y_pr y_ps. Its partners agree to tol_omega only; distinct
    # roots sit at different ω, so their singles parts need not be orthogonal and
    # are left alone
    start = 0
    for stop in range(1, nroots + 1):
        if stop < nroots and omega[stop] - omega[stop - 1] < tol_omega:
            continue
        if stop - start > 1:
            blk = y_out[:, start:stop]
            s, U = np.linalg.eigh(blk.T @ blk)
            y_out[:, start:stop] = blk @ (U / np.sqrt(s)) @ U.T
        start = stop
    return FoldResult(omega, y_out, t1_out[order], steps[order],
                      [loop[i] for i in order], converged[order], embed)

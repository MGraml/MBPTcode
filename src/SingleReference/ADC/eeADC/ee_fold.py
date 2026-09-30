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

FOLD_LEVELS = ('adc2', 'gf2', 'gw')
_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}
_DEGENERATE = 1e-8      # Hartree: seeds spread by at most this are one level
_SEED_RESIDUAL = 1e-9   # seeds of M: Ritz error |r|²/gap far below _DEGENERATE
_DUPLICATE = 0.5        # full-vector overlap above which two roots are one


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
    if pieces['level'] == 'gw' and pieces.get('dnorm2') is None:
        # the adc2 flat layout would read none of the (k, c, m) blocks
        raise ValueError("level='gw' needs pieces['dnorm2'], the norm of its "
                         "(k, c, m) doubles")


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
        order (i, j, a, b), or, at level gw, shape (no, nv, nm), index order
        (k, c, m)), 'level', 'no', 'nv', 'be'; 'dnorm2' (optional: SB doubles
        -> float sum_K Y_K², for a doubles layout other than adc2's).
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
    dnorm2 = pieces.get('dnorm2')
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
        Yt = be.divide(V(y1), denom)
        if dnorm2 is not None:
            # sum_K Y_K² in the pieces' own doubles layout
            return float(dnorm2(Yt))
        # Y_K = sum_jb V_K,jb y_jb / (D_K - ω), flat; then sum_K Y_K²
        Yf = doubles_sb_to_flat(Yt, no, nv)
        return float(Yf @ Yf)

    return matvec, dmatvec, diag_s, embed, restrict


def dense_effective(matvec, n):
    """A_eff as an (n, n) array from n unit-vector matvecs, symmetrised."""
    A = np.column_stack([matvec(np.eye(n)[:, k]) for k in range(n)])
    return 0.5 * (A + A.T)


class FoldResult:
    """Per-root results of solve_folded.

    omega : ndarray, shape (nout,), Hartree, ascending
    y : ndarray, shape (n, nout), unit singles vectors in the channel basis
    x1 : ndarray, shape (n, nout), sqrt(T1) y, the singles part of the unit full
        eigenvector; embed(x1[:, r]) is the full flat spin-orbital singles vector
    t1 : ndarray, shape (nout,), singles weights in (0, 1]
    level : ndarray, shape (nout,), int, the level of each root; the partners of a
        degenerate level share its index and its omega, and a seed level that
        A_eff splits gives each of its roots a level of its own
    steps : ndarray, shape (nout,), outer iterations used
    loop : list of str, 'newton' or 'fixed' per root
    converged : ndarray, shape (nout,), bool
    embed : callable, channel vector -> full flat singles vector

    nout >= nroots: a degenerate level cut by nroots comes back whole.
    """

    def __init__(self, omega, y, t1, steps, loop, converged, embed, level):
        self.omega, self.y, self.t1 = omega, y, t1
        self.steps, self.loop, self.converged = steps, loop, converged
        self.embed = embed
        self.level = np.asarray(level, int)
        self.x1 = y * np.sqrt(t1)[None, :]


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


def _levels(e, tol=_DEGENERATE):
    """(start, stop) runs of ascending values e, shape (m,), whose spread, max minus
    min, is at most tol: the degenerate levels, a lone value being a level of one."""
    runs, start = [], 0
    for k in range(1, len(e) + 1):
        if k == len(e) or e[k] - e[start] > tol:
            runs.append((start, k))
            start = k
    return runs


def _doubles_dot(pieces, Ya, Yb):
    """sum_K Ya_K Yb_K over SB doubles, in the metric of the full eigenvector."""
    dnorm2 = pieces.get('dnorm2')
    if dnorm2 is not None:
        # polarisation: <a, b> = (|a + b|² - |a - b|²) / 4
        return 0.25 * (float(dnorm2(Ya + Yb)) - float(dnorm2(Ya - Yb)))
    no, nv = pieces['no'], pieces['nv']
    # sum_K Ya_K Yb_K on the flat adc2 layout
    return float(doubles_sb_to_flat(Ya, no, nv) @ doubles_sb_to_flat(Yb, no, nv))


def _doubles_image(pieces, u, omega, embed):
    """Ỹ_K = sum_jb V_K,jb u_jb / (ω - D_K) of a channel vector u, SB doubles."""
    no, nv, be = pieces['no'], pieces['nv'], pieces['be']
    y1 = singles_flat_to_sb(embed(u), no, nv)
    return be.divide(pieces['V'](y1), omega - pieces['D'])


def _duplicates(omega, y, t1, Yt, dot, tol_omega):
    """Pairs (r, s), r < s, of roots that landed on one root: |ω_r - ω_s| below
    2 tol_omega (two copies of a converged root differ by up to that) and a
    full-vector overlap

        |x_r · x_s| = sqrt(T1_r T1_s) |y_r · y_s + Ỹ_r · Ỹ_s|

    above _DUPLICATE; it vanishes for distinct roots at any T1, where the singles
    overlap alone need not. omega, t1: shape (nout,); y: shape (n, nout), unit
    singles in the channel basis; Yt: the roots' Ỹ (_doubles_image), indexed by
    root, read only for roots within 2 tol_omega of another;
    dot: the doubles inner product (_doubles_dot)."""
    out = []
    for s in range(len(omega)):
        for r in range(s):
            if abs(omega[s] - omega[r]) >= 2.0 * tol_omega:
                continue
            ov = np.sqrt(t1[r] * t1[s]) * abs(float(y[:, r] @ y[:, s])
                                              + dot(Yt[r], Yt[s]))
            if ov > _DUPLICATE:
                out.append((r, s))
    return out


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


def _subspace_pick(Y):
    """A davidson1 `pick` keeping the nroots Ritz vectors of largest projection
    |Y^T x|² onto span(Y), Y (n, g) orthonormal, sorted ascending; the score is the
    same for any basis of span(Y), unlike overlap_pick's with one vector."""
    Y = np.asarray(Y, float)

    def pick(w, v, nroots, envs):
        xs = envs['xs']
        # R_ck = sum_p Y_pc (xs_k)_p, the trial vectors on span(Y)
        R = np.array([[float(Y[:, c] @ x) for x in xs] for c in range(Y.shape[1])])
        # score_m = sum_c (sum_k R_ck v_km)², Ritz vector m projected on span(Y)
        score = np.sum((R @ v) ** 2, axis=0)
        idx = np.argsort(-score)[:nroots]
        order = idx[np.argsort(w[idx])]
        return w[order], v[:, order], order

    return pick


def _eig_level(pieces, omega, spin, Y, dense, tol_residual, label):
    """The g eigenpairs of A_eff(ω) of largest projection onto span(Y), Y (n, g)
    orthonormal. Returns λ (g,), Z (n, g) the eigenvectors, ZR (n, g) = Z rotated
    onto Y by the orthogonal Procrustes R, and T1 (g,) of the columns of ZR."""
    matvec, dmatvec, diag_s, _, _ = folded_operator(pieces, omega, spin)
    n, g = Y.shape
    if dense:
        # sum_q A_pq v_qm = w_m v_pm, A built from n matvecs
        w, v = np.linalg.eigh(dense_effective(matvec, n))
        score = np.sum((Y.T @ v) ** 2, axis=0)    # sum_c (sum_p Y_pc v_pm)²
        k = np.sort(np.argsort(-score)[:g])
        lam, Z = w[k], v[:, k]
    else:
        # sum_q A_pq x_qm = λ_m x_pm for the g Ritz vectors of largest projection
        lam, Z, _ = solve_symmetric(matvec, diag_s, nroots=g, x0=Y,
                                    pick=_subspace_pick(Y),
                                    tol_residual=tol_residual, label=label)
        lam, Z = np.asarray(lam, float), np.asarray(Z, float)
    # sum_p Z_pk Y_pc = sum_j U_kj s_j Wt_jc; the orthogonal R_kc = sum_j U_kj Wt_jc
    # minimises sum_pc (sum_k Z_pk R_kc - Y_pc)², and ZR_pc = sum_k Z_pk R_kc
    U, _, Wt = np.linalg.svd(Z.T @ Y)
    ZR = Z @ (U @ Wt)
    t1 = np.array([1.0 / (1.0 + dmatvec(ZR[:, j])) for j in range(g)])
    return lam, Z, ZR, t1


def _iterate(step, om, tol_omega, t_min, max_newton, max_fixed, label, verbose):
    """λ(ω) = ω for one root or one level: step(ω) -> (λ, T1, state); Newton
    ω + (λ - ω) T1 while T1 >= t_min, else the DIIS fixed point. Returns
    (λ, T1, state, steps, loop, converged, |λ - ω|) of the last step."""
    mode, hist_o, hist_e, k = 'newton', [], [], 0
    while True:
        lam, t1, state = step(om)
        err = lam - om
        k += 1
        if verbose:
            print(f'{label} step {k} ({mode}): omega = {om:.8f} '
                  f'lambda = {lam:.8f} T1 = {t1:.4f}', flush=True)
        if abs(err) < tol_omega:
            return lam, t1, state, k, mode, True, abs(err)
        if mode == 'newton' and t1 < t_min:
            mode, hist_o, hist_e = 'fixed', [], []
        if mode == 'newton':
            if k >= max_newton:
                return lam, t1, state, k, mode, False, abs(err)
            om = om + err * t1                         # Newton on λ(ω) - ω
        else:
            hist_o.append(om)
            hist_e.append(err)
            hist_o, hist_e = hist_o[-6:], hist_e[-6:]
            if k >= max_newton + max_fixed:
                return lam, t1, state, k, mode, False, abs(err)
            om = _diis_step(hist_o, hist_e)


def _root_step(pieces, spin, y, nfollow, dense, tol_residual, label):
    """step(ω) for one root, following the previous vector."""
    ref = [y]

    def step(om):
        lam, yk, t1, _ = _eig_at(pieces, om, spin, ref[0], nfollow, dense,
                                 tol_residual, label)
        ref[0] = yk
        return lam, t1, yk

    return step


def _level_step(pieces, spin, Y, dense, tol_residual, label):
    """step(ω) for a level: the mean eigenvalue and the mean T1 of its partners."""
    ref = [Y]

    def step(om):
        lam, Z, ZR, t1 = _eig_level(pieces, om, spin, ref[0], dense, tol_residual,
                                    label)
        ref[0] = ZR
        return float(lam.mean()), float(t1.mean()), (lam, Z, ZR, t1)

    return step


def _seed_levels(m0, diag_s, n, nroots, dense, tol_residual, label):
    """Seeds of M, lowest first, e (ns,) and X (n, ns), and the levels among them
    that start below nroots; on the Davidson branch the seeds are solved to
    _SEED_RESIDUAL and extended until the last such level ends before the last seed,
    so a level cut by nroots comes back whole."""
    if dense:
        # sum_q M_pq v_qr = w_r v_pr, M built from n matvecs
        e, X = np.linalg.eigh(dense_effective(m0, n))
        return e, X, [(a, b) for a, b in _levels(e) if a < nroots]
    extra = 2
    while True:
        ns = min(nroots + extra, n)
        # sum_q M_pq x_qr = e_r x_pr, the ns lowest by Davidson
        e, X, _ = solve_symmetric(m0, diag_s, nroots=ns,
                                  tol_residual=min(tol_residual, _SEED_RESIDUAL),
                                  label=label + ' seeds')
        e, X = np.asarray(e, float), np.asarray(X, float)
        runs = [(a, b) for a, b in _levels(e) if a < nroots]
        if runs[-1][1] < ns or ns == n:
            return e, X, runs
        extra *= 2


def solve_folded(pieces, nroots, spin=None, tol_omega=1e-6, tol_residual=1e-6,
                 t_min=0.3, max_newton=12, max_fixed=30, dense_limit=2000,
                 verbose=0):
    """The nroots lowest folded roots, each level at its own frequency.

    Seeds: the lowest eigenpairs of the bare singles block M on the channel (eigh
    for n ≤ dense_limit, else Davidson to residual 1e-9), grouped into levels whose
    eigenvalues spread by at most 1e-8 Ha. A level of one is solved per root: λ, y
    at ω_k by eigh or Davidson with overlap following, then the Newton step
    ω_{k+1} = ω_k + (λ - ω_k) T1 while T1 ≥ t_min, else the DIIS-accelerated fixed
    point ω_{k+1} = λ. A level of g > 1 is solved jointly: one ω, the g eigenpairs
    of A_eff(ω) of largest projection onto the span of its partners, rotated onto
    them (orthogonal Procrustes), Newton on the mean eigenvalue with the mean T1;
    the partners come back at one ω, orthonormal, with one level index. A level
    whose g eigenvalues spread by more than 1e-8 Ha at convergence is re-solved per
    root, with a RuntimeWarning. Stops at |λ - ω| < tol_omega; a root or a level
    that exhausts max_newton (then max_fixed) steps comes back converged False with
    a RuntimeWarning. A level cut by nroots comes back whole: up to g - 1 roots
    more than asked. Two roots that land on one root (|Δω| < 2 tol_omega,
    full-vector overlap above 0.5) warn, and the later is marked converged False;
    the root it missed is not recovered.

    Limits: the vectors of two distinct roots split by δ are determined to about
    tol_residual/δ on the Davidson branch, and to the ω error times |dA/dω|/δ on
    either, so a pair closer than the ω accuracy has ill-determined vectors; a
    near-degeneracy the fold creates, absent from M, can collapse two seeds onto one
    root, which the duplicate check reports.

    Parameters
    ----------
    pieces : dict, from ``build_operator(..., pieces=True)`` at adc2 or gf2, or
        ``ee_gw_pieces.build_pieces_gw`` (level gw).
    nroots : int
    spin : {'singlet', 'triplet', None}
    tol_omega, tol_residual : float, Hartree and residual norm.
    t_min : float, singles weight below which the fixed-point loop takes over.
    max_newton, max_fixed : int, step budgets per root or level.
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
    e0, X0, runs = _seed_levels(m0, diag_s, n, nroots, dense, tol_residual, label)
    nout = runs[-1][1]
    omega, t1_out = np.empty(nout), np.empty(nout)
    y_out = np.empty((n, nout))
    level, steps = np.empty(nout, int), np.zeros(nout, int)
    loop, converged = [''] * nout, np.zeros(nout, bool)
    kw = dict(tol_omega=tol_omega, t_min=t_min, max_newton=max_newton,
              max_fixed=max_fixed, verbose=verbose)

    def solve_root(r, om, y):
        step = _root_step(pieces, spin, y, nfollow, dense, tol_residual, label)
        lam, t1, yk, k, mode, conv, err = _iterate(step, om,
                                                   label=f'{label} root {r}', **kw)
        if not conv:
            warnings.warn(f'{label}: root {r} not converged after {k} steps '
                          f'({mode}), |lambda - omega| = {err:.2e} Ha',
                          RuntimeWarning, stacklevel=3)
        omega[r], y_out[:, r], t1_out[r], steps[r] = lam, yk, t1, k
        loop[r], converged[r] = mode, conv

    nlev = 0
    for lv, (a, b) in enumerate(runs):
        level[a:b] = nlev
        nlev += 1
        if b - a == 1:
            # λ(ω) = ω on the root's own followed vector
            solve_root(a, float(e0[a]), X0[:, a])
            continue
        # λ̄(ω) = ω, λ̄ the mean of the b - a eigenvalues of largest projection
        # onto the span of the level's seeds X0[:, a:b]
        step = _level_step(pieces, spin, X0[:, a:b], dense, tol_residual, label)
        lbar, _, (lam, Z, ZR, t1), k, mode, conv, err = _iterate(
            step, float(e0[a:b].mean()), label=f'{label} level {lv}', **kw)
        if conv and lam.max() - lam.min() > _DEGENERATE:
            warnings.warn(f'{label}: level {lv} split by '
                          f'{lam.max() - lam.min():.1e} Ha in A_eff, not degenerate; '
                          're-solved per root', RuntimeWarning, stacklevel=2)
            level[a:b] = nlev - 1 + np.arange(b - a)    # no partners: one level each
            nlev += b - a - 1
            # λ_j(ω) = ω per root, each from the level's final eigenvector Z_pj
            for j, r in enumerate(range(a, b)):
                solve_root(r, float(lam[j]), Z[:, j])
            continue
        if not conv:
            warnings.warn(f'{label}: level {lv} not converged after {k} steps '
                          f'({mode}), |lambda - omega| = {err:.2e} Ha',
                          RuntimeWarning, stacklevel=2)
        omega[a:b], y_out[:, a:b], t1_out[a:b] = lbar, ZR, t1
        steps[a:b], converged[a:b] = k, conv
        loop[a:b] = [mode] * (b - a)
    # an image is a whole doubles vector: build it only for a root within
    # 2 tol_omega of another, the only pairs _duplicates compares
    near = {i for s in range(nout) for r in range(s)
            if abs(omega[s] - omega[r]) < 2.0 * tol_omega for i in (r, s)}
    Yt = {r: _doubles_image(pieces, y_out[:, r], omega[r], embed) for r in near}
    for r, s in _duplicates(omega, y_out, t1_out, Yt,
                            lambda Ya, Yb: _doubles_dot(pieces, Ya, Yb), tol_omega):
        if converged[s]:
            warnings.warn(f'{label}: roots {r} and {s} landed on one root at omega '
                          f'= {omega[r]:.8f} Ha; root {s} marked unconverged, the '
                          'root it missed is not recovered', RuntimeWarning,
                          stacklevel=2)
        converged[s] = False
    # the fold can reorder the seeds: a lower root of M may land above a higher
    order = np.argsort(omega, kind='stable')
    return FoldResult(omega[order], y_out[:, order], t1_out[order], steps[order],
                      [loop[i] for i in order], converged[order], embed,
                      level[order])

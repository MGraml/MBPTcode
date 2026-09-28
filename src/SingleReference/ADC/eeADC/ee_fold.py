"""Doubles fold of the EE-ADC operator: the singles problem at its own frequency.

At block order o_dd = 0 (levels adc2 and gf2) the doubles block of the EE-ADC
supermatrix is the bare diagonal D_ijab = ε_a + ε_b - ε_i - ε_j, so the doubles
can be eliminated without approximation,

    [ M    Vᵀ ] [ y ]       [ y ]                 Y = (ω - D)⁻¹ V y
    [ V    D  ] [ Y ]  =  ω [ Y ]      ==>        A_eff(ω) y = ω y,

    A_eff(ω) = M - Vᵀ (D - ω)⁻¹ V .

For the eigenpair (λ(ω), y) of A_eff(ω) with yᵀy = 1,

    dλ/dω = -‖(D - ω)⁻¹ V y‖²,      T1 = 1 / (1 - dλ/dω)      (singles weight),

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
import numpy as np

from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

FOLD_LEVELS = ('adc2', 'gf2')
_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}


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
        From ``build_operator(..., pieces=True)``: 'M' (SB singles block), 'V'
        and 'Vt' (coupling callables), 'D' (ndarray (no, no, nv, nv), index
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
        u (n,) -> ‖(D - ω)⁻¹ V u‖² (a float), so that dλ/dω = -dmatvec(y) for a
        unit eigenvector y; 0.0 when omega is None.
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
            # - Vt (D - ω)⁻¹ V y : the folded doubles
            Y = be.divide(V(y1), denom)
            w1 = w1 - Vt(Y)
        return restrict(singles_sb_to_flat(w1, no, nv))

    def dmatvec(u):
        if denom is None:
            return 0.0
        u = _check_vector(u, n)
        y1 = singles_flat_to_sb(embed(u), no, nv)
        Yf = doubles_sb_to_flat(be.divide(V(y1), denom), no, nv)   # (D - ω)⁻¹ V y
        return float(Yf @ Yf)

    return matvec, dmatvec, diag_s, embed, restrict


def dense_effective(matvec, n):
    """A_eff as an (n, n) array from n unit-vector matvecs, symmetrised."""
    A = np.column_stack([matvec(np.eye(n)[:, k]) for k in range(n)])
    return 0.5 * (A + A.T)

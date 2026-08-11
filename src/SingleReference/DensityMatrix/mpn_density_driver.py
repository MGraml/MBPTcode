"""Hand-written combination layer wiring generated_mpn/mpn_density_pieces.py
(generated amplitude/overlap/density-cross-term pieces) together via the
general MPn density recursion:

    T_rank^(order) = numerator_rank_order / D_rank
    E^(order)      = 1/4 sum_ijab <ij||ab> t2^(order-1)_ijab
    M^(n) = sum_{m=0}^n <Psi^(m)|a+_s a_r|Psi^(n-m)>
    N^(n) = sum_{m=1}^{n-1} <Psi^(m)|Psi^(n-m)>
    c^(0)=1, c^(n) = -sum_{k=1}^n N^(k) c^(n-k)
    Delta_gamma^(n) = sum_{k=0}^n c^(k) M^(n-k)

Analogous to how src/SingleReference/CC/restricted_solver.py wires together
generated_restricted/*.py -- this module is plain Python, not generated
output, and is safe to hand-edit.

Scope: n=2, n=3, and now n=4. n=5 needs more work (rank-5 amplitudes, letter budget).

n=4 additionally needs the E^(k) energy-correction term in the amplitude
recursion (T_rank^(order) = [numerator - sum_k E^(k) T_rank^(order-k)] / D),
which was always exactly zero for order<=3 but is nonzero starting here: T2^(3) needs -E^(2)*T2^(1), and
T1^(4) needs -E^(2)*T1^(2). compute_E implements E^(k) = 1/4 sum <ij||ab>
t2^(k-1)_ijab (the standard MPn correlation-energy formula, at whichever
order's t2 amplitude is passed in).
"""
import inspect
import re

import numpy as np

from src.SingleReference.DensityMatrix.generated_mpn import mpn_density_pieces as gen


def _to_l(t_arr, rank):
    """Convert a t{rank}-convention array (my storage: axes
    [vir_asc(rank), occ_asc(rank)], antisymmetric under any same-type pair
    swap) into the l{rank}-convention array the generator's own 'l{rank}' tag
    expects (axes [occ_asc(rank), vir_DESC(rank)]).

    Empirically confirmed (not just derived): the raw generated overlap<rank>
    terms print e.g. 'l2(i,j,b,a)' (occ ascending i,j; vir DESCENDING b,a) vs
    't2(b,a,i,j)' (vir first, occ last -- but any relative vir/occ order
    within the generic 't2' tag needs no correction, since that
    tag's numeric convention already matches this module's own t{rank}
    storage bit-for-bit -- only the 'l' tag's axis layout genuinely differs).
    Reversing the vir block (rank elements) is an odd permutation whenever
    rank//2 is odd (transposition count for reversing n elements = n//2), so
    a sign flip of (-1)**(rank//2) is needed on top of the transpose --
    verified numerically for rank=1 (no flip) and rank=2 (flip) against an
    independent 0.25*sum(t2**2)-style overlap oracle; rank=3's flip follows
    the same n//2 parity argument (1 transposition to reverse 3 elements).
    """
    axes = list(range(rank, 2 * rank)) + list(range(rank - 1, -1, -1))
    sign = (-1) ** (rank // 2)
    return sign * t_arr.transpose(*axes)


def g_vvoo_df(B, nocc):
    """<ab||ij> block (nv,nv,no,no) from a spin-orbital DF factor B
    (naux, nso, nso) -- t2_1_numerator's entire content (that generated
    function is a plain g[v,v,o,o] gather), for g-free callers. O(no^2*nv^2)
    output, the amplitude itself -- unavoidable and fine."""
    norb = B.shape[1]
    o, v = slice(0, nocc), slice(nocc, norb)
    Bvo = B[:, v][:, :, o]
    direct = np.einsum('Qai,Qbj->abij', Bvo, Bvo, optimize=True)
    exchange = np.einsum('Qaj,Qbi->abij', Bvo, Bvo, optimize=True)
    return direct - exchange


def t1_2_numerator_df(B, nocc, t2_1):
    """t1_2_numerator (generated: -0.5*<kj||bi>*t2[bakj] - 0.5*<ja||bc>*t2[bcij])
    from a spin-orbital DF factor B instead of dense g -- every intermediate
    rank<=3, never materializing the O(no*nv^3) g[o,v,v,v] block. Each of the
    two generated terms splits into direct/exchange two-step B contractions:

      term1 direct   X1[Q,a,j] = sum_kb B[Q,k,b] t2[b,a,k,j];  sum_Qj X1*B[Q,j,i]
      term1 exchange X2[Q,a,k] = sum_jb B[Q,j,b] t2[b,a,k,j];  sum_Qk X2*B[Q,k,i]
      term2 direct   Y1[Q,c,i] = sum_jb B[Q,j,b] t2[b,c,i,j];  sum_Qc B[Q,a,c]*Y1
      term2 exchange Y2[Q,b,i] = sum_jc B[Q,j,c] t2[b,c,i,j];  sum_Qb B[Q,a,b]*Y2

    Validated identical to the generated dense function on random symmetric B.
    t2_1: (nv,nv,no,no) native-convention amplitude. Returns (nv,no)."""
    norb = B.shape[1]
    o, v = slice(0, nocc), slice(nocc, norb)
    Bov = B[:, o][:, :, v]
    Boo = B[:, o][:, :, o]
    Bvv = B[:, v][:, :, v]

    X1 = np.einsum('Qkb,bakj->Qaj', Bov, t2_1, optimize=True)
    t1_dir = np.einsum('Qaj,Qji->ai', X1, Boo, optimize=True)
    X2 = np.einsum('Qjb,bakj->Qak', Bov, t2_1, optimize=True)
    t1_exc = np.einsum('Qak,Qki->ai', X2, Boo, optimize=True)
    term1 = t1_dir - t1_exc

    Y1 = np.einsum('Qjb,bcij->Qci', Bov, t2_1, optimize=True)
    t2_dir = np.einsum('Qac,Qci->ai', Bvv, Y1, optimize=True)
    Y2 = np.einsum('Qjc,bcij->Qbi', Bov, t2_1, optimize=True)
    t2_exc = np.einsum('Qab,Qbi->ai', Bvv, Y2, optimize=True)
    term2 = t2_dir - t2_exc

    return -0.5 * term1 - 0.5 * term2


def _denom(eps_o, eps_v, rank):
    """D[a,b,...,i,j,...] = sum(eps_occ) - sum(eps_vir), rank virtual axes
    (leading) then rank occupied axes (trailing) -- matches t{rank}'s own
    [vir..., occ...] storage convention."""
    no, nv = eps_o.shape[0], eps_v.shape[0]
    d = np.zeros((nv,) * rank + (no,) * rank)
    for r in range(rank):
        shape = [1] * (2 * rank)
        shape[rank + r] = no
        d = d + eps_o.reshape(shape)
    for r in range(rank):
        shape = [1] * (2 * rank)
        shape[r] = nv
        d = d - eps_v.reshape(shape)
    return d


class MPnDensityDriver:
    """Spin-orbital MPn density-matrix correction, built from the generic
    generated pieces. eps_spin/g_anti_spin/nocc_spin match
    MP2DensityMatrixSolver's constructor convention (interleaved spin-orbital
    basis, antisymmetrized <pq||rs> integrals).
    """

    def __init__(self, eps_spin, g_anti_spin, nocc_spin):
        self.eps = eps_spin
        self.g = g_anti_spin
        self.nocc = nocc_spin
        self.norb = len(eps_spin)
        self.o = slice(0, nocc_spin)
        self.v = slice(nocc_spin, self.norb)
        self.no = nocc_spin
        self.nv = self.norb - nocc_spin
        self.kd = np.eye(self.norb)

    def _args(self):
        return dict(g=self.g, kd=self.kd, o=self.o, v=self.v, nv=self.nv, no=self.no)

    def compute_t2_1(self):
        num = gen.t2_1_numerator(**self._args())
        return num / _denom(self.eps[self.o], self.eps[self.v], 2)

    def compute_t2_1_custom_denom(self, D):
        """compute_t2_1 but with an externally supplied denominator `D`
        (same shape/axis convention as _denom(eps_o, eps_v, 2): [vir,vir,occ,
        occ]) instead of the bare MP1 one -- e.g. an Epstein-Nesbet-dressed D
        from EpsteinNesbet.epstein_nesbet_denominator. Reuses the SAME
        numerator as compute_t2_1, so this is a strict denominator swap, not an
        independent amplitude construction."""
        num = gen.t2_1_numerator(**self._args())
        return num / D

    def compute_order2_amplitudes(self, t2_1=None, max_rank=4):
        """max_rank: highest T^(2) rank actually materialized (returned slots
        above it are None) -- mirrors MPnDensityDriverRestricted's parameter
        of the same name/contract. T4^(2) is O(nv^4*no^4): a caller that only
        needs t1_2/t2_2/t3_2 must pass max_rank=3: forming T4^(2)
        unconditionally costs 770 GiB+ on molecules as small as CH. Default 4
        is for callers that genuinely need it (e.g. compute_delta_gamma4)."""
        if t2_1 is None:
            t2_1 = self.compute_t2_1()
        args = self._args()
        d1 = _denom(self.eps[self.o], self.eps[self.v], 1)
        t1_2 = gen.t1_2_numerator(**args, t2_1=t2_1) / d1
        t2_2 = t3_2 = t4_2 = None
        if max_rank >= 2:
            d2 = _denom(self.eps[self.o], self.eps[self.v], 2)
            t2_2 = gen.t2_2_numerator(**args, t2_1=t2_1) / d2
        if max_rank >= 3:
            d3 = _denom(self.eps[self.o], self.eps[self.v], 3)
            t3_2 = gen.t3_2_numerator(**args, t2_1=t2_1) / d3
        if max_rank >= 4:
            d4 = _denom(self.eps[self.o], self.eps[self.v], 4)
            t4_2 = gen.t4_2_numerator(**args, t2_1=t2_1) / d4
        return t1_2, t2_2, t3_2, t4_2

    def compute_E(self, t2_order):
        """E^(k) = 1/4 sum_ijab <ij||ab> t_ij^{ab(k-1)}, k-1's t2 array
        passed in as `t2_order` -- see module docstring."""
        return 0.25 * np.einsum('ijab,abij->', self.g[self.o, self.o, self.v, self.v], t2_order, optimize=True)

    def compute_order3_amplitudes(self, t1_2, t2_2, t3_2, t4_2, t2_1, E2=None):
        """T2^(3) needs the -E^(2)*T2^(1) correction (nonzero starting here,
        see module docstring); T1^(3)/T3^(3)/T4^(3) don't (their own
        E^(2)*T_rank^(1) term vanishes since T_rank^(1)=0 for rank!=2)."""
        args = self._args()
        if E2 is None:
            E2 = self.compute_E(t2_1)
        d1 = _denom(self.eps[self.o], self.eps[self.v], 1)
        d2 = _denom(self.eps[self.o], self.eps[self.v], 2)
        d3 = _denom(self.eps[self.o], self.eps[self.v], 3)
        t1_3 = gen.t1_3_numerator(**args, t1_2=t1_2, t2_2=t2_2, t3_2=t3_2) / d1
        t2_3 = (gen.t2_3_numerator(**args, t1_2=t1_2, t2_2=t2_2, t3_2=t3_2, t4_2=t4_2) - E2 * t2_1) / d2
        t3_3 = gen.t3_3_numerator(**args, t1_2=t1_2, t2_2=t2_2, t3_2=t3_2, t4_2=t4_2) / d3
        return t1_3, t2_3, t3_3

    def compute_t1_3(self, t1_2, t2_2, t3_2):
        args = self._args()
        d1 = _denom(self.eps[self.o], self.eps[self.v], 1)
        num = gen.t1_3_numerator(**args, t1_2=t1_2, t2_2=t2_2, t3_2=t3_2)
        return num / d1

    def compute_t1_4(self, t1_3, t2_3, t3_3, t1_2, E2=None):
        """T1^(4) needs the -E^(2)*T1^(2) correction (see module docstring)."""
        args = self._args()
        if E2 is None:
            E2 = self.compute_E(self.compute_t2_1())
        d1 = _denom(self.eps[self.o], self.eps[self.v], 1)
        num = gen.t1_4_numerator(**args, t1_3=t1_3, t2_3=t2_3, t3_3=t3_3) - E2 * t1_2
        return num / d1

    def compute_delta_gamma2(self):
        args = self._args()
        t2_1 = self.compute_t2_1()
        t1_2, _, _, _ = self.compute_order2_amplitudes(t2_1, max_rank=1)

        N2 = gen.overlap2(l_amp=_to_l(t2_1, 2), t_amp=t2_1)
        c2 = -N2

        gamma_hf = {'oo': self.kd[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            m11 = getattr(gen, f'm2_{block}_11')(**args, l2=_to_l(t2_1, 2), t2=t2_1)
            m20 = getattr(gen, f'm2_{block}_20')(**args, l1=_to_l(t1_2, 1))
            m02 = getattr(gen, f'm2_{block}_02')(**args, t1=t1_2)
            M2 = m11 + m20 + m02
            blocks[block] = M2 + c2 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def compute_delta_gamma3(self, t2_1=None, t1_2=None, t2_2=None, t3_2=None, t4_2=None):
        args = self._args()
        if t2_1 is None:
            t2_1 = self.compute_t2_1()
        if t1_2 is None:
            t1_2, t2_2, t3_2, t4_2 = self.compute_order2_amplitudes(t2_1)
        t1_3 = self.compute_t1_3(t1_2, t2_2, t3_2)

        N3 = 2.0 * gen.overlap2(l_amp=_to_l(t2_1, 2), t_amp=t2_2)
        c3 = -N3

        gamma_hf = {'oo': self.kd[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            # Delta_gamma^(3) = sum_{k=0}^3 c^(k) M^(3-k) = M^(3) + c^(1)*M^(2)
            # + c^(2)*M^(1) + c^(3)*M^(0). c^(1)=0 by construction (N^(1)=0),
            # and M^(1) = <Phi0|op|Psi1> + <Psi1|op|Phi0> = 0 identically
            # (Psi1 is pure rank-2; a 1-body op can't bridge rank 0 <-> rank 2)
            # -- so only the M^(3) and c^(3)*M^(0) terms survive; NOT c^(2)*M^(2)
            # (an earlier version of this driver wrongly used c^(2)*M^(2) here,
            # which is off by O(1e-5) relative to density_matrix.py's oracle --
            # c^(2) only ever multiplies M^(1), never M^(2), in the n=3 recursion).
            m3_12 = getattr(gen, f'm3_{block}_12')(**args, l2=_to_l(t2_1, 2), t1=t1_2, t2=t2_2, t3=t3_2)
            m3_21 = getattr(gen, f'm3_{block}_21')(**args, l1=_to_l(t1_2, 1), l2=_to_l(t2_2, 2), l3=_to_l(t3_2, 3), t2=t2_1)
            m3_30 = getattr(gen, f'm3_{block}_30')(**args, l1=_to_l(t1_3, 1))
            m3_03 = getattr(gen, f'm3_{block}_03')(**args, t1=t1_3)
            M3 = m3_12 + m3_21 + m3_30 + m3_03

            blocks[block] = M3 + c3 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def compute_delta_gamma4(self):
        """Delta_gamma^(4) = sum_{k=0}^4 c^(k) M^(4-k) = M^(4) + c^(2)*M^(2)
        + c^(4)*M^(0) (c^(1)=0 and M^(1)=0 as in compute_delta_gamma3;
        c^(3)*M^(1)=0 too). Unlike n=3, c^(2)*M^(2) is genuinely nonzero
        here -- the first order where it survives (see module docstring's
        general recursion  for the
        physical adjacency argument for which M^(n)/N^(n) pieces exist)."""
        args = self._args()
        t2_1 = self.compute_t2_1()
        t1_2, t2_2, t3_2, t4_2 = self.compute_order2_amplitudes(t2_1)
        E2 = self.compute_E(t2_1)
        t1_3, t2_3, t3_3 = self.compute_order3_amplitudes(t1_2, t2_2, t3_2, t4_2, t2_1, E2=E2)
        t1_4 = self.compute_t1_4(t1_3, t2_3, t3_3, t1_2, E2=E2)

        l2_1, l1_2, l2_2, l3_2, l4_2 = _to_l(t2_1, 2), _to_l(t1_2, 1), _to_l(t2_2, 2), _to_l(t3_2, 3), _to_l(t4_2, 4)
        l1_3, l2_3, l3_3 = _to_l(t1_3, 1), _to_l(t2_3, 2), _to_l(t3_3, 3)
        l1_4 = _to_l(t1_4, 1)

        # N^(4) = 2*<Psi1|Psi3> (rank-2-matched only, Psi1 pure rank2) +
        # <Psi2|Psi2> (rank-matched sum over Psi^(2)'s own ranks 1-4).
        N4 = (2.0 * gen.overlap2(l_amp=l2_1, t_amp=t2_3)
             + gen.overlap1(l_amp=l1_2, t_amp=t1_2) + gen.overlap2(l_amp=l2_2, t_amp=t2_2)
             + gen.overlap3(l_amp=l3_2, t_amp=t3_2) + gen.overlap4(l_amp=l4_2, t_amp=t4_2))
        N2 = gen.overlap2(l_amp=l2_1, t_amp=t2_1)
        c2 = -N2
        c4 = N2 ** 2 - N4

        gamma_hf = {'oo': self.kd[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            m2_11 = getattr(gen, f'm2_{block}_11')(**args, l2=l2_1, t2=t2_1)
            m2_20 = getattr(gen, f'm2_{block}_20')(**args, l1=l1_2)
            m2_02 = getattr(gen, f'm2_{block}_02')(**args, t1=t1_2)
            M2 = m2_11 + m2_20 + m2_02

            m4_13 = getattr(gen, f'm4_{block}_13')(**args, l2=l2_1, t1=t1_3, t2=t2_3, t3=t3_3)
            m4_31 = getattr(gen, f'm4_{block}_31')(**args, l1=l1_3, l2=l2_3, l3=l3_3, t2=t2_1)
            m4_22 = getattr(gen, f'm4_{block}_22')(**args, l1=l1_2, l2=l2_2, l3=l3_2, l4=l4_2,
                                                   t1=t1_2, t2=t2_2, t3=t3_2, t4=t4_2)
            m4_40 = getattr(gen, f'm4_{block}_40')(**args, l1=l1_4)
            m4_04 = getattr(gen, f'm4_{block}_04')(**args, t1=t1_4)
            M4 = m4_13 + m4_31 + m4_22 + m4_40 + m4_04

            blocks[block] = M4 + c2 * M2 + c4 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

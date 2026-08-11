"""ADCSolverSpinBlocked: restricted (RHF, closed-shell) IP/EA-ADC(3) in the
SPIN-BLOCKED determinant basis rather than the spin-adapted CSF basis.

WHY THIS EXISTS
---------------
Determinant-wise Epstein-Nesbet (Jiang & Engel, JCP 125, 184108 (2006)) gives
the same-spin and opposite-spin doubles different denominators, which breaks
the closed-shell singlet relation

    t2_1_aaaa == t2_1_abab - t2_1_abab^T

so the EN-dressed first-order wavefunction is NOT a spin eigenfunction. The
resulting U^(2) then has components on QUARTET (S=3/2) 2h1p/2p1h
configurations, which the doublet-CSF basis of ADCSolverRestricted simply does
not contain.

ADCSolverRestricted remains the right solver for the SPIN-ADAPTED EN variant
(spin_adapted=True, the default), where a single averaged shift per channel
restores the singlet relation and keeps U^(2) inside the CSF span. This module
is for the determinant-wise variant only.

THE BASIS
---------
The alpha-Dyson sector: states reachable by removing/adding one ALPHA electron.
This is exactly one of the two identical spin blocks the spin-orbital
supermatrix factors into for a closed-shell reference, so every matrix element
here is the corresponding SPIN-ORBITAL element evaluated with spatial integrals
plus spin bookkeeping -- there is no new algebra, only spin-pattern selection.

2h1p (Sz = -1/2), two spin patterns:
    'aaa'  holes i<j both alpha, particle a alpha   -> nP_o * V configs
    'abb'  hole i alpha, hole j beta, particle a beta -> O * O * V configs
2p1h (Sz = +1/2), the mirror:
    'aaa'  particles a<b both alpha, hole i alpha   -> nP_v * O configs
    'abb'  particle a alpha, particle b beta, hole i beta -> V * V * O configs
"""
import numpy as np


class ADCSolverSpinBlocked:
    """
    Restricted spin-BLOCKED ADC(3) for determinant-wise Epstein-Nesbet.
    """

    # spin patterns per sector: (name, hole spins, particle spins)
    _PATTERNS = {
        '2h1p': [('aaa', (0, 0), (0,)), ('abb', (0, 1), (1,))],
        '2p1h': [('aaa', (0,), (0, 0)), ('abb', (1,), (0, 1))],
    }

    def __init__(self, eps, eri_chemist=None, B_aa=None, u2_denom_dress=None):
        if u2_denom_dress and u2_denom_dress.get('spin_adapted', False):
            raise ValueError(
                "ADCSolverSpinBlocked is the determinant-wise EN solver; "
                "u2_denom_dress={'spin_adapted': True} belongs to "
                "ADCSolverRestricted (the CSF solver), whose basis is complete "
                "for that variant. Passing it here would be a contradiction.")
        self.eps = np.asarray(eps, dtype=float)
        self.eri = eri_chemist
        self.B_aa = B_aa
        self.norb = len(self.eps)
        self.u2_denom_dress = u2_denom_dress
        self._cache = {}

    # ------------------------------------------------------------------
    # configuration space
    # ------------------------------------------------------------------
    def configs(self, nocc, sector):
        """Spin-blocked configurations of one sector, as a list of dicts with
        'name', 'holes' (ncfg, nh) spatial, 'parts' (ncfg, npart) spatial
        (vir-local, 0-based), 'K' (ncfg,), and 'offset' into the sector.

        Enumeration order within a pattern is C-order over the free indices,
        with the same-spin pair restricted to i<j (a<b) -- the deduplication
        the spin-orbital determinant list performs.
        """
        key = ('configs', nocc, sector)
        if key in self._cache:
            return self._cache[key]
        O, V = nocc, self.norb - nocc
        eo, ev = self.eps[:O], self.eps[O:]
        iu_o, ju_o = np.triu_indices(O, k=1)
        iu_v, ju_v = np.triu_indices(V, k=1)
        out, off = [], 0
        for name, hs, ps in self._PATTERNS[sector]:
            if sector == '2h1p':
                if name == 'aaa':                      # i<j alpha, a alpha
                    h = np.stack([np.repeat(iu_o, V), np.repeat(ju_o, V)], 1)
                    p = np.tile(np.arange(V), len(iu_o))[:, None]
                else:                                  # i alpha, j beta, a beta
                    ii, jj = np.meshgrid(np.arange(O), np.arange(O), indexing='ij')
                    h = np.stack([np.repeat(ii.ravel(), V),
                                  np.repeat(jj.ravel(), V)], 1)
                    p = np.tile(np.arange(V), O * O)[:, None]
                K = eo[h[:, 0]] + eo[h[:, 1]] - ev[p[:, 0]]
            else:
                if name == 'aaa':                      # a<b alpha, i alpha
                    p = np.stack([np.tile(iu_v, O), np.tile(ju_v, O)], 1)
                    h = np.repeat(np.arange(O), len(iu_v))[:, None]
                else:                                  # a alpha, b beta, i beta
                    aa, bb = np.meshgrid(np.arange(V), np.arange(V), indexing='ij')
                    p = np.stack([np.tile(aa.ravel(), O),
                                  np.tile(bb.ravel(), O)], 1)
                    h = np.repeat(np.arange(O), V * V)[:, None]
                K = ev[p[:, 0]] + ev[p[:, 1]] - eo[h[:, 0]]
            out.append(dict(name=name, hole_spins=hs, part_spins=ps,
                            holes=h, parts=p, K=K, offset=off, n=len(K)))
            off += len(K)
        self._cache[key] = out
        return out

    def dimensions(self, nocc):
        d = {'norb': self.norb}
        for sec in ('2h1p', '2p1h'):
            blocks = self.configs(nocc, sec)
            d[f'n{sec}'] = sum(b['n'] for b in blocks)
            for b in blocks:
                d[f'n{sec}_{b["name"]}'] = b['n']
        d['nH'] = self.norb + d['n2h1p'] + d['n2p1h']
        return d

    def k_diagonal(self, nocc, sector):
        return np.concatenate([b['K'] for b in self.configs(nocc, sector)])

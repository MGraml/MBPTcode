"""Determinant-space Hbar = e^{-T} H e^{T} for exact (exponential-scaling) EOM-CCSDT on small systems.

Projecting Hbar onto the EOM excitation manifolds (see eom.py) IS EOM-CCSDT,
exactly rather than via generated einsum code; also serves as an oracle for
future generated sigma/Davidson code.

Conventions (must match integrals.py/amplitudes.py): spin orbitals
interleaved (2p=alpha, 2p+1=beta), energy-sorted; h/g are soei/antisymmetrized
physicist ERIs; t2[a,b,i,j], t3[a,b,c,i,j,k] index order. Determinants are
integer bitmasks (orbital p <-> bit p), phase convention
|D> = a^dag_{p1} a^dag_{p2} ... |vac> for p1 < p2 < ... .

Locked in by tests/test_eom_ccsdt.py: <ref|Hbar|ref> == E_cc and
<mu|Hbar|ref> == 0 for every mu at converged amplitudes.
"""
from itertools import combinations

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# determinant bit-twiddling
# ---------------------------------------------------------------------------

def _popcount_below(det, p):
    """Number of occupied orbitals below p (for the fermionic phase)."""
    return bin(det & ((1 << p) - 1)).count('1')


def apply_annihilation(det, p):
    """a_p |det> -> (phase, new_det) or None if p unoccupied."""
    if not (det >> p) & 1:
        return None
    phase = 1 - 2 * (_popcount_below(det, p) & 1)
    return phase, det & ~(1 << p)


def apply_creation(det, p):
    """a^dag_p |det> -> (phase, new_det) or None if p occupied."""
    if (det >> p) & 1:
        return None
    phase = 1 - 2 * (_popcount_below(det, p) & 1)
    return phase, det | (1 << p)


def apply_ops(det, creations, annihilations):
    """Apply a^dag_{c1} a^dag_{c2} ... a_{a1} a_{a2} ... (annihilations
    applied right-to-left, i.e. a_{a_last} first). Returns (phase, det) or
    None."""
    phase = 1
    for p in reversed(annihilations):
        r = apply_annihilation(det, p)
        if r is None:
            return None
        ph, det = r
        phase *= ph
    for p in reversed(creations):
        r = apply_creation(det, p)
        if r is None:
            return None
        ph, det = r
        phase *= ph
    return phase, det


def occupied_orbitals(det):
    occ = []
    p = 0
    while det >> p:
        if (det >> p) & 1:
            occ.append(p)
        p += 1
    return occ


# ---------------------------------------------------------------------------
# sector basis
# ---------------------------------------------------------------------------

class SectorBasis:
    """All determinants of `nelec` electrons in `norb` spin orbitals,
    classified by particle rank (number of electrons in orbitals >= nocc_ref,
    i.e. reference virtuals)."""

    def __init__(self, norb, nelec, nocc_ref):
        self.norb = norb
        self.nelec = nelec
        self.nocc_ref = nocc_ref
        dets = []
        for occ in combinations(range(norb), nelec):
            d = 0
            for p in occ:
                d |= (1 << p)
            dets.append(d)
        virt_mask = ((1 << norb) - 1) ^ ((1 << nocc_ref) - 1)
        # sort by particle rank, then bitmask -- so manifold truncations are
        # contiguous leading blocks
        dets.sort(key=lambda d: (bin(d & virt_mask).count('1'), d))
        self.dets = dets
        self.index = {d: i for i, d in enumerate(dets)}
        self.virt_mask = virt_mask
        self.ranks = np.array([bin(d & virt_mask).count('1') for d in dets])

    @property
    def dim(self):
        return len(self.dets)

    def manifold_indices(self, max_particle_rank):
        """Indices of all determinants with particle rank <= max_particle_rank
        (contiguous leading block by construction)."""
        return np.where(self.ranks <= max_particle_rank)[0]


# ---------------------------------------------------------------------------
# operators in a sector basis
# ---------------------------------------------------------------------------

def build_hamiltonian(basis, h, g):
    """Sparse H in the sector basis via Slater-Condon rules.
    h = soei, g[p,q,r,s] = <pq||rs> (antisymmetrized physicist)."""
    rows, cols, vals = [], [], []
    norb = basis.norb
    for col, det in enumerate(basis.dets):
        occ = occupied_orbitals(det)
        unocc = [p for p in range(norb) if not (det >> p) & 1]

        # diagonal
        e = sum(h[p, p] for p in occ)
        e += 0.5 * sum(g[p, q, p, q] for p in occ for q in occ)
        rows.append(col)
        cols.append(col)
        vals.append(e)

        # singles i -> a
        for i in occ:
            for a in unocc:
                r = apply_ops(det, [a], [i])
                phase, newdet = r
                val = h[a, i] + sum(g[a, k, i, k] for k in occ if k != i)
                rows.append(basis.index[newdet])
                cols.append(col)
                vals.append(phase * val)

        # doubles (i<j) -> (a<b)
        for ii in range(len(occ)):
            for jj in range(ii + 1, len(occ)):
                i, j = occ[ii], occ[jj]
                for aa in range(len(unocc)):
                    for bb in range(aa + 1, len(unocc)):
                        a, b = unocc[aa], unocc[bb]
                        r = apply_ops(det, [a, b], [j, i])
                        phase, newdet = r
                        rows.append(basis.index[newdet])
                        cols.append(col)
                        vals.append(phase * g[a, b, i, j])

    return sp.csr_matrix((vals, (rows, cols)), shape=(basis.dim, basis.dim))


def build_one_body_operator(basis, p, q):
    """Sparse matrix of a^dag_p a_q in the sector basis."""
    rows, cols, vals = [], [], []
    for col, det in enumerate(basis.dets):
        r = apply_ops(det, [p], [q])
        if r is None:
            continue
        phase, newdet = r
        rows.append(basis.index[newdet])
        cols.append(col)
        vals.append(phase)
    return sp.csr_matrix((vals, (rows, cols)), shape=(basis.dim, basis.dim))


def build_t_operator(basis, t1, t2, t3, nocc):
    """Sparse matrix of T = T1 + T2 + T3 in the sector basis.

    T1 = sum_{ai} t1[a,i] a^dag_a a_i,
    T2 = sum_{a<b,i<j} t2[a,b,i,j] a^dag_a a^dag_b a_j a_i,
    T3 = sum_{a<b<c,i<j<k} t3[a,b,c,i,j,k] a^dag_a a^dag_b a^dag_c a_k a_j a_i
    (restricted sums absorb the 1/4 and 1/36 prefactors via amplitude
    antisymmetry). Amplitudes may act in ANY particle-number sector -- T only
    moves electrons from reference-occupied to reference-virtual orbitals."""
    nv = t1.shape[0]
    rows, cols, vals = [], [], []
    thresh = 0.0
    for col, det in enumerate(basis.dets):
        occ_o = [p for p in occupied_orbitals(det) if p < nocc]
        unocc_v = [p for p in range(nocc, nocc + nv) if not (det >> p) & 1]

        for i in occ_o:
            for a in unocc_v:
                val = t1[a - nocc, i]
                if val == thresh:
                    continue
                phase, newdet = apply_ops(det, [a], [i])
                rows.append(basis.index[newdet])
                cols.append(col)
                vals.append(phase * val)

        for ii in range(len(occ_o)):
            for jj in range(ii + 1, len(occ_o)):
                i, j = occ_o[ii], occ_o[jj]
                for aa in range(len(unocc_v)):
                    for bb in range(aa + 1, len(unocc_v)):
                        a, b = unocc_v[aa], unocc_v[bb]
                        val = t2[a - nocc, b - nocc, i, j]
                        if val == thresh:
                            continue
                        phase, newdet = apply_ops(det, [a, b], [j, i])
                        rows.append(basis.index[newdet])
                        cols.append(col)
                        vals.append(phase * val)

        for ii in range(len(occ_o)):
            for jj in range(ii + 1, len(occ_o)):
                for kk in range(jj + 1, len(occ_o)):
                    i, j, k = occ_o[ii], occ_o[jj], occ_o[kk]
                    for aa in range(len(unocc_v)):
                        for bb in range(aa + 1, len(unocc_v)):
                            for cc in range(bb + 1, len(unocc_v)):
                                a, b, c = unocc_v[aa], unocc_v[bb], unocc_v[cc]
                                val = t3[a - nocc, b - nocc, c - nocc, i, j, k]
                                if val == thresh:
                                    continue
                                phase, newdet = apply_ops(det, [a, b, c], [k, j, i])
                                rows.append(basis.index[newdet])
                                cols.append(col)
                                vals.append(phase * val)

    return sp.csr_matrix((vals, (rows, cols)), shape=(basis.dim, basis.dim))


def expm_nilpotent(T):
    """exp(T) for a (numerically) nilpotent sparse T -- the Taylor series
    terminates exactly because every application of T raises the particle
    rank by at least one."""
    dim = T.shape[0]
    E = sp.identity(dim, format='csr')
    term = sp.identity(dim, format='csr')
    k = 0
    while True:
        k += 1
        term = (term @ T) / k
        nnz_max = np.abs(term.data).max() if term.nnz else 0.0
        if nnz_max == 0.0:
            break
        E = E + term
        if k > 50:
            raise RuntimeError("expm_nilpotent did not terminate -- T is not "
                               "a pure excitation operator?")
    return E.tocsr()


def build_hbar(basis, h, g, t1, t2, t3, nocc):
    """Hbar = e^{-T} H e^{T} in the sector basis (sparse)."""
    H = build_hamiltonian(basis, h, g)
    T = build_t_operator(basis, t1, t2, t3, nocc)
    E = expm_nilpotent(T)
    Einv = expm_nilpotent(-T)
    return Einv @ H @ E, E, Einv

import numpy as np
from pyscf import scf, ao2mo

def get_effective_one_electron_integrals(mol, mf, representation='spatial'):
    fock_ao = mf.get_fock()
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        mo_a, mo_b = mf.mo_coeff[0], mf.mo_coeff[1]
        f_mo_a = mo_a.T @ fock_ao[0] @ mo_a
        f_mo_b = mo_b.T @ fock_ao[1] @ mo_b
        if representation == 'spin':
            norb = f_mo_a.shape[0]
            f_spin = np.zeros((2*norb, 2*norb))
            f_spin[0::2, 0::2] = f_mo_a
            f_spin[1::2, 1::2] = f_mo_b
            return f_spin
        else:
            return f_mo_a, f_mo_b
    else:
        mo = mf.mo_coeff
        f_mo = mo.T @ fock_ao @ mo
        if representation == 'spin':
            norb = f_mo.shape[0]
            f_spin = np.zeros((2*norb, 2*norb))
            f_spin[0::2, 0::2] = f_mo
            f_spin[1::2, 1::2] = f_mo
            return f_spin
        else:
            return f_mo

def get_orbital_energies(mf, representation='spatial'):
    mo_energy = mf.mo_energy
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        eps_a, eps_b = mo_energy[0], mo_energy[1]
        if representation == 'spin':
            norb = len(eps_a)
            eps_spin = np.zeros(2*norb)
            eps_spin[0::2] = eps_a
            eps_spin[1::2] = eps_b
            return eps_spin
        else:
            return eps_a, eps_b
    else:
        if representation == 'spin':
            norb = len(mo_energy)
            eps_spin = np.zeros(2*norb)
            eps_spin[0::2] = mo_energy
            eps_spin[1::2] = mo_energy
            return eps_spin
        else:
            return mo_energy

def get_two_electron_integrals_chemist(mol, mf, representation='spatial'):
    is_uhf = isinstance(mf, scf.uhf.UHF)
    if is_uhf:
        mo_a, mo_b = mf.mo_coeff[0], mf.mo_coeff[1]
        norb = mo_a.shape[1]
        # aa|aa
        eri_aa = ao2mo.general(mol, (mo_a, mo_a, mo_a, mo_a), compact=False).reshape(norb, norb, norb, norb)
        # aa|bb
        eri_ab = ao2mo.general(mol, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(norb, norb, norb, norb)
        # bb|bb
        eri_bb = ao2mo.general(mol, (mo_b, mo_b, mo_b, mo_b), compact=False).reshape(norb, norb, norb, norb)

        if representation == 'spin':
            eri_spin = np.zeros((2*norb, 2*norb, 2*norb, 2*norb))
            eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aa
            eri_spin[0::2, 0::2, 1::2, 1::2] = eri_ab
            eri_spin[1::2, 1::2, 0::2, 0::2] = eri_ab.transpose(2, 3, 0, 1)
            eri_spin[1::2, 1::2, 1::2, 1::2] = eri_bb
            return eri_spin
        else:
            return eri_aa, eri_ab, eri_bb
    else:
        mo = mf.mo_coeff
        norb = mo.shape[1]
        eri_chemist = ao2mo.kernel(mol, mo, compact=False).reshape(norb, norb, norb, norb)
        if representation == 'spin':
            eri_spin = np.zeros((2*norb, 2*norb, 2*norb, 2*norb))
            eri_spin[0::2, 0::2, 0::2, 0::2] = eri_chemist
            eri_spin[0::2, 0::2, 1::2, 1::2] = eri_chemist
            eri_spin[1::2, 1::2, 0::2, 0::2] = eri_chemist
            eri_spin[1::2, 1::2, 1::2, 1::2] = eri_chemist
            return eri_spin
        else:
            return eri_chemist

def get_two_electron_integrals_physicist(mol, mf, representation='spatial'):
    eri_chem = get_two_electron_integrals_chemist(mol, mf, representation=representation)
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if representation == 'spin':
        return convert_chemist_to_physicist(eri_chem)
    else:
        if is_uhf:
            eri_aa, eri_ab, eri_bb = eri_chem
            return (
                convert_chemist_to_physicist(eri_aa),
                eri_ab.transpose(0, 2, 1, 3), # physicist <ac|bd> = chemist (ab|cd)
                convert_chemist_to_physicist(eri_bb)
            )
        else:
            return convert_chemist_to_physicist(eri_chem)

def convert_chemist_to_physicist(eri_chemist):
    # <pq|rs> = (pr|qs)
    return eri_chemist.transpose(0, 2, 1, 3)

def get_antisymmetrized_integrals(eri_physicist):
    # <pq||rs> = <pq|rs> - <pq|sr>
    return eri_physicist - eri_physicist.transpose(0, 1, 3, 2)

def get_coulomb_exchange_diagonals(eri_physicist):
    """(J_pq, K_pq) = (<pq|pq>, <pq|qp>) = ((pp|qq), (pq|pq)) for one physicist block."""
    direct   = np.einsum('pqpq->pq', eri_physicist, optimize=True)
    exchange = np.einsum('pqqp->pq', eri_physicist, optimize=True)
    return direct, exchange

def get_coulomb_exchange_diagonals_df(B_block):
    """get_coulomb_exchange_diagonals from a DF factor block B_block[Q,p,q], in
    O(naux*n^2) -- never forms the (n,n,n,n) tensor."""
    diag     = np.einsum('Qpp->Qp', B_block, optimize=True)
    direct   = diag.T @ diag
    exchange = np.einsum('Qpq,Qpq->pq', B_block, B_block, optimize=True)
    return direct, exchange

def get_density_fitting_coefficients(mol, mf, representation='spatial'):
    is_uhf = isinstance(mf, scf.uhf.UHF)

    # Check if df is used and initialized in PySCF
    has_df = hasattr(mf, 'with_df') and mf.with_df is not None

    if is_uhf:
        mo_a, mo_b = mf.mo_coeff[0], mf.mo_coeff[1]
        norb = mo_a.shape[1]
        if has_df:
            try:
                from pyscf import lib
                coeff_a_list = []
                coeff_b_list = []
                for chunk in mf.with_df.loop(blksize=200):
                    ao_3c = lib.unpack_tril(chunk)
                    naux_block = ao_3c.shape[0]
                    nao = ao_3c.shape[1]
                    nmo_a = mo_a.shape[1]
                    nmo_b = mo_b.shape[1]

                    tmp_a = (ao_3c.reshape(-1, nao) @ mo_a).reshape(naux_block, nao, nmo_a)
                    tmp_t_a = tmp_a.transpose(0, 2, 1).reshape(-1, nao)
                    mo_3c_a = (tmp_t_a @ mo_a).reshape(naux_block, nmo_a, nmo_a)
                    coeff_a_list.append(mo_3c_a)

                    tmp_b = (ao_3c.reshape(-1, nao) @ mo_b).reshape(naux_block, nao, nmo_b)
                    tmp_t_b = tmp_b.transpose(0, 2, 1).reshape(-1, nao)
                    mo_3c_b = (tmp_t_b @ mo_b).reshape(naux_block, nmo_b, nmo_b)
                    coeff_b_list.append(mo_3c_b)
                coeff_a = np.concatenate(coeff_a_list, axis=0)
                coeff_b = np.concatenate(coeff_b_list, axis=0)
            except Exception:
                has_df = False
        if not has_df:
            # Fallback: decompose the AO-basis (spin-independent) ERI ONCE,
            # then transform by mo_a/mo_b separately -- mirrors the has_df
            # branch's own ao_3c-then-per-spin-transform structure exactly.
            # Decomposing eri_aa and eri_bb independently (as an earlier
            # version of this fallback did) gives alpha and beta unrelated
            # auxiliary bases: the aa/bb diagonal blocks come back exact
            # either way, but the abab cross block is then unreproducible by
            # construction (no shared Q to contract alpha against beta) --
            # caught by DFIntegrals.reconstruct_g's cross-check against
            # get_antisymmetrized_spin_block_eri (~0.08 max abs error on a
            # UHF water/cc-pVDZ test, vs ~3e-13 for aaaa/bbbb).
            eri_ao = mol.intor('int2e')
            nao = eri_ao.shape[0]
            w, vv = np.linalg.eigh(eri_ao.reshape(nao*nao, nao*nao))
            keep = w > 1e-12
            coeff_ao = (vv[:, keep] * np.sqrt(w[keep])).T.reshape(-1, nao, nao)
            coeff_a = np.einsum('Qmn,mp,nr->Qpr', coeff_ao, mo_a, mo_a)
            coeff_b = np.einsum('Qmn,mp,nr->Qpr', coeff_ao, mo_b, mo_b)

        if representation == 'spin':
            # Same shared naux for both spin blocks (see fallback comment
            # above for why alpha/beta must share one auxiliary index range,
            # not get disjoint naux_a/naux_b slices) -- mirrors the RHF
            # branch below's coeff_spin[:, 0::2/1::2, ...] = coeff pattern.
            naux = coeff_a.shape[0]
            coeff_spin = np.zeros((naux, 2*norb, 2*norb))
            coeff_spin[:, 0::2, 0::2] = coeff_a
            coeff_spin[:, 1::2, 1::2] = coeff_b
            return coeff_spin
        else:
            return coeff_a, coeff_b
    else:
        mo = mf.mo_coeff
        norb = mo.shape[1]
        if has_df:
            try:
                from pyscf import lib
                coeff_list = []
                for chunk in mf.with_df.loop(blksize=200):
                    ao_3c = lib.unpack_tril(chunk)
                    naux_block = ao_3c.shape[0]
                    nao = ao_3c.shape[1]
                    nmo = mo.shape[1]

                    tmp = (ao_3c.reshape(-1, nao) @ mo).reshape(naux_block, nao, nmo)
                    tmp_t = tmp.transpose(0, 2, 1).reshape(-1, nao)
                    mo_3c = (tmp_t @ mo).reshape(naux_block, nmo, nmo)
                    coeff_list.append(mo_3c)
                coeff = np.concatenate(coeff_list, axis=0)
            except Exception:
                has_df = False
        if not has_df:
            eri_chemist = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
            w, v = np.linalg.eigh(eri_chemist.reshape(norb*norb, norb*norb))
            keep = w > 1e-12
            coeff = (v[:, keep] * np.sqrt(w[keep])).T.reshape(-1, norb, norb)

        if representation == 'spin':
            naux = coeff.shape[0]
            coeff_spin = np.zeros((naux, 2*norb, 2*norb))
            coeff_spin[:, 0::2, 0::2] = coeff
            coeff_spin[:, 1::2, 1::2] = coeff
            return coeff_spin
        else:
            return coeff

class DFIntegrals:
    """Container for the RI/DF 3-index MO factor B_Q,pq, parallel to
    get_antisymmetrized_spin_block_eri's dense (g_aaaa, g_bbbb, g_abab)
    triple -- one B block per spin channel (B_aa, B_bb; RHF aliases
    B_bb = B_aa, same convention get_antisymmetrized_spin_block_eri uses for
    g_bbbb = g_aaaa), each shaped (naux, norb, norb).

    Reconstructs the same antisymmetrized-within-spin physicist blocks
    get_antisymmetrized_spin_block_eri returns:
        g_aaaa[p,q,r,s] = sum_Q B_aa[Q,p,r]*B_aa[Q,q,s] - sum_Q B_aa[Q,p,s]*B_aa[Q,q,r]
        g_bbbb[p,q,r,s] = sum_Q B_bb[Q,p,r]*B_bb[Q,q,s] - sum_Q B_bb[Q,p,s]*B_bb[Q,q,r]
        g_abab[p,q,r,s] = sum_Q B_aa[Q,p,r]*B_bb[Q,q,s]                      (no exchange)
    reconstruct_g() is a plumbing self-check only (materializes the norb^4
    tensor it exists to avoid in production) -- see df_codegen.py for the
    codegen pass that consumes B_aa/B_bb without ever forming g_pqrs.
    """
    def __init__(self, B_aa, B_bb):
        self.B_aa = B_aa
        self.B_bb = B_bb
        self.naux_aa = B_aa.shape[0]
        self.naux_bb = B_bb.shape[0]

    @classmethod
    def from_scf(cls, mol, mf, exact=False):
        """exact=True forces the eigh/Cholesky fallback (naux = norb^2) even
        when mf carries real DF (mf.with_df) approximation."
        """
        is_uhf = isinstance(mf, scf.uhf.UHF)
        if exact:
            mf_for_coeffs = mf
            has_df_attr = hasattr(mf, 'with_df')
            saved = mf.with_df if has_df_attr else None
            try:
                mf.with_df = None
                coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
            finally:
                if has_df_attr:
                    mf.with_df = saved
        else:
            coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        if is_uhf:
            B_aa, B_bb = coeff
        else:
            B_aa = B_bb = coeff
        return cls(B_aa, B_bb)

    def reconstruct_g(self):
        """Dense (g_aaaa, g_bbbb, g_abab) triple, for validation only."""
        g_aaaa = (np.einsum('Qpr,Qqs->pqrs', self.B_aa, self.B_aa)
                  - np.einsum('Qps,Qqr->pqrs', self.B_aa, self.B_aa))
        g_bbbb = (np.einsum('Qpr,Qqs->pqrs', self.B_bb, self.B_bb)
                  - np.einsum('Qps,Qqr->pqrs', self.B_bb, self.B_bb))
        g_abab = np.einsum('Qpr,Qqs->pqrs', self.B_aa, self.B_bb)
        return g_aaaa, g_bbbb, g_abab


def get_antisymmetrized_spin_block_eri(mol, mf, eri_chemist=None):
    """Spin-block antisymmetrized-within-spin physicist ERIs (g_aaaa, g_bbbb, g_abab), spatial-MO.

    g_aaaa/g_bbbb = <pq|rs> - <pq|sr> within one spin channel (same formula
    get_antisymmetrized_integrals applies in the restricted spin-orbital
    case); g_abab = <p_a q_b|r_a s_b> with NO exchange term, since the
    exchange integral <p_a q_b|s_a r_b> vanishes by spin orthogonality
    whenever p,q have different spins (the same reason
    get_antisymmetrized_spin_eri's off-diagonal spin blocks come out zero
    before its final subtraction). Works for both RHF (all three blocks
    built from the one restricted physicist tensor) and UHF mf.

    eri_chemist: RHF only. If the caller already has mf's own spatial
    chemist-notation ERI in hand (e.g. for the static-correction contraction
    downstream), pass it here to skip re-running the O(N^5) AO->MO
    transform and holding a second, numerically identical norb^4 array --
    get_two_electron_integrals_physicist's own internal chemist tensor is
    just this same array transposed to physicist order. Ignored for UHF
    (whose per-spin blocks aren't derivable from a single spatial tensor).
    """
    is_uhf = isinstance(mf, scf.uhf.UHF)
    if not is_uhf and eri_chemist is not None:
        eri_phys = convert_chemist_to_physicist(eri_chemist)
    else:
        eri_phys = get_two_electron_integrals_physicist(mol, mf, representation='spatial')
    if is_uhf:
        phys_aa, phys_ab, phys_bb = eri_phys
    else:
        phys_aa = phys_bb = phys_ab = eri_phys

    g_aaaa = get_antisymmetrized_integrals(phys_aa)
    # For RHF, alias g_bbbb = g_aaaa (same object) instead of building an
    # independent, numerically identical norb^4 copy (~1.2GB at cc-pVQZ CO).
    # This makes `self.g_aaaa is self.g_bbbb` true, activating
    # MP3DensityMatrixSolverUnrestricted._is_restricted()'s symmetry-reduced
    # fast-path in compute_t3_2/compute_t1_3/compute_gamma3_blocks; that fast
    # path's abb-from-aab relabeling had a sign/transpose bug (fixed in
    # density_matrix.py, see compute_t3_2's abb branch) and is now verified
    # to reproduce the full (non-fast-path) UHF branch to ~1e-18 -- see
    # tests/test_mp3_density_matrix.py.
    g_bbbb = g_aaaa if not is_uhf else get_antisymmetrized_integrals(phys_bb)
    g_abab = phys_ab
    return g_aaaa, g_bbbb, g_abab

def uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b):
    """Index permutation mapping [alpha-block, beta-block] (mo-index order within
    each spin) to block-stacked [occ_alpha, occ_beta, virt_alpha, virt_beta] order.

    Shared by get_uhf_spin_orbital_arrays_blockstacked (for eps/g) and any caller
    that needs to place its own per-spin arrays (e.g. a density-matrix correction)
    into that same ordering.
    """
    return np.concatenate([np.arange(0, nocc_a), np.arange(norb_a, norb_a + nocc_b),
                            np.arange(nocc_a, norb_a), np.arange(norb_a + nocc_b, norb_a + norb_b)])

def get_uhf_spin_orbital_arrays_blockstacked(mol, mf):
    """UHF spin-orbital (eps_spin, g_anti_spin, nocc_spin) in block-stacked order
    [occ_alpha, occ_beta, virt_alpha, virt_beta].

    Unlike get_orbital_energies(..., representation='spin')'s even/odd alpha/beta
    interleaving (which only yields a contiguous occ/virt split when
    nocc_a == nocc_b), this ordering gives a valid contiguous occ = arange(0,
    nocc_spin) / virt = arange(nocc_spin, norb) split for ANY UHF occupation,
    including genuine open-shell (nocc_a != nocc_b) -- required by any spin-orbital
    consumer (e.g. ADCSolver) that slices strictly by position, not by spin label.
    Built from get_antisymmetrized_spin_block_eri's (g_aaaa, g_bbbb, g_abab):
    the alpha-beta exchange blocks <p_a q_b||r_b s_a> vanish via the direct term
    (mismatched spins) and are nonzero only via the exchange term, which reduces to
    a sign-flipped transpose of g_abab (see get_antisymmetrized_spin_block_eri's
    docstring for the same identity applied to the "direct-order" abab block).
    """
    nocc_a, nocc_b = mf.nelec
    eps_a, eps_b = get_orbital_energies(mf, representation='spatial')
    g_aaaa, g_bbbb, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
    norb_a, norb_b = len(eps_a), len(eps_b)

    order = uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b)
    eps_spin = np.concatenate([eps_a, eps_b])[order]

    nso = norb_a + norb_b
    g_so = np.zeros((nso, nso, nso, nso))
    g_so[0:norb_a, 0:norb_a, 0:norb_a, 0:norb_a] = g_aaaa
    g_so[norb_a:, norb_a:, norb_a:, norb_a:] = g_bbbb
    g_so[0:norb_a, norb_a:, 0:norb_a, norb_a:] = g_abab
    g_so[norb_a:, 0:norb_a, norb_a:, 0:norb_a] = g_abab.transpose(1, 0, 3, 2)
    g_so[0:norb_a, norb_a:, norb_a:, 0:norb_a] = -g_abab.transpose(0, 1, 3, 2)
    g_so[norb_a:, 0:norb_a, 0:norb_a, norb_a:] = -g_abab.transpose(1, 0, 2, 3)
    g_anti_spin = g_so[np.ix_(order, order, order, order)]

    return eps_spin, g_anti_spin, nocc_a + nocc_b

def get_uhf_spin_orbital_df_factor_blockstacked(mol, mf, exact=False):
    """UHF spin-orbital DF factor B_so (naux, nso, nso) in the SAME
    block-stacked order as get_uhf_spin_orbital_arrays_blockstacked
    ([occ_alpha, occ_beta, virt_alpha, virt_beta]), built block-diagonal in
    spin from DFIntegrals.from_scf's separate B_aa/B_bb (naux, norb_a,norb_a)/
    (naux, norb_b,norb_b) -- B_so[Q,p,r] nonzero only when spin(p)==spin(r),
    the same convention get_df_spin_orbital_factor uses for the RHF/
    interleaved case (that function assumes B_bb==B_aa; this one does not).

    Reproduces get_uhf_spin_orbital_arrays_blockstacked's g_anti_spin exactly
    via the same antisymmetrized formula g_elem_df already implements:
        <pq||rs> = sum_Q B_so[Q,p,r]*B_so[Q,q,s] - sum_Q B_so[Q,p,s]*B_so[Q,q,r]
    The cross-spin exchange term vanishes automatically from the
    block-diagonal structure (spin(p)!=spin(s) or spin(q)!=spin(r) forces one
    factor to zero), so unlike get_antisymmetrized_spin_block_eri this needs
    no separate abab-transpose bookkeeping -- one formula covers aaaa/bbbb/
    abab/abba/abba-exchange all at once, verified against the dense route in
    tests before use.

    O(naux*nso^2) to store, vs O(nso^4) for the dense g_anti_spin.
    """
    nocc_a, nocc_b = mf.nelec
    df = DFIntegrals.from_scf(mol, mf, exact=exact)
    B_aa, B_bb = df.B_aa, df.B_bb
    if B_aa.shape[0] != B_bb.shape[0]:
        raise ValueError(f"alpha/beta auxiliary dimensions differ "
                         f"({B_aa.shape[0]} vs {B_bb.shape[0]}) -- the "
                         f"block-diagonal embedding below assumes a shared "
                         f"naux axis")
    naux = B_aa.shape[0]
    norb_a, norb_b = B_aa.shape[1], B_bb.shape[1]
    nso = norb_a + norb_b

    order = uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b)
    B_so_pre = np.zeros((naux, nso, nso))
    B_so_pre[:, 0:norb_a, 0:norb_a] = B_aa
    B_so_pre[:, norb_a:, norb_a:] = B_bb
    return B_so_pre[np.ix_(np.arange(naux), order, order)]

def embed_spatial_eri_in_spin_orbitals(eri_physicist):
    """Spin-orbital (interleaved alpha/beta) <PQ|RS> from a spatial physicist <pq|rs>,
    NOT antisymmetrized. Only the four spin-allowed blocks are nonzero; the exchange
    integral <p_a q_b|s_a r_b> vanishes by spin orthogonality, so antisymmetrizing the
    result (get_antisymmetrized_integrals) gives the correct spin selection rules."""
    norb = eri_physicist.shape[0]
    n_spin = 2 * norb
    phys_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    phys_spin[0::2, 0::2, 0::2, 0::2] = eri_physicist
    phys_spin[0::2, 1::2, 0::2, 1::2] = eri_physicist
    phys_spin[1::2, 0::2, 1::2, 0::2] = eri_physicist
    phys_spin[1::2, 1::2, 1::2, 1::2] = eri_physicist
    return phys_spin

def get_spin_orbital_eri_physicist(eri_chemist):
    """<PQ|RS> restricted-reference spin-orbital ERI (interleaved alpha/beta), NOT
    antisymmetrized, from a spatial chemist (pq|rs) tensor. Works for the bare ERI or
    for any tensor in the same layout (e.g. a screened W)."""
    return embed_spatial_eri_in_spin_orbitals(convert_chemist_to_physicist(eri_chemist))

def get_antisymmetrized_spin_eri(eri_chemist):
    """<pq||rs> restricted-reference spin-orbital antisymmetrized ERI (interleaved alpha/beta) from spatial chemist ERI."""
    return get_antisymmetrized_integrals(get_spin_orbital_eri_physicist(eri_chemist))

def get_df_spin_orbital_factor(B_aa):
    """Spin-orbital (interleaved alpha/beta) DF factor B_spin (naux, 2*norb,
    2*norb) from the RHF spatial factor B_aa (naux, norb, norb), the
    spin-orbital analog of get_antisymmetrized_spin_eri.
    Block-diagonal in spin (B_spin[Q,p,r]
    nonzero only when spin(p)==spin(r)), so
        sum_Q B_spin[Q,p,r]*B_spin[Q,q,s] - sum_Q B_spin[Q,p,s]*B_spin[Q,q,r]
    reconstructs get_antisymmetrized_spin_eri's g_anti_spin[p,q,r,s] exactly
    (the direct term's spin selection rule -- spin(p)=spin(r) and
    spin(q)=spin(s) -- falls out automatically from this block structure,
    same as the phys_spin[0::2,0::2,0::2,0::2]-etc. assignment there)."""
    naux, norb, _ = B_aa.shape
    n_spin = 2 * norb
    B_spin = np.zeros((naux, n_spin, n_spin))
    B_spin[:, 0::2, 0::2] = B_aa
    B_spin[:, 1::2, 1::2] = B_aa
    return B_spin

def g_elem_df(B_spin, p, q, r, s):
    """Elementwise/broadcast <pq||rs> from a spin-orbital DF factor B_spin
    (naux, nso, nso) -- a drop-in replacement for a dense slice g[p, q, r, s] that
    never materializes the gathered block:

        <pq||rs> = sum_Q B[Q,p,r] B[Q,q,s] - sum_Q B[Q,p,s] B[Q,q,r]

    p,q,r,s must ALL be int/array (np.ix_ block gathers and per-config fancy indexing
    both qualify); they broadcast together exactly as numpy advanced indexing on 4
    contiguous axes does. NOT valid for patterns mixing a bare `:` slice with arrays,
    e.g. g[a, :, i, j]."""
    p, q, r, s = np.broadcast_arrays(np.asarray(p), np.asarray(q), np.asarray(r), np.asarray(s))
    direct = np.einsum('q...,q...->...', B_spin[:, p, r], B_spin[:, q, s], optimize=True)
    exchange = np.einsum('q...,q...->...', B_spin[:, p, s], B_spin[:, q, r], optimize=True)
    return direct - exchange

class GProxy:
    """[p, q, r, s] indexing that dispatches to g_elem_df(B_spin, ...) when B_spin is
    given, else to plain dense-tensor indexing. Lets a consumer swap `g` for
    `GProxy(B_spin, g)` and keep every existing g[p, q, r, s] call site unchanged (see
    g_elem_df for which indexing patterns are valid)."""
    __slots__ = ('_B_spin', '_g_dense')

    def __init__(self, B_spin, g_dense):
        self._B_spin = B_spin
        self._g_dense = g_dense

    def __getitem__(self, key):
        if self._B_spin is not None:
            p, q, r, s = key
            return g_elem_df(self._B_spin, p, q, r, s)
        return self._g_dense[key]

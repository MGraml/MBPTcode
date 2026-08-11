"""G0W@CC: GW self-energy built from a coupled-cluster polarizability (Lewis & Berkelbach, JCTC 15, 2925 (2019), Eq. 12).

Same Lehmann structure as GW@RPA/GW@BSE, but with Casida excitation
energies/amplitudes replaced by EOM-CC ones (CCSDT or CCSD, from eom.py):

    Sigma_c,pq(w) = sum_{n,i} V^n_pi Vt^n_qi / (w - eps_i + Omega_n - i eta)
                  + sum_{n,a} V^n_pa Vt^n_qa / (w - eps_a - Omega_n + i eta)
    V^n_pq  = sum_rs (pq|rs) rho_n(rs),   rho_n(rs)  = <Psi~_0|r^dag s|Psi_n>
    Vt^n_pq = sum_rs (pq|rs) rho*_n(rs),  rho*_n(rs) = <Psi~_n|r^dag s|Psi_0>

rho != rho* since Hbar is non-Hermitian. HF reference: no v_xc correction,
so QP eq. is w = eps_p + Re Sigma_c,pp(w).

Spin-orbital, full-ERI. EOM-CCSDT (or CCSD) via eom.py's generated
sigma-vector Davidson solver -- scales far better than exact determinant
enumeration, though still full-ERI/no-symmetry so realistically small-to-
medium systems only. Validated against the paper's Table 1 (H2/He, def2-SVP)
by tests/test_gw_cc_polarizability.py.
"""
import numpy as np

from src.Base.constants import DEFAULT_BROADENING_ETA
from src.Solvers.qp_equation import solve_qp_equation
from src.SingleReference.CC.eom import EOMCC

HARTREE_TO_EV = 27.2114


def dipole_integrals_so(mf):
    """(3, nso, nso) electric-dipole (position) integrals in the interleaved
    spin-orbital MO basis, for EOMResult.polarizability()."""
    dip_ao = mf.mol.intor('int1e_r')
    C = mf.mo_coeff
    dip_mo = np.einsum('xuv,up,vq->xpq', dip_ao, C, C, optimize=True)
    n = 2 * C.shape[1]
    dip_so = np.zeros((3, n, n))
    dip_so[:, 0::2, 0::2] = dip_mo
    dip_so[:, 1::2, 1::2] = dip_mo
    return dip_so


def spin_orbital_chemist_eri(mf):
    """(pq|rs) in the interleaved spin-orbital MO basis (chemist notation,
    same orbital ordering as integrals.py)."""
    from pyscf import ao2mo
    nmo = mf.mo_coeff.shape[1]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), nmo)
    n = 2 * nmo
    eri_so = np.zeros((n, n, n, n))
    for s1 in (0, 1):
        for s2 in (0, 1):
            eri_so[s1::2, s1::2, s2::2, s2::2] = eri
    return eri_so


class GWCCSelfEnergy:
    """Correlation self-energy with CC screening (Eq. 12 of the paper).

    Parameters: an EOMCC instance (or mf + level to build one). All EE
    eigenstates and transition densities are computed once at construction;
    sigma_c(w) evaluations are then cheap.
    """

    def __init__(self, mf, level='ccsdt', eom=None, nroots=8, verbose=False):
        """nroots: EE states included in the Lehmann sum. Pad generously
        beyond the number of "real" low-lying states you care about --
        non-Hermitian Davidson needs elbow room past the exact size of a
        degenerate cluster (e.g. a p-shell excitation manifold) to reliably
        resolve every member of it; requesting too few silently returns an
        incomplete/inconsistent cluster (caught downstream in
        transition_densities' biorthogonalization, which raises rather than
        returning wrong numbers -- confirmed on He/def2-SVP's exactly
        3-fold-degenerate lowest excitation)."""
        self.eom = eom if eom is not None else EOMCC(mf, level=level, verbose=verbose)
        self.res = self.eom.kernel('ee', nroots=nroots)
        rho, rho_star = self.res.transition_densities()

        eri_so = spin_orbital_chemist_eri(mf)
        # Kept complex: degenerate EOM roots give complex-conjugate eigenvector
        # pairs whose Im*Im cross term is dropped if truncated to real early.
        self.V = np.einsum('pqrs,nrs->npq', eri_so, rho, optimize=True)
        self.Vt = np.einsum('pqrs,nrs->npq', eri_so, rho_star, optimize=True)

        self.omega_n = self.res.omega            # excitation energies > 0
        self.eps = np.diagonal(self.eom.ints['fock']).copy()
        self.nocc = self.eom.nocc

    def sigma_c(self, w, p, q=None, eta=DEFAULT_BROADENING_ETA):
        """Sigma_c,pq(w) (complex). Diagonal element if q is None."""
        if q is None:
            q = p
        o = slice(None, self.nocc)
        v = slice(self.nocc, None)
        num_o = self.V[:, p, o] * self.Vt[:, q, o]      # (n, i)
        num_v = self.V[:, p, v] * self.Vt[:, q, v]      # (n, a)
        den_o = w - self.eps[o][None, :] + self.omega_n[:, None] - 1j * eta
        den_v = w - self.eps[v][None, :] - self.omega_n[:, None] + 1j * eta
        return np.sum(num_o / den_o) + np.sum(num_v / den_v)

    def solve_qp(self, p, eta=DEFAULT_BROADENING_ETA, method='graphical'):
        """Solve w = eps_p + Re Sigma_c,pp(w) (HF reference). Returns Hartree."""
        func = lambda w: w - self.eps[p] - self.sigma_c(w, p, eta=eta).real
        return solve_qp_equation(func, self.eps[p], method=method)


def calc_qp_energy_gwcc(mf, level='ccsdt', state='homo', eta=DEFAULT_BROADENING_ETA,
                        method='graphical', nroots=8, return_solver=False, verbose=False):
    """G0W@CC quasiparticle energy (eV, matching calc_qp_energy's convention).

    mf: converged closed-shell RHF (the HF reference the paper uses -- a KS
    reference would additionally need a v_xc correction, not implemented).
    level: 'ccsdt' or 'ccsd' screening (EOM-CCSDT / EOM-CCSD polarizability).
    state: 'homo', 'lumo', or a spin-orbital index.
    nroots: EE states in the Lehmann sum -- see GWCCSelfEnergy's docstring
        for why this needs generous padding past any degenerate manifold.
    """
    solver = GWCCSelfEnergy(mf, level=level, nroots=nroots, verbose=verbose)
    if state == 'homo':
        p = solver.nocc - 1
    elif state == 'lumo':
        p = solver.nocc
    else:
        p = int(state)
    qp_ev = solver.solve_qp(p, eta=eta, method=method) * HARTREE_TO_EV
    if return_solver:
        return qp_ev, solver
    return qp_ev

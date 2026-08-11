import numpy as np
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.base import get_occ_virt_indices
from src.Base.constants import DEFAULT_BROADENING_ETA, DEFAULT_BLOCK_SIZE, get_method_info
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.Solvers.qp_equation import solve_qp_equation_newton

# 'PSD3' is intentionally absent -- unimplemented, raises ValueError rather than
# silently falling back to plain GW.
KNOWN_VERTEX_MODES = ('GW', 'GWGammaInf', 'PSD1', 'PSD2', 'PSD4', 'PSD5', 'PSD6', 'PSD7', 'PSD8', 'PSD9')

class SelfEnergySolver(AmplitudeGenerator):
    """Diagonal GW and vertex-corrected self-energies (Sigma_pp): restricted/unrestricted spin, bare/screened, DF or full ERIs."""
    def __init__(self, eps, df_coeff=None, eri_chemist=None, spin_mode='restricted',
                 eta=DEFAULT_BROADENING_ETA, block_size=DEFAULT_BLOCK_SIZE):
        """eps/df_coeff/eri_chemist: single array if restricted, (alpha, beta[, ab]) tuple if unrestricted."""
        self.spin_mode = spin_mode.lower()
        self.eta = eta
        self.block_size = block_size

        if self.spin_mode == 'unrestricted':
            self.eps_a, self.eps_b = eps
            if df_coeff is not None:
                if len(df_coeff) == 3:
                    self.df_a, self.df_b, self.df_ab = df_coeff
                else:
                    self.df_a, self.df_b = df_coeff
                    self.df_ab = None
            else:
                self.df_a, self.df_b, self.df_ab = None, None, None
            self.eri_a, self.eri_b, self.eri_ab = eri_chemist if eri_chemist is not None else (None, None, None)
            self.norb_a = len(self.eps_a)
            self.norb_b = len(self.eps_b)
            if self.df_a is not None:
                self.naux = self.df_a.shape[0]
        else:
            self.eps = eps
            self.df_coeff = df_coeff
            self.eri_chemist = eri_chemist
            self.norb = len(self.eps)
            if self.df_coeff is not None:
                self.naux = self.df_coeff.shape[0]

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)

    def _denom_grid(self, w, eps, nocc_spin, eigenvalues_casida, calc_imag):
        """Vectorized (nexciton, norb) energy-denominator grid: denom[S,i] = f(w - eps[i] + sign(i)*Omega[S]), sign=+1 iff i<nocc_spin."""
        norb = len(eps)
        sign = np.where(np.arange(norb) < nocc_spin, 1.0, -1.0)
        energy = w - eps[None, :] + sign[None, :] * eigenvalues_casida[:, None]
        if calc_imag:
            return -sign[None, :] * self.eta / (energy**2 + self.eta**2)
        return energy / (energy**2 + self.eta**2)

    def calculate_self_energy(self, p_state, freq, nocc, eigenvalues_casida, chiXYa, chiXYb=None,
                              eigenvalues_casida_t=None, chiXYb_t=None, spin_channel='alpha', vertex_mode='GW',
                              calc_imag=False):
        """Diagonal self-energy Sigma_pp(omega) for a single frequency or grid; vertex_mode: 'GW', 'GWGammaInf', 'PSD1'-'PSD9'."""
        is_scalar = np.isscalar(freq)
        freq_grid = np.atleast_1d(freq)

        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
            prefactor = 1.0
        else:
            eps = self.eps
            nocc_spin = nocc
            prefactor = 2.0

        norb = len(eps)
        nexciton = len(eigenvalues_casida)

        if chiXYa.ndim == 2:
            amp2 = chiXYa
        else:
            amp2 = chiXYa[:, :, p_state]

        if chiXYb is not None:
            if chiXYb.ndim == 2:
                amp = chiXYb
            else:
                amp = chiXYb[:, :, p_state]
        else:
            amp = None

        if vertex_mode not in KNOWN_VERTEX_MODES:
            raise ValueError(
                f"Unknown vertex_mode '{vertex_mode}'; expected one of {KNOWN_VERTEX_MODES}."
            )

        sigma_grid = []
        for w in freq_grid:
            denom = self._denom_grid(w, eps, nocc_spin, eigenvalues_casida, calc_imag)

            if vertex_mode == 'GW' or amp is None:
                val = prefactor * np.sum((amp2**2) * denom)
            elif vertex_mode == 'GWGammaInf':
                amp_L = amp2 - 0.5 * amp
                val = prefactor * np.sum(amp2 * amp_L * denom)
            else:
                # PSD1/2/4/5/6/7/8/9 all build on the plain-GW sum and the
                # vertex-corrected "2*amp2-amp" sum; computed once here.
                amp_psd = 2.0 * amp2 - amp
                if self.spin_mode == 'unrestricted':
                    mask = (np.linalg.norm(amp2, axis=1) > 1e-5)[:, None]
                    amp_psd = amp_psd * mask
                sigmaGW = prefactor * np.sum((amp2**2) * denom)
                sigmaPSDI = 0.25 * prefactor * np.sum(amp_psd * amp_psd * denom)

                if vertex_mode in ('PSD1', 'PSD2', 'PSD4'):
                    val = sigmaPSDI
                elif vertex_mode == 'PSD5':
                    val = 0.5 * (0.75 * sigmaGW + sigmaPSDI)
                elif vertex_mode in ('PSD6', 'PSD7'):
                    val = 0.5 * (sigmaGW + sigmaPSDI)
                else:  # PSD8, PSD9
                    amp_eff = amp * mask if self.spin_mode == 'unrestricted' else amp
                    sigmaTs = 0.5 * prefactor * np.sum(amp_eff * amp_eff * denom)
                    val = 0.5 * (sigmaGW + sigmaPSDI + sigmaTs)

            if eigenvalues_casida_t is not None and chiXYb_t is not None:
                if self.spin_mode == 'unrestricted':
                    omega_t_ba, omega_t_ab = eigenvalues_casida_t
                    chi_t_ba, chi_t_ab = chiXYb_t

                    def compute_channel_sigmaTt(omega_t, amp_t_raw):
                        if amp_t_raw.ndim == 2:
                            amp_t = amp_t_raw
                        else:
                            amp_t = amp_t_raw[:, :, p_state]
                        denom_t = self._denom_grid(w, eps, nocc_spin, omega_t, calc_imag)
                        return (0.5 * prefactor) * np.sum((amp_t**2) * denom_t)

                    sigmaTt_ba = compute_channel_sigmaTt(omega_t_ba, chi_t_ba)
                    sigmaTt_ab = compute_channel_sigmaTt(omega_t_ab, chi_t_ab)
                    if vertex_mode == 'PSD2':
                        val += 1.0 * (sigmaTt_ba + sigmaTt_ab)
                    elif vertex_mode == 'PSD4':
                        val = 0.5 * val + 0.5 * (sigmaTt_ba + sigmaTt_ab)
                    elif vertex_mode == 'PSD7':
                        val += 0.5 * (sigmaTt_ba + sigmaTt_ab)
                    elif vertex_mode == 'PSD9':
                        val += 1.0 * (sigmaTt_ba + sigmaTt_ab)
                else:
                    if chiXYb_t.ndim == 2:
                        amp_t = chiXYb_t
                    else:
                        amp_t = chiXYb_t[:, :, p_state]
                    denom_t = self._denom_grid(w, eps, nocc_spin, eigenvalues_casida_t, calc_imag)
                    sigmaTt = (0.5 * prefactor) * np.sum((amp_t**2) * denom_t)

                    if vertex_mode == 'PSD2':
                        val += 1.5 * sigmaTt
                    elif vertex_mode == 'PSD4':
                        val = 0.5 * val + 0.75 * sigmaTt
                    elif vertex_mode == 'PSD7':
                        val += 0.75 * sigmaTt
                    elif vertex_mode == 'PSD9':
                        val += 1.5 * sigmaTt

            sigma_grid.append(val)

        if is_scalar:
            return sigma_grid[0]
        else:
            return np.array(sigma_grid)

    def calculate_self_energy_matrix(self, nocc, eigenvalues_casida, chiXYa, chiXYb=None, eigenvalues=None, vertex_mode='GW'):
        """Full self-energy matrix at the QP energies (eigenvalues), symmetrized."""
        if eigenvalues is None:
            eigenvalues = self.eps

        if self.spin_mode == 'unrestricted':
            eps = eigenvalues
            nocc_spin = nocc
            prefactor = 0.5
        else:
            eps = eigenvalues
            nocc_spin = nocc
            prefactor = 1.0

        nmo = len(eps)
        nexciton = len(eigenvalues_casida)

        sign = np.zeros(nmo)
        for i in range(nmo):
            sign[i] = 1.0 if i < nocc_spin else -1.0

        eps_p = eps[None, None, :]
        eps_r = eps[None, :, None]
        sign_r_omega_S = (sign[:, None] * eigenvalues_casida[None, :]).T[:, :, None]

        energy = eps_p - eps_r + sign_r_omega_S
        denom = energy / (energy**2 + self.eta**2)

        if vertex_mode == 'GW' or chiXYb is None:
            chi_T = chiXYa * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)
        elif vertex_mode == 'GWGammaInf':
            chi_T = (chiXYa - 0.5 * chiXYb) * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)
        elif vertex_mode in ['PSD1', 'PSD2', 'PSD4']:
            chi_T = (2.0 * chiXYa - chiXYb) * denom
            tmp = 0.5 * np.einsum('Srq, Srp -> qp', chi_T, chi_T)
        elif vertex_mode == 'PSD5':
            tmp_gw = np.einsum('Srq, Srp -> qp', chiXYa, chiXYa * denom)
            amp_psd = 2.0 * chiXYa - chiXYb
            tmp_psdi = 0.5 * np.einsum('Srq, Srp -> qp', amp_psd, amp_psd * denom)
            tmp = 0.5 * (tmp_gw + tmp_psdi)
        else:
            chi_T = chiXYa * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)

        self_energy_matrix = prefactor * (tmp + tmp.T)
        return self_energy_matrix

    def calculate_self_energy_diagonal_batch(self, freq, nocc, eigenvalues_casida, chiXYa,
                                              chiXYb=None, spin_channel='alpha', vertex_mode='GW',
                                              calc_imag=False):
        """Vectorized batch counterpart to calculate_self_energy: Sigma_pp(freq[p]) for every p at once (freq is length-norb)."""
        eps = self.eps
        nocc_spin = nocc
        prefactor = 2.0
        norb = len(eps)

        sign = np.where(np.arange(norb) < nocc_spin, 1.0, -1.0)
        sign_r_omega_S = (sign[:, None] * eigenvalues_casida[None, :]).T[:, :, None]
        energy = freq[None, None, :] - eps[None, :, None] + sign_r_omega_S
        if calc_imag:
            denom = -sign[None, :, None] * self.eta / (energy**2 + self.eta**2)
        else:
            denom = energy / (energy**2 + self.eta**2)

        amp2 = chiXYa
        if vertex_mode == 'GW' or chiXYb is None:
            val = prefactor * np.einsum('Srp,Srp->p', amp2**2, denom)
        elif vertex_mode == 'GWGammaInf':
            amp_L = amp2 - 0.5 * chiXYb
            val = prefactor * np.einsum('Srp,Srp->p', amp2 * amp_L, denom)
        else:
            amp_psd = 2.0 * amp2 - chiXYb
            sigmaGW = prefactor * np.einsum('Srp,Srp->p', amp2**2, denom)
            sigmaPSDI = 0.25 * prefactor * np.einsum('Srp,Srp->p', amp_psd**2, denom)
            if vertex_mode in ('PSD1', 'PSD2', 'PSD4'):
                val = sigmaPSDI
            elif vertex_mode == 'PSD5':
                val = 0.5 * (0.75 * sigmaGW + sigmaPSDI)
            elif vertex_mode in ('PSD6', 'PSD7'):
                val = 0.5 * (sigmaGW + sigmaPSDI)
            else:
                sigmaTs = 0.5 * prefactor * np.einsum('Srp,Srp->p', chiXYb**2, denom)
                val = 0.5 * (sigmaGW + sigmaPSDI + sigmaTs)
        return val

    def calculate_sigma_hx(self, mol, mf, dm, mo_coeff):
        """HF contribution beyond kinetic/external energy (Sigma_Hx = V_H + V_x) in the MO basis, for a given density matrix."""
        V_Hx = mf.get_veff(mol, dm)
        if self.spin_mode == 'unrestricted':
            mo_coeff_a, mo_coeff_b = mo_coeff
            V_Hx_a, V_Hx_b = V_Hx
            V_Hx_mo_a = mo_coeff_a.T @ V_Hx_a @ mo_coeff_a
            V_Hx_mo_b = mo_coeff_b.T @ V_Hx_b @ mo_coeff_b
            return V_Hx_mo_a, V_Hx_mo_b
        else:
            V_Hx_mo = mo_coeff.T @ V_Hx @ mo_coeff
            return V_Hx_mo

    def solve_quasiparticle_energy(self, p_state, nocc, vertex_mode='GW', W_rpa=None, w_aux=None, triplet=False, solver_mode='newton', mol=None, mf=None):
        """Solve the QP equation for a target MO state (Newton or graphical solver_mode; DFT/HF via xc_correction)."""

        if self.spin_mode == 'unrestricted':
            raise NotImplementedError("High-level solve_quasiparticle_energy is currently only implemented for restricted spin.")

        method_info = get_method_info(vertex_mode)

        eps = self.eps
        eri = self.eri_chemist
        coeff = self.df_coeff
        norb = len(eps)

        xc_correction = 0.0
        if mf is not None and mol is not None and hasattr(mf, 'xc'):
            from pyscf import scf, dft
            dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
            V_Hxc = mf.get_veff(mol, dm)
            if isinstance(mf, (scf.uhf.UHF, dft.uks.UKS)):
                mf_hf = scf.UHF(mol)
            else:
                mf_hf = scf.RHF(mol)

            V_Hxc_mo = mf.mo_coeff.T @ V_Hxc @ mf.mo_coeff
            V_Hx_mo = self.calculate_sigma_hx(mol, mf_hf, dm, mf.mo_coeff)
            xc_correction = V_Hx_mo[p_state, p_state] - V_Hxc_mo[p_state, p_state]

        lr_solver = LinearResponseSolver(eps, coeff_df=coeff, eri_chemist=eri, spin_mode=self.spin_mode, eta=self.eta)

        if w_aux is None:
            w_aux = lr_solver.static_screening_aux(nocc)

        if not method_info['force_rpa_casida']:
            if W_rpa is None:
                if coeff is not None:
                    W_rpa = w_aux
                else:
                    occ, virt = lr_solver._get_occ_virt_indices(eps, nocc)
                    n_pair = len(occ) * len(virt)
                    d = np.array([eps[a] - eps[i] for i in occ for a in virt])
                    f = lr_solver._get_f_rpa(d, 0.0, is_imaginary=False)
                    chi0 = np.diag(2.0 * f)

                    V_trans = eri[np.ix_(occ, virt, occ, virt)].reshape(n_pair, n_pair)
                    chi = chi0 @ np.linalg.inv(np.eye(n_pair) - V_trans @ chi0)

                    eri_ov = eri[np.ix_(occ, virt)].reshape(n_pair, norb, norb)
                    tmp = chi @ eri_ov.reshape(n_pair, -1)
                    W_rpa = eri + np.einsum('Spq, Srs -> pqrs', eri_ov, tmp.reshape(n_pair, norb, norb))

        need_triplet = False
        if method_info['force_rpa_casida'] and w_aux is None:
            A_rpa, B_rpa = lr_solver.build_casida_matrices(nocc, lBSE=False)
            omega, X, Y = CasidaSolver(A_rpa, B_rpa).solve()
            chi_a = self.get_chi_a(nocc, X, Y, p_state=p_state)

            func = lambda w: w - eps[p_state] - xc_correction - self.calculate_self_energy(
                p_state, w, nocc, omega, chi_a, None, vertex_mode='GW'
            )
        else:
            A_s, B_s = lr_solver.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux, triplet=False)
            omega_s, X_s, Y_s = CasidaSolver(A_s, B_s).solve()
            chi_a_s = self.get_chi_a(nocc, X_s, Y_s, p_state=p_state)
            chi_b_s_vertex = self.get_chi_b_vertex(nocc, X_s, Y_s, eri_w=W_rpa, p_state=p_state)

            if not method_info['needs_vertex']:
                func = lambda w: w - eps[p_state] - xc_correction - self.calculate_self_energy(
                    p_state, w, nocc, omega_s, chi_a_s, None, vertex_mode='GW'
                )
            elif vertex_mode == 'GWGammaInf':
                func = lambda w: w - eps[p_state] - xc_correction - self.calculate_self_energy(
                    p_state, w, nocc, omega_s, chi_a_s, chi_b_s_vertex, vertex_mode=vertex_mode
                )
            else:
                need_triplet = triplet or method_info['needs_triplet']
                if need_triplet:
                    A_t, B_t = lr_solver.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux, triplet=True)
                    omega_t, X_t, Y_t = CasidaSolver(A_t, B_t).solve()
                    chi_b_t_vertex = self.get_chi_b_vertex(nocc, X_t, Y_t, eri_w=W_rpa, p_state=p_state)

                    func = lambda w: w - eps[p_state] - xc_correction - self.calculate_self_energy(
                        p_state, w, nocc, omega_s, chi_a_s, chi_b_s_vertex,
                        eigenvalues_casida_t=omega_t, chiXYb_t=chi_b_t_vertex,
                        vertex_mode=vertex_mode
                    )
                else:
                    func = lambda w: w - eps[p_state] - xc_correction - self.calculate_self_energy(
                        p_state, w, nocc, omega_s, chi_a_s, chi_b_s_vertex,
                        vertex_mode=vertex_mode
                    )

        if solver_mode == 'graphical':
            from src.Solvers.qp_equation import solve_qp_equation_graphical
            qp = solve_qp_equation_graphical(func, eps[p_state])
        else:
            qp = solve_qp_equation_newton(func, eps[p_state])

        return qp

    def calculate_spectral_function(self, p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb=None,
                                    eigenvalues_casida_t=None, chiXYb_t=None, spin_channel='alpha', vertex_mode='GW',
                                    V_xc_mo=0.0):
        """Spectral function A(omega) on a frequency grid for a target MO state."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
        else:
            eps = self.eps

        eps_p = eps[p_state]

        sigma_re = self.calculate_self_energy(
            p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb,
            eigenvalues_casida_t, chiXYb_t, spin_channel, vertex_mode, calc_imag=False
        )

        sigma_im = self.calculate_self_energy(
            p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb,
            eigenvalues_casida_t, chiXYb_t, spin_channel, vertex_mode, calc_imag=True
        )

        denom = (omega_grid - eps_p - sigma_re - V_xc_mo)**2 + sigma_im**2
        spectral_function = -sigma_im / (np.pi * denom)

        return spectral_function, sigma_re, sigma_im

import os
import sys
import numpy as np
from pyscf import gto, scf, df

from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients, get_two_electron_integrals_chemist
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.Solvers.qp_equation import solve_qp_equation
from src.Base.constants import get_method_info, DEFAULT_BROADENING_ETA

def calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True, eta=DEFAULT_BROADENING_ETA, state='homo', spin_channel='alpha', printSpectralFunction=False, dm_ccsd=None, dm_correction=None, tda=False, qp_solver='graphical'):
    """Quasiparticle energies for single-reference states.

    selfenergy: 'GW'/'GWGammaInf'/'PSD1'.../'PSD9' or a list of these.
    polarizability: 'RPA' (Hartree-only), 'BSE' (RPA-screened exchange), or
    'TDHF' (bare exchange) -- screening for the Casida states feeding the
    self-energy (unless force_rpa_casida overrides it, e.g. plain 'GW').
    Vertex-correction screening (GWGammaInf/PSDn) always uses the static RPA W.
    tda: solve every Casida problem in the Tamm-Dancoff approximation (Y=0);
    the static screening W_aux itself is unaffected.
    state: 'homo', int index, or list of indices.
    dm_correction: optional AO 1RDM in place of the mean-field density for the
    static Sigma_Hx term (e.g. a CCSD 1RDM or compute_gw_density_matrix output).
    dm_ccsd: deprecated alias for dm_correction.
    qp_solver: root selection for the QP equation. 'graphical' (default) returns
    the root nearest the mean-field eigenvalue; 'pole_strength' returns the root
    carrying the largest weight Z, which is what deep valence/semicore states
    need -- there a Z~0.03 satellite can sit closer to eps than the real QP pole.
    The two agree wherever only one root exists (e.g. most HOMOs).
    """
    if dm_correction is None:
        dm_correction = dm_ccsd
    mol = mf.mol
    is_uhf = isinstance(mf, scf.uhf.UHF)

    if is_uhf:
        nocc = mf.nelec  # (nocc_a, nocc_b)
        nocc_spin = nocc[0] if spin_channel == 'alpha' else nocc[1]
    else:
        nocc = mol.nelectron // 2
        nocc_spin = nocc

    eps = get_orbital_energies(mf, representation='spatial')

    if state == 'homo':
        p_state = nocc_spin - 1
        states = [p_state]
    elif isinstance(state, list):
        states = state
    elif isinstance(state, int):
        states = [state]
    else:
        raise ValueError(f"Invalid state parameter: {state}")

    if df:
        df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        if is_uhf:
            df_a, df_b = df_coeff
            mo_a, mo_b = mf.mo_coeff
            S_AO = mol.intor_symmetric('int1e_ovlp')
            S_ab = mo_a.T @ S_AO @ mo_b
            df_ab = np.einsum('ia, pik -> pka', S_ab, df_a)
            df_coeff = (df_a, df_b, df_ab)
        eri = None
    else:
        df_coeff = None
        eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')

    spin_mode = 'unrestricted' if is_uhf else 'restricted'
    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, eri_chemist=eri, spin_mode=spin_mode, eta=eta)
    se_solver = SelfEnergySolver(eps, df_coeff=df_coeff, eri_chemist=eri, spin_mode=spin_mode, eta=eta)

    w_aux = lr_solver.static_screening_aux(nocc)

    polarizability_mode = polarizability.upper()
    if polarizability_mode not in ('RPA', 'BSE', 'TDHF'):
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose 'RPA', 'BSE', or 'TDHF'.")
    lBSE_flag = polarizability_mode != 'RPA'
    # BSE screens exchange with the static RPA W; TDHF uses bare exchange (W_aux=None).
    w_aux_for_casida = w_aux if polarizability_mode == 'BSE' else None

    # S_z=0 (spin-conserved) excitations -- singlet-only for closed-shell.
    A_s, B_s = lr_solver.build_casida_matrices(nocc, lBSE=lBSE_flag, W_aux=w_aux_for_casida, triplet=False)
    omega_s, X_s, Y_s = CasidaSolver(A_s, B_s).solve(tda=tda)

    if isinstance(selfenergy, list):
        methods = selfenergy
    else:
        methods = [selfenergy]

    # Fail fast on typos/unimplemented methods rather than silently falling back to GW.
    method_infos = {m: get_method_info(m) for m in methods}

    need_triplet = any(method_infos[m]['needs_triplet'] for m in methods)
    if need_triplet:
        if is_uhf:
            W_aux_sf = w_aux_for_casida if df else None
            A_t_ba, B_t_ba = lr_solver.build_spin_flip_casida_matrices(nocc, lBSE=lBSE_flag, W_aux=W_aux_sf, channel='ba')
            omega_t_ba, X_t_ba, Y_t_ba = CasidaSolver(A_t_ba, B_t_ba).solve(tda=tda)

            A_t_ab, B_t_ab = lr_solver.build_spin_flip_casida_matrices(nocc, lBSE=lBSE_flag, W_aux=W_aux_sf, channel='ab')
            omega_t_ab, X_t_ab, Y_t_ab = CasidaSolver(A_t_ab, B_t_ab).solve(tda=tda)

            omega_t = (omega_t_ba, omega_t_ab)
            X_t = (X_t_ba, X_t_ab)
            Y_t = (Y_t_ba, Y_t_ab)
        else:
            A_t, B_t = lr_solver.build_casida_matrices(nocc, lBSE=lBSE_flag, W_aux=w_aux_for_casida, triplet=True)
            omega_t, X_t, Y_t = CasidaSolver(A_t, B_t).solve(tda=tda)
    else:
        omega_t, X_t, Y_t = None, None, None

    if df:
        eri_w_singlet = w_aux
        eri_w_triplet = w_aux
    else:
        eri_w_singlet, eri_w_triplet = lr_solver.construct_4d_w_rpa(nocc, spin_channel)

    results = {}
    for p_state in states:
        results[p_state] = {}

        chi_a_s = se_solver.get_chi_a(nocc, X_s, Y_s, spin_channel=spin_channel, p_state=p_state)

        # RPA-screened amplitude, needed by methods with force_rpa_casida (e.g. plain GW).
        if any(method_infos[m]['force_rpa_casida'] for m in methods):
            if not lBSE_flag:
                omega_rpa_calc, X_rpa_calc, Y_rpa_calc = omega_s, X_s, Y_s
            else:
                A_rpa, B_rpa = lr_solver.build_casida_matrices(nocc, lBSE=False)
                omega_rpa_calc, X_rpa_calc, Y_rpa_calc = CasidaSolver(A_rpa, B_rpa).solve(tda=tda)
            chi_a_rpa = se_solver.get_chi_a(nocc, X_rpa_calc, Y_rpa_calc, spin_channel=spin_channel, p_state=p_state)
        else:
            omega_rpa_calc, chi_a_rpa = None, None

        for method in methods:
            info = method_infos[method]

            omega_val = omega_rpa_calc if info['force_rpa_casida'] and lBSE_flag else omega_s
            chi_a_val = chi_a_rpa if info['force_rpa_casida'] and lBSE_flag else chi_a_s

            if not info['needs_vertex']:
                chi_b_val = None
                omega_t_val = None
                chi_b_t_val = None
            else:
                chi_b_val = se_solver.get_chi_b_vertex(nocc, X_s, Y_s, spin_channel=spin_channel, eri_w=eri_w_singlet, p_state=p_state)
                if info['needs_triplet']:
                    if is_uhf:
                        omega_t_ba, omega_t_ab = omega_t
                        X_t_ba, X_t_ab = X_t
                        Y_t_ba, Y_t_ab = Y_t
                        if df:
                            chi_b_t_ba = se_solver.get_chi_b_vertex_sf_df(nocc, X_t_ba, Y_t_ba, spin_channel=spin_channel, channel='ba', eri_w=eri_w_triplet, p_state=p_state)
                            chi_b_t_ab = se_solver.get_chi_b_vertex_sf_df(nocc, X_t_ab, Y_t_ab, spin_channel=spin_channel, channel='ab', eri_w=eri_w_triplet, p_state=p_state)
                        else:
                            chi_b_t_ba = se_solver.get_chi_b_vertex_sf_full(nocc, X_t_ba, Y_t_ba, spin_channel=spin_channel, channel='ba', eri_w=eri_w_triplet, p_state=p_state)
                            chi_b_t_ab = se_solver.get_chi_b_vertex_sf_full(nocc, X_t_ab, Y_t_ab, spin_channel=spin_channel, channel='ab', eri_w=eri_w_triplet, p_state=p_state)
                        chi_b_t_val = (chi_b_t_ba, chi_b_t_ab)
                    else:
                        chi_b_t_val = se_solver.get_chi_b_vertex(nocc, X_t, Y_t, spin_channel=spin_channel, eri_w=eri_w_triplet, p_state=p_state)
                    omega_t_val = omega_t
                else:
                    chi_b_t_val = None
                    omega_t_val = None

            # DFT/HF mean-field xc, or a post-mean-field (CCSD/GW) density correction if given.
            xc_correction = 0.0
            if hasattr(mf, 'xc') or dm_correction is not None:
                from pyscf import dft
                dm_mf = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
                dm_for_hx = dm_correction if dm_correction is not None else dm_mf
                V_Hxc = mf.get_veff(mol, dm_mf)
                if is_uhf:
                    mf_hf = scf.UHF(mol)
                    mo_a, mo_b = mf.mo_coeff
                    V_Hxc_a, V_Hxc_b = V_Hxc
                    V_Hxc_mo = mo_a.T @ V_Hxc_a @ mo_a if spin_channel == 'alpha' else mo_b.T @ V_Hxc_b @ mo_b
                else:
                    mf_hf = scf.RHF(mol)
                    V_Hxc_mo = mf.mo_coeff.T @ V_Hxc @ mf.mo_coeff

                V_Hx_mo = se_solver.calculate_sigma_hx(mol, mf_hf, dm_for_hx, mf.mo_coeff)
                if is_uhf:
                    V_Hx_mo_p = V_Hx_mo[0] if spin_channel == 'alpha' else V_Hx_mo[1]
                else:
                    V_Hx_mo_p = V_Hx_mo
                xc_correction = V_Hx_mo_p[p_state, p_state] - V_Hxc_mo[p_state, p_state]

            eps_spin = eps[0] if is_uhf and spin_channel == 'alpha' else (eps[1] if is_uhf else eps)
            eigKS = eps_spin[p_state]
            func = lambda w: w - eigKS - xc_correction - se_solver.calculate_self_energy(
                p_state, w, nocc, omega_val, chi_a_val, chi_b_val,
                eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
                spin_channel=spin_channel,
                vertex_mode=info['vertex_mode'], calc_imag=False
            )
            qp_ev = solve_qp_equation(func, eigKS, method=qp_solver) * 27.2114
            results[p_state][method] = qp_ev

            if printSpectralFunction:
                print(f"\nSpectral Function for State {p_state} ({method}):", flush=True)
                print(f"{'omega (eV)':>12s} | {'A(omega)':>12s} | {'Re Sigma (eV)':>14s} | {'Im Sigma (eV)':>14s}", flush=True)
                print("-" * 60, flush=True)
                qp_ha = qp_ev / 27.2114
                omega_grid = np.linspace(qp_ha - 0.5, qp_ha + 0.5, 50)

                spec, sig_re, sig_im = se_solver.calculate_spectral_function(
                    p_state, omega_grid, nocc, omega_val, chi_a_val, chi_b_val,
                    eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
                    spin_channel=spin_channel,
                    vertex_mode=info['vertex_mode']
                )

                for w_val, spec_val, re_val, im_val in zip(omega_grid, spec, sig_re, sig_im):
                    print(f"{w_val * 27.2114:12.6f} | {spec_val:12.6f} | {re_val * 27.2114:14.6f} | {im_val * 27.2114:14.6f}", flush=True)

    if not isinstance(state, list) and not isinstance(selfenergy, list):
        return results[states[0]][methods[0]]
    return results

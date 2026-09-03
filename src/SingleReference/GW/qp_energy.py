"""Quasiparticle energies for single-reference states.

`calc_qp_energy` is the front end. Three routes produce Sigma_c and all solve
the same equation w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w):

  casida         diagonalize (A, B), build W from the spectrum.  O(N^6), and the
                 only route carrying vertex corrections or a CC polarizability.
  imagfrequency  chi0(i.omega) by direct particle-hole summation.  O(N^4).
  space-time     chi0(i.tau) from a separable ISDF factorization.  O(N^3).
"""
import numpy as np
from pyscf import scf

from src.Base.constants import (get_method_info, DEFAULT_BROADENING_ETA,
                                HARTREE_TO_EV)
from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_density_fitting_coefficients,
                                      get_two_electron_integrals_chemist)
from src.Base.solvent_screening import solvent_static_selfenergy
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.cc_polarizability import GWCCSelfEnergy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
from src.Solvers.qp_equation import solve_qp_equation

IMAGINARY_AXIS_MODES = ('imagfrequency', 'imag-frequency', 'space-time')


def calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True,
                   eta=DEFAULT_BROADENING_ETA, state='homo',
                   spin_channel='alpha', printSpectralFunction=False,
                   dm_correction=None, tda=False, qp_solver='pole_strength',
                   nroots=8, mode='casida', **route_kwargs):
    """Quasiparticle energies, in eV.

    selfenergy:     'GW'/'GWGammaInf'/'PSD1'...'PSD9', or a list of these.
    polarizability: screening of the Casida states feeding Sigma -- 'RPA'
                    (Hartree only), 'BSE' (RPA-screened exchange) or 'TDHF'
                    (bare exchange). Methods with force_rpa_casida, plain GW
                    among them, always use RPA; vertex screening always uses the
                    static RPA W. 'CCSD'/'CCSDT' replaces the Casida
                    polarizability with an EOM-CC one (G0W@CC, Lewis and
                    Berkelbach), a separate spin-orbital path in cc_polarizability.py.
    mode:           how Sigma_c is built; see the module docstring. The two
                    imaginary-axis routes implement GW@RPA only and reject every
                    other combination rather than silently downgrading it.
    state:          'homo', an orbital index, or a list of them.
    dm_correction:  AO 1RDM used in place of the mean-field density in the
                    static Sigma_Hx term, e.g. a CCSD or GW 1RDM.
    tda:            solve every Casida problem with Y = 0; the static screening
                    W_aux is unaffected.
    nroots:         EE states in the Lehmann sum, CC polarizability only.
    qp_solver:      root selection. 'pole_strength' (default) returns the root
                    of largest weight Z, which deep valence and semicore states
                    need -- a Z ~ 0.03 satellite can sit closer to eps than the
                    quasiparticle. 'graphical' returns the root nearest eps;
                    they agree wherever only one root exists. 'newton' and
                    'bisection' are also accepted.
    """
    mol = mf.mol
    mode_key = str(mode).lower().replace('_', '-')
    if mode_key in IMAGINARY_AXIS_MODES:
        return _qp_energy_imaginary_axis_route(
            mf, mol, mode_key, selfenergy, polarizability, df, state,
            spin_channel, qp_solver, dm_correction, tda, route_kwargs)
    if mode_key != 'casida':
        raise ValueError(
            f"mode='{mode}'; choose 'casida', 'imagfrequency' or 'space-time'.")
    if route_kwargs:
        raise TypeError(
            f"unexpected keyword(s) {sorted(route_kwargs)} for mode='casida'")

    is_uhf = isinstance(mf, scf.uhf.UHF)
    nocc = mf.nelec if is_uhf else mol.nelectron // 2
    nocc_spin = (nocc[0] if spin_channel == 'alpha' else nocc[1]) if is_uhf else nocc
    eps = get_orbital_energies(mf, representation='spatial')
    states = _resolve_states(state, nocc_spin)

    # Static COHSEX reaction field of an attached solvent, first order in
    # vtilde. Sigma_c is second order in it, so this term carries the
    # polarization energy. None in the gas phase.
    sigma_solvent = solvent_static_selfenergy(mf, mol)
    if isinstance(sigma_solvent, tuple):
        sigma_solvent = sigma_solvent[0 if spin_channel == 'alpha' else 1]

    if polarizability.upper() in ('CCSD', 'CCSDT'):
        results = _qp_energy_cc_polarizability(
            mf, states, polarizability.lower(), selfenergy, eta, nroots,
            sigma_solvent, spin_channel, qp_solver, is_uhf)
        if not isinstance(state, list) and not isinstance(selfenergy, list):
            return results[states[0]]['GW']
        return results

    methods = selfenergy if isinstance(selfenergy, list) else [selfenergy]
    # Fail fast on typos rather than silently falling back to GW.
    method_infos = {m: get_method_info(m) for m in methods}

    df_coeff, eri = _two_electron_integrals(mol, mf, df, is_uhf)
    spin_mode = 'unrestricted' if is_uhf else 'restricted'
    lr_solver = LinearResponseSolver(eps, coeff_df=df_coeff, eri_chemist=eri,
                                     spin_mode=spin_mode, eta=eta)
    se_solver = SelfEnergySolver(eps, df_coeff=df_coeff, eri_chemist=eri,
                                 spin_mode=spin_mode, eta=eta)
    w_aux = lr_solver.static_screening_aux(nocc)
    spectrum = _casida_spectrum(lr_solver, nocc, polarizability, w_aux, tda,
                                method_infos, methods, is_uhf, df)

    # A vertex contracts against W: the auxiliary form under DF, the explicit
    # four-index one otherwise.
    if df:
        eri_w_singlet = eri_w_triplet = w_aux
    else:
        eri_w_singlet, eri_w_triplet = lr_solver.construct_4d_w_rpa(nocc, spin_channel)

    eps_spin = (eps[0] if spin_channel == 'alpha' else eps[1]) if is_uhf else eps

    results = {}
    for p_state in states:
        results[p_state] = {}
        amps = _self_energy_amplitudes(se_solver, nocc, spectrum, method_infos,
                                       methods, spin_channel, p_state,
                                       eri_w_singlet, eri_w_triplet, is_uhf, df)
        xc_correction = _static_correction(mf, mol, se_solver, dm_correction,
                                           sigma_solvent, spin_channel,
                                           p_state, is_uhf)
        eigKS = eps_spin[p_state]

        for method in methods:
            info = method_infos[method]
            omega_val, chi_a_val, chi_b_val, omega_t_val, chi_b_t_val = amps[method]

            func = lambda w: w - eigKS - xc_correction - se_solver.calculate_self_energy(
                p_state, w, nocc, omega_val, chi_a_val, chi_b_val,
                eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
                spin_channel=spin_channel,
                vertex_mode=info['vertex_mode'], calc_imag=False)
            qp_ev = solve_qp_equation(func, eigKS,
                                      method=qp_solver) * HARTREE_TO_EV
            results[p_state][method] = qp_ev

            if printSpectralFunction:
                _print_spectral_function(se_solver, p_state, method, qp_ev, nocc,
                                         omega_val, chi_a_val, chi_b_val,
                                         omega_t_val, chi_b_t_val, spin_channel,
                                         info)

    if not isinstance(state, list) and not isinstance(selfenergy, list):
        return results[states[0]][methods[0]]
    return results


def _resolve_states(state, nocc_spin):
    """'homo', an index, or a list of them -> list of orbital indices."""
    if state == 'homo':
        return [nocc_spin - 1]
    if isinstance(state, list):
        return state
    if isinstance(state, int):
        return [state]
    raise ValueError(f"Invalid state parameter: {state}")


def _two_electron_integrals(mol, mf, df, is_uhf):
    """(df_coeff, eri): the three-index DF factors, or the full chemist ERI."""
    if not df:
        return None, get_two_electron_integrals_chemist(mol, mf,
                                                        representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    if is_uhf:
        # The alpha-beta block needs the overlap between the two MO sets.
        df_a, df_b = df_coeff
        mo_a, mo_b = mf.mo_coeff
        S_ab = mo_a.T @ mol.intor_symmetric('int1e_ovlp') @ mo_b
        df_coeff = (df_a, df_b, np.einsum('ia, pik -> pka', S_ab, df_a))
    return df_coeff, None


def _casida_spectrum(lr_solver, nocc, polarizability, w_aux, tda, method_infos,
                     methods, is_uhf, df):
    """Casida excitations feeding the self-energy, as (omega, X, Y) per channel.

    Holds the S_z = 0 solution; the triplet, or the two spin-flip channels when
    unrestricted, where a vertex needs it; and the RPA solution where a method
    forces RPA screening regardless of `polarizability`.
    """
    mode = polarizability.upper()
    if mode not in ('RPA', 'BSE', 'TDHF'):
        raise ValueError(f"Unknown polarizability '{polarizability}'; choose "
                         "'RPA', 'BSE', 'TDHF', 'CCSD', or 'CCSDT'.")
    lBSE = mode != 'RPA'
    # BSE screens exchange with the static RPA W; TDHF uses bare exchange.
    w_casida = w_aux if mode == 'BSE' else None

    A_s, B_s = lr_solver.build_casida_matrices(nocc, lBSE=lBSE, W_aux=w_casida,
                                               triplet=False)
    out = {'lBSE': lBSE, 'singlet': CasidaSolver(A_s, B_s).solve(tda=tda)}

    if any(method_infos[m]['needs_triplet'] for m in methods):
        if is_uhf:
            W_sf = w_casida if df else None
            channels = [CasidaSolver(*lr_solver.build_spin_flip_casida_matrices(
                            nocc, lBSE=lBSE, W_aux=W_sf, channel=c)).solve(tda=tda)
                        for c in ('ba', 'ab')]
            out['triplet'] = tuple(zip(*channels))
        else:
            A_t, B_t = lr_solver.build_casida_matrices(nocc, lBSE=lBSE,
                                                       W_aux=w_casida,
                                                       triplet=True)
            out['triplet'] = CasidaSolver(A_t, B_t).solve(tda=tda)
    else:
        out['triplet'] = (None, None, None)

    if lBSE and any(method_infos[m]['force_rpa_casida'] for m in methods):
        A_rpa, B_rpa = lr_solver.build_casida_matrices(nocc, lBSE=False)
        out['rpa'] = CasidaSolver(A_rpa, B_rpa).solve(tda=tda)
    else:
        out['rpa'] = out['singlet']
    return out


def _self_energy_amplitudes(se_solver, nocc, spectrum, method_infos, methods,
                            spin_channel, p_state, eri_w_singlet, eri_w_triplet,
                            is_uhf, df):
    """Per method, the (omega, chi_a, chi_b, omega_t, chi_b_t) Sigma is built from."""
    omega_s, X_s, Y_s = spectrum['singlet']
    omega_r, X_r, Y_r = spectrum['rpa']
    lBSE = spectrum['lBSE']

    chi_a_s = se_solver.get_chi_a(nocc, X_s, Y_s, spin_channel=spin_channel,
                                  p_state=p_state)
    force_rpa = lBSE and any(method_infos[m]['force_rpa_casida'] for m in methods)
    chi_a_rpa = (se_solver.get_chi_a(nocc, X_r, Y_r, spin_channel=spin_channel,
                                     p_state=p_state) if force_rpa else None)

    out = {}
    for method in methods:
        info = method_infos[method]
        use_rpa = info['force_rpa_casida'] and lBSE
        omega_val = omega_r if use_rpa else omega_s
        chi_a_val = chi_a_rpa if use_rpa else chi_a_s

        if not info['needs_vertex']:
            out[method] = (omega_val, chi_a_val, None, None, None)
            continue

        chi_b_val = se_solver.get_chi_b_vertex(nocc, X_s, Y_s,
                                               spin_channel=spin_channel,
                                               eri_w=eri_w_singlet,
                                               p_state=p_state)
        if not info['needs_triplet']:
            out[method] = (omega_val, chi_a_val, chi_b_val, None, None)
            continue

        omega_t, X_t, Y_t = spectrum['triplet']
        if is_uhf:
            get_sf = (se_solver.get_chi_b_vertex_sf_df if df
                      else se_solver.get_chi_b_vertex_sf_full)
            chi_b_t_val = tuple(
                get_sf(nocc, X_t[i], Y_t[i], spin_channel=spin_channel,
                       channel=channel, eri_w=eri_w_triplet, p_state=p_state)
                for i, channel in enumerate(('ba', 'ab')))
        else:
            chi_b_t_val = se_solver.get_chi_b_vertex(nocc, X_t, Y_t,
                                                     spin_channel=spin_channel,
                                                     eri_w=eri_w_triplet,
                                                     p_state=p_state)
        out[method] = (omega_val, chi_a_val, chi_b_val, omega_t, chi_b_t_val)
    return out


def _static_correction(mf, mol, se_solver, dm_correction, sigma_solvent,
                       spin_channel, p_state, is_uhf):
    """<Sigma_Hx - v_Hxc>_pp, plus the solvent reaction field.

    Zero on a Hartree-Fock reference with no density correction, where Sigma_Hx
    is already the mean-field potential.
    """
    xc_correction = 0.0
    if hasattr(mf, 'xc') or dm_correction is not None:
        beta = is_uhf and spin_channel != 'alpha'
        dm_mf = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        dm_for_hx = dm_correction if dm_correction is not None else dm_mf
        V_Hxc = mf.get_veff(mol, dm_mf)
        if is_uhf:
            mf_hf = scf.UHF(mol)
            mo = mf.mo_coeff[1 if beta else 0]
            V_Hxc_mo = mo.T @ V_Hxc[1 if beta else 0] @ mo
        else:
            mf_hf = scf.RHF(mol)
            V_Hxc_mo = mf.mo_coeff.T @ V_Hxc @ mf.mo_coeff
        V_Hx_mo = se_solver.calculate_sigma_hx(mol, mf_hf, dm_for_hx, mf.mo_coeff)
        if is_uhf:
            V_Hx_mo = V_Hx_mo[1 if beta else 0]
        xc_correction = V_Hx_mo[p_state, p_state] - V_Hxc_mo[p_state, p_state]
    if sigma_solvent is not None:
        xc_correction = xc_correction + sigma_solvent[p_state, p_state]
    return xc_correction


def _print_spectral_function(se_solver, p_state, method, qp_ev, nocc, omega_val,
                             chi_a_val, chi_b_val, omega_t_val, chi_b_t_val,
                             spin_channel, info):
    """A(omega) and Sigma on a window around the quasiparticle solution."""
    print(f"\nSpectral Function for State {p_state} ({method}):", flush=True)
    print(f"{'omega (eV)':>12s} | {'A(omega)':>12s} | {'Re Sigma (eV)':>14s} | "
          f"{'Im Sigma (eV)':>14s}", flush=True)
    print("-" * 60, flush=True)
    qp_ha = qp_ev / HARTREE_TO_EV
    omega_grid = np.linspace(qp_ha - 0.5, qp_ha + 0.5, 50)
    spec, sig_re, sig_im = se_solver.calculate_spectral_function(
        p_state, omega_grid, nocc, omega_val, chi_a_val, chi_b_val,
        eigenvalues_casida_t=omega_t_val, chiXYb_t=chi_b_t_val,
        spin_channel=spin_channel, vertex_mode=info['vertex_mode'])
    for w_val, spec_val, re_val, im_val in zip(omega_grid, spec, sig_re, sig_im):
        print(f"{w_val * HARTREE_TO_EV:12.6f} | {spec_val:12.6f} | "
              f"{re_val * HARTREE_TO_EV:14.6f} | "
              f"{im_val * HARTREE_TO_EV:14.6f}", flush=True)


def _qp_energy_cc_polarizability(mf, states, level, selfenergy, eta, nroots,
                                 sigma_solvent, spin_channel, qp_solver, is_uhf):
    """G0W@CC: the same QP equation, with Sigma_c from an EOM-CC Lehmann sum.

    A separate branch because none of the Casida machinery applies -- the CC
    route is spin-orbital, full-ERI, and takes its transition densities from
    EOM-CC rather than from X/Y.
    """
    methods = selfenergy if isinstance(selfenergy, list) else [selfenergy]
    if methods != ['GW']:
        raise NotImplementedError(
            f"CC polarizability screens the plain GW self-energy only, got {selfenergy}. "
            "The vertex-corrected self-energies (GWGammaInf/PSDn) are built from Casida "
            "amplitudes, which the EOM-CC route does not produce.")
    if is_uhf:
        raise NotImplementedError("CC polarizability is restricted (closed-shell RHF) only.")
    if hasattr(mf, 'xc'):
        raise NotImplementedError(
            "CC polarizability assumes an HF reference; a KS starting point would "
            "additionally need a v_xc correction, which this route does not build.")

    solver = GWCCSelfEnergy(mf, level=level, nroots=nroots)
    spin_offset = 0 if spin_channel == 'alpha' else 1

    results = {}
    for p_state in states:
        # Spatial orbital p -> spin orbital 2p (+1 for beta), interleaved.
        p_so = 2 * p_state + spin_offset
        shift = 0.0 if sigma_solvent is None else sigma_solvent[p_state, p_state]
        qp_ha = solver.solve_qp(p_so, eta=eta, method=qp_solver, static_shift=shift)
        results[p_state] = {'GW': qp_ha * HARTREE_TO_EV}
    return results


def _qp_energy_imaginary_axis_route(mf, mol, mode_key, selfenergy, polarizability,
                                    df, state, spin_channel, qp_solver,
                                    dm_correction, tda, route_kwargs):
    """Dispatch to the imaginary-frequency or space-time driver.

    Both implement GW@RPA on a restricted, density-fitted reference and nothing
    else, so every unsupported combination is rejected rather than quietly
    returning a GW@RPA number under another name.
    """
    if isinstance(selfenergy, list) or str(selfenergy).upper() != 'GW':
        raise ValueError(
            f"mode='{mode_key}' implements GW only, got selfenergy={selfenergy!r}. "
            f"Vertex corrections (GWGammaInf, PSDn) need the Casida route.")
    if str(polarizability).upper() != 'RPA':
        raise ValueError(
            f"mode='{mode_key}' implements RPA screening only, got "
            f"polarizability={polarizability!r}. BSE/TDHF/CCSD need mode='casida'.")
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(f"mode='{mode_key}' is restricted-spin only.")
    if not df:
        raise ValueError(f"mode='{mode_key}' requires density fitting (df=True).")
    if tda:
        raise ValueError(f"tda has no meaning for mode='{mode_key}'; it never "
                         f"forms a Casida problem.")

    nocc = mol.nelectron // 2
    states = _resolve_states(state, nocc)

    if mode_key == 'space-time':
        # One call for the whole window: chi0, the Dyson inversion, the tau
        # sweep and <Sigma_x - v_xc> are all shared across states.
        out = solve_qp_energy_space_time(mf, mol, nocc, np.asarray(states),
                                         solver_mode=qp_solver,
                                         dm_correction=dm_correction,
                                         **route_kwargs)
    else:
        out = [solve_qp_energy_imaginary_axis(mf, mol, nocc, p,
                                              solver_mode=qp_solver,
                                              dm_correction=dm_correction,
                                              **route_kwargs)
               for p in states]

    out = [e * HARTREE_TO_EV for e in out]
    return out[0] if not isinstance(state, list) else out

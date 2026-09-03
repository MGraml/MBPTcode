"""Pieces shared by every imaginary-axis quasiparticle route.

The imaginary-frequency and space-time routes differ ONLY in how they produce
Sigma_c sampled on the imaginary axis. Everything downstream of that -- the
static Sigma_x - v_xc term, the Pade continuation to the real axis, and the
root search on the QP equation -- is identical, so it lives here rather than
being duplicated in each driver.

A route is therefore expected to supply just `(z_fit, sigma_iw)` and hand off.

References
----------
Vidberg and Serene, J. Low Temp. Phys. 29, 179 (1977) -- Pade continuation of
a function sampled on the imaginary axis via Thiele's continued fraction, the
algorithm behind `Base/utils/analyticalContinuation.thiele_coefficients`.
"""
import numpy as np

from src.Base.solvent_screening import solvent_static_selfenergy
from src.Base.utils.analyticalContinuation import (greedy_pade_order,
                                                   thiele_coefficients,
                                                   pade_eval)
from src.Solvers.qp_equation import solve_qp_equation


def static_exchange_matrix(mf, mol, dm_correction=None):
    """
    <p| Sigma_x - v_xc |q> over the whole MO basis.
    Zero by construction on a Hartree-Fock reference
    dm_correction: optional AO 1RDM used in place of the
    mean-field density for Sigma_x only (a CCSD or GW density, say).

    An attached solvent screening adds its first-order reaction-field
    (static COHSEX) operator here: the static self-energy is where the
    polarization energy lives, Sigma_c being second order in vtilde, and
    this matrix is the one static object every imaginary-axis route shares.
    None attached (gas phase) adds nothing.
    """
    dm = mf.make_rdm1()
    dm_for_hx = dm if dm_correction is None else dm_correction
    sig_x = -0.5 * mf.get_k(mol, dm_for_hx)
    v_xc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
    mo = mf.mo_coeff
    out = mo.T @ (sig_x - v_xc) @ mo
    sigma_solvent = solvent_static_selfenergy(mf, mol)
    if sigma_solvent is not None:
        if isinstance(sigma_solvent, tuple):
            raise NotImplementedError(
                "solvent screening through static_exchange_matrix is "
                "restricted-only, like its consumers")
        out = out + sigma_solvent
    return out


def static_exchange_correction(mf, mol, p_state, dm_correction=None):
    """
    <p| Sigma_x - v_xc |p>:
    static_exchange_matrix for a single state.
    If you want several states, rather call static_exchange_matrix directly
    """
    return static_exchange_matrix(mf, mol, dm_correction=dm_correction)[p_state, p_state]


def solve_qp_from_imaginary_axis(eps, p_state, xc_correction, z_fit, sigma_iw,
                                 greedy=True, solver_mode='pole_strength',
                                 max_order=None):
    """Pade-continue Sigma_c off the imaginary axis and solve the QP equation.

        w = eps_p + <Sigma_x - v_xc>_pp + Re Sigma_c(w)

    eps           : KS orbitale energies
    p_state       : state of interest 
    xc_correction : static part of self-energy (corrected for DFT starting point)
    z_fit         : the sample points
    sigma_iw      : Sigma_c values on them
    greedy        : Default algorithm for analytical continuation
    solver_mode   : By default, look for solution with largest Z-factor.
    max_order     : cap on the number of Pade nodes, applied after the greedy
                    ordering. None keeps all of them, which is the T = 0 behaviour; pass
                    `matsubara.ir_continuation_order(beta, wmax)` for a metal.
    """
    z_ord, f_ord = z_fit, sigma_iw
    if greedy:
        order = greedy_pade_order(z_fit, sigma_iw)
        z_ord, f_ord = z_fit[order], sigma_iw[order]
    if max_order is not None and max_order < len(z_ord):
        # At finite temperature the imaginary-axis data can only resolve so
        # many independent structures and asking Pade for more nodes than
        # that is asking it to fit noise. Truncating AFTER the greedy ordering
        # keeps the most informative points, since that ordering puts them
        # first by construction.
        if max_order < 2:
            raise ValueError(
                f"max_order = {max_order} leaves too few points for a Pade "
                f"fit; at least two are needed.")
        z_ord, f_ord = z_ord[:max_order], f_ord[:max_order]
    pade_coeffs = thiele_coefficients(z_ord, f_ord)

    def residual(w):
        sigma_c_w = pade_eval(np.array([w], dtype=complex), z_ord, pade_coeffs)[0]
        return w - eps[p_state] - xc_correction - sigma_c_w.real

    return solve_qp_equation(residual, eps[p_state], method=solver_mode)


def imaginary_axis_sample_points(freq_points, nocc, p_state, mu):
    """The (z_fit, query) pair both routes sample on: the line Re z = mu, on the
    side of the gap the state sits on."""
    sign = -1.0 if p_state < nocc else 1.0
    iw_query = sign * np.asarray(freq_points)
    return mu + 1j * iw_query, iw_query

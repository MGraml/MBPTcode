import numpy as np

from src.Base.constants import DEFAULT_BROADENING_ETA
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid, minimax_supported_sizes
from src.Base.utils.analyticalContinuation import greedy_pade_order, thiele_coefficients, pade_eval
from src.Base.pyscf_interface import get_orbital_energies, get_density_fitting_coefficients
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.base import get_occ_virt_indices
from src.Solvers.qp_equation import solve_qp_equation_graphical, solve_qp_equation_newton


def solve_screening_imaginary_axis(lr_solver, nocc, freq_points):
    """W(i*omega_k) = [I - P(i*omega_k)]^-1 on the imaginary-frequency grid, in the whitened DF/RI-V auxiliary metric."""
    return lr_solver.solve_rpa_screening(freq_points, nocc, is_imaginary=True)


def self_energy_imaginary_axis(df_coeff, eps, nocc, p_state, freq_points, freq_weights,
                                W_grid, query_freqs):
    """Correlation self-energy Sigma_c,pp(i*query_freqs) by numerical convolution over the imaginary-frequency grid.

    Sigma_pp(i*w) = -(1/2pi) sum_m sum_k weight_k * [G_m(i*(w+wk)) + G_m(i*(w-wk))] * w_pp^m(i*wk),
    folded to 0..infty via W(i*omega)'s even parity. G_m(i*x) = 1/(i*x-eps_m);
    Wc = W - I is the correlation-only screened interaction (exchange handled
    separately via the static Sigma_x - v_xc term). Returns complex array of shape (len(query_freqs),).
    """
    Wc_grid = W_grid - np.eye(W_grid.shape[-1])[None, :, :]

    # w_pm^m(i*w_k) for all m, all grid points k: shape (nfreq, norb)
    Cp = df_coeff[:, p_state, :]          # (naux, norb)
    w_pm = np.einsum('Pm,kPQ,Qm->km', Cp, Wc_grid, Cp)

    query_freqs = np.atleast_1d(query_freqs)
    sigma = np.zeros(len(query_freqs), dtype=complex)
    for iq, w in enumerate(query_freqs):
        for k in range(len(freq_points)):
            wk = freq_points[k]
            Gp = 1.0 / (1j * (w + wk) - eps)     # (norb,)
            Gm = 1.0 / (1j * (w - wk) - eps)
            sigma[iq] += freq_weights[k] * np.sum((Gp + Gm) * w_pm[k])
    sigma *= -1.0 / (2.0 * np.pi)
    return sigma


def self_energy_matrix_imaginary_axis(df_coeff, eps, freq_points, freq_weights, W_grid,
                                       query_freqs=None):
    """Matrix-valued counterpart to self_energy_imaginary_axis: full Sigma_c,pq(i*query_freq) for every (p,q) at once.

    Vectorized as one matmul per (internal grid point, query point) pair,
    O(nfreq^2) total, no Python loop over norb. Diagonal validated against
    self_energy_imaginary_axis in tests/test_imaginary_axis_matrix.py.
    """
    naux, norb, _ = df_coeff.shape
    nfreq = len(freq_points)
    Wc_grid = W_grid - np.eye(naux)[None, :, :]
    if query_freqs is None:
        query_freqs = freq_points
    query_freqs = np.atleast_1d(query_freqs)
    nq = len(query_freqs)

    C_flat = df_coeff.reshape(naux, -1)
    sigma = np.zeros((nq, norb, norb), dtype=complex)
    for k in range(nfreq):
        wk = freq_points[k]
        D_k = (Wc_grid[k] @ C_flat).reshape(naux, norb, norb)
        for iq, w in enumerate(query_freqs):
            Gsum_m = 1.0 / (1j * (w + wk) - eps) + 1.0 / (1j * (w - wk) - eps)
            C_tilde = df_coeff * Gsum_m[None, :, None]
            sigma[iq] += freq_weights[k] * np.einsum('Qpm,Qmq->pq', D_k, C_tilde)
    sigma *= -1.0 / (2.0 * np.pi)
    return sigma


def solve_qp_energy_imaginary_axis(mf, mol, nocc, p_state, nfreq=20, w0=None, grid='minimax',
                                    eta=DEFAULT_BROADENING_ETA, solver_mode='graphical', greedy=True,
                                    dm_correction=None):
    """GW@RPA quasiparticle energy via the imaginary-frequency-axis route: RPA W(i*omega) -> convolution -> Pade continuation.

    Independent, non-Casida route to the same quantity as
    calc_qp_energy(selfenergy='GW', polarizability='RPA'); should agree to
    grid/continuation precision. Restricted spin, DF only.

    grid: 'minimax' (default, tabulated GreenX grid, prefer when nfreq in
    minimax_supported_sizes()) or 'gauss_legendre' (fallback, uses w0).
    dm_correction: optional AO 1RDM in place of mf's density for the static
    Sigma_x - v_xc term (same convention as calc_qp_energy).
    """
    from pyscf import scf

    eps = get_orbital_energies(mf, representation='spatial')
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=eta)

    if grid == 'minimax':
        occ, virt = get_occ_virt_indices(eps, nocc)
        e_min = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        freq_points, freq_weights = minimax_frequency_grid(nfreq, e_min, e_max)
    elif grid == 'gauss_legendre':
        if w0 is None:
            w0 = gap_scaled_w0(eps, nocc)
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    else:
        raise ValueError(f"Unknown grid '{grid}'; choose 'minimax' or 'gauss_legendre'.")
    W_grid = solve_screening_imaginary_axis(lr, nocc, freq_points)

    sign = -1.0 if p_state < nocc else 1.0
    iw_query = sign * freq_points
    sigma_iw = self_energy_imaginary_axis(df_coeff, eps, nocc, p_state, freq_points,
                                           freq_weights, W_grid, iw_query)
    z_fit = 1j * iw_query

    dm = mf.make_rdm1()
    dm_for_hx = dm_correction if dm_correction is not None else dm
    Sigx = -0.5 * mf.get_k(mol, dm_for_hx)
    vxc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
    mo = mf.mo_coeff
    xc_correction = (mo.T @ (Sigx - vxc) @ mo)[p_state, p_state]

    # Pade fit depends only on (z_fit, sigma_iw), not the trial frequency w --
    # build once here rather than inside func (called ~150+ times by the root solver).
    z_ord = z_fit
    f_ord = sigma_iw
    if greedy:
        order = greedy_pade_order(z_fit, sigma_iw)
        z_ord, f_ord = z_fit[order], sigma_iw[order]
    pade_coeffs = thiele_coefficients(z_ord, f_ord)

    def func(w):
        sigma_c_w = pade_eval(np.array([w], dtype=complex), z_ord, pade_coeffs)[0]
        return w - eps[p_state] - xc_correction - sigma_c_w.real

    if solver_mode == 'graphical':
        return solve_qp_equation_graphical(func, eps[p_state])
    else:
        return solve_qp_equation_newton(func, eps[p_state])

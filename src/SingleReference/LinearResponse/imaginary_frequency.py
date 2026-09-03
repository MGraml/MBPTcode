"""RPA polarizability and screened interaction on the IMAGINARY-FREQUENCY axis,
by direct summation of particle-hole bubbles.

    chi0(i.w) = 2 (C_ov f) C_ov^T,   f(i.w) = -2 d/(d^2 + w^2),   W = [1-chi0]^-1

The whole occupied-virtual pair axis is carried through every contraction, so a
frequency point costs O(naux^2 n_occ n_vir): the N^4 route.

Its two peers produce the same W. `casida.py` diagonalizes the full (A, B)
problem and also yields the excitations themselves, at O(n_ov^3). `space_time.py`
factorizes the bubble over a separable ERI and costs O(M^2 (n_occ + n_vir)).
Casida and this module agree to the frequency-quadrature error; this module and
the space-time route agree to the accuracy of the factorization.

Free functions taking a `LinearResponseSolver` as first argument, so the three
routes stay separable; the solver's methods delegate here.

References
----------
Kaltak, Klimes and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) -- the
minimax imaginary-frequency quadrature that makes a handful of points enough
here (grid tables in `src/Base/utils/grids.py`).
"""
import numpy as np

from src.SingleReference.base import get_occ_virt_indices


def _f_rpa(lr, d, w, is_imaginary):
    """Frequency factor of the non-interacting particle-hole propagator.

    On the imaginary axis, -2d/(d^2 + w^2): minus the cosine transform of
    e^{-d tau}, the identity `space_time.py` runs the other way.
    """
    if is_imaginary:
        return -2.0 * d / (d**2 + w**2)
    else:
        if np.abs(w) < 1e-12:
            return -2.0 * d / (d**2 + lr.eta**2)
        else:
            return (w - d) / ((w - d)**2 + lr.eta**2) - (w + d) / ((w + d)**2 + lr.eta**2)


def solve_rpa_screening(lr, omega_grid, nocc, is_imaginary=False):
    """W(w), dispatched to the density-fitted or the full-ERI build."""
    if (lr.spin_mode == 'unrestricted' and lr.coeff_a is not None) or \
       (lr.spin_mode != 'unrestricted'
        and (lr.df_coeff is not None or lr.coeff_ov is not None)):
        return solve_rpa_screening_df(lr, omega_grid, nocc, is_imaginary)
    else:
        return solve_rpa_screening_full(lr, omega_grid, nocc, is_imaginary)


def solve_rpa_screening_df(lr, omega_grid, nocc, is_imaginary=False):
    """W(w) = [1 - chi0(w)]^-1 in the auxiliary basis, from the DF factors."""
    if lr.spin_mode == 'unrestricted':
        nocc_a, nocc_b = nocc
        occ_a, virt_a = get_occ_virt_indices(lr.eps_a, nocc_a)
        occ_b, virt_b = get_occ_virt_indices(lr.eps_b, nocc_b)
        
        d_a = (lr.eps_a[virt_a][None, :] - lr.eps_a[occ_a][:, None]).ravel()
        d_b = (lr.eps_b[virt_b][None, :] - lr.eps_b[occ_b][:, None]).ravel()
        
        C_ov_a = lr.coeff_a[:, occ_a[:, None], virt_a].reshape(lr.naux, -1)
        C_ov_b = lr.coeff_b[:, occ_b[:, None], virt_b].reshape(lr.naux, -1)
        
        W_grid = []
        for w in omega_grid:
            f_a = _f_rpa(lr, d_a, w, is_imaginary)
            f_b = _f_rpa(lr, d_b, w, is_imaginary)
            chi0 = (C_ov_a * f_a) @ C_ov_a.T + (C_ov_b * f_b) @ C_ov_b.T
            W_w = np.linalg.inv(np.eye(lr.naux) - chi0)
            W_grid.append(W_w)
        return np.array(W_grid)
    else:
        occ, virt = get_occ_virt_indices(lr.eps, nocc)
        d = (lr.eps[virt][None, :] - lr.eps[occ][:, None]).ravel()
        # Prefer the standalone ov block: slicing it out of a full
        # (naux, norb, norb) coeff_df forces the caller to build that tensor.
        C_ov = (lr.coeff_ov if lr.coeff_ov is not None
                else lr.df_coeff[:, occ[:, None], virt].reshape(lr.naux, -1))
        # One reused scratch buffer for C_ov * f; allocating inside the loop
        # keeps C_ov and its scaled copy live at once, doubling the peak.
        scaled = np.empty_like(C_ov)
        W_grid = []
        eye = np.eye(lr.naux)
        for w in omega_grid:
            f = _f_rpa(lr, d, w, is_imaginary)
            np.multiply(C_ov, f, out=scaled)
            chi0 = 2.0 * scaled @ C_ov.T
            W_grid.append(np.linalg.inv(eye - chi0))
        return np.array(W_grid)


def solve_rpa_screening_full(lr, omega_grid, nocc, is_imaginary=False):
    """W(w) in the particle-hole pair basis, from the full ERI tensor."""
    if lr.spin_mode == 'unrestricted':
        nocc_a, nocc_b = nocc
        occ_a, virt_a = get_occ_virt_indices(lr.eps_a, nocc_a)
        occ_b, virt_b = get_occ_virt_indices(lr.eps_b, nocc_b)
        
        nocc_a_val, nvirt_a_val = len(occ_a), len(virt_a)
        nocc_b_val, nvirt_b_val = len(occ_b), len(virt_b)
        n_pair_a = nocc_a_val * nvirt_a_val
        n_pair_b = nocc_b_val * nvirt_b_val
        n_pair = n_pair_a + n_pair_b
        
        d_a = (lr.eps_a[virt_a][None, :] - lr.eps_a[occ_a][:, None]).ravel()
        d_b = (lr.eps_b[virt_b][None, :] - lr.eps_b[occ_b][:, None]).ravel()
        
        V_aa = lr.eri_a[np.ix_(occ_a, virt_a, occ_a, virt_a)].reshape(n_pair_a, n_pair_a)
        V_bb = lr.eri_b[np.ix_(occ_b, virt_b, occ_b, virt_b)].reshape(n_pair_b, n_pair_b)
        V_ab = lr.eri_ab[np.ix_(occ_a, virt_a, occ_b, virt_b)].reshape(n_pair_a, n_pair_b)
        
        V_trans = np.block([[V_aa, V_ab], [V_ab.T, V_bb]])
        
        W_grid = []
        for w in omega_grid:
            f_a = _f_rpa(lr, d_a, w, is_imaginary)
            f_b = _f_rpa(lr, d_b, w, is_imaginary)
            chi0_trans = np.diag(np.concatenate([f_a, f_b]))
            W_w = V_trans @ np.linalg.inv(np.eye(n_pair) - chi0_trans @ V_trans)
            W_grid.append(W_w)
        return np.array(W_grid)
    else:
        occ, virt = get_occ_virt_indices(lr.eps, nocc)
        n_pair = len(occ) * len(virt)
        d = (lr.eps[virt][None, :] - lr.eps[occ][:, None]).ravel()
        
        V_trans = lr.eri_chemist[np.ix_(occ, virt, occ, virt)].reshape(n_pair, n_pair)
        
        W_grid = []
        for w in omega_grid:
            f = _f_rpa(lr, d, w, is_imaginary)
            chi0_trans = np.diag(2.0 * f)
            W_w = V_trans @ np.linalg.inv(np.eye(n_pair) - chi0_trans @ V_trans)
            W_grid.append(W_w)
        return np.array(W_grid)

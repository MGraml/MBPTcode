"""self_energy_matrix_imaginary_time_df: the dense-DF space-time self-energy.

Successor to tests/test_imaginary_axis_matrix.py, which pinned the
frequency-axis matrix routine this replaced. Two of that file's three checks
carry over unchanged in intent -- the diagonal must agree with the scalar
`self_energy_imaginary_axis`, which is still production code behind
`solve_qp_energy_imaginary_axis`, and Sigma must be p<->q symmetric. The third,
a bit-exact pin of the factorized form against a naive one, went with the
routine it described.

What changes is the tolerance on the first. The two routines are no longer the
same formula on the same grid: one is a convolution over imaginary frequency,
the other a product in imaginary time, each with its own grid error. So the
check is a convergence statement, and it is made by showing the difference
FALL with the number of time points rather than by asserting one number.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, gto, scf

from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.Base.utils.grids import minimax_frequency_grid, minimax_time_grid
from src.SingleReference.GW.imaginary_axis import (self_energy_imaginary_axis,
                                                   solve_screening_imaginary_axis)
from src.SingleReference.GW.imaginary_time import (
    self_energy_fit_ranges, self_energy_matrix_imaginary_time_df)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    norb = len(eps)
    df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=df_coeff, spin_mode='restricted', eta=1e-3)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    query = np.array([0.05, 0.2, 0.6])

    all_ok = True

    # Reference: the frequency-axis scalar routine, pushed to a converged grid.
    # 24 points is where it stops moving on this system (nfreq 20 -> 24 shifts
    # the HOMO by 0.4 meV, 24 -> 28 by under 0.001).
    fp, fw = minimax_frequency_grid(24, *rW)
    W_ref = solve_screening_imaginary_axis(lr, nocc, fp)
    ref = np.array([self_energy_imaginary_axis(df_coeff, eps - mu, nocc, p, fp, fw,
                                               W_ref, query) for p in range(norb)])

    # 1. The diagonal must converge onto that scalar reference as ntau grows.
    print('1. dense-DF time diagonal -> scalar frequency-axis reference:')
    diffs = []
    for ntau in (14, 18, 22):
        om_in = minimax_frequency_grid(ntau, *rW)[0]
        tau = 0.5 * minimax_time_grid(ntau, *rS)[0]
        sig = self_energy_matrix_imaginary_time_df(
            df_coeff, solve_screening_imaginary_axis(lr, nocc, om_in), eps, nocc,
            tau, om_in, query, mu=mu, ranges=(rW, rS))
        d = max(np.max(np.abs(sig[:, p, p] - ref[p])) for p in range(norb))
        diffs.append(d)
        print(f'   ntau={ntau:3d}: maxdiff={d:.2e}')
    ok = diffs[0] > diffs[1] > diffs[2] and diffs[-1] < 5e-6
    all_ok &= ok
    print(f"   {'OK' if ok else 'FAIL'}: falls monotonically, below 5e-6 by ntau=22")

    # 2. Sigma_pq = Sigma_qp. Structural, not grid-limited: the aux contraction
    #    A_pmq is symmetric under p<->q for symmetric W, and the Green's-function
    #    reweighting touches only the middle index. A tight bound here catches a
    #    transposed operand that the diagonal check above would not see.
    sig_sym = np.max(np.abs(sig - sig.transpose(0, 2, 1)))
    ok_sym = sig_sym < 1e-10
    all_ok &= ok_sym
    print(f'2. p<->q symmetry: maxdiff={sig_sym:.2e}  '
          f"{'OK' if ok_sym else 'FAIL'}")

    # 3. Sigma(mu - i.w) == conj Sigma(mu + i.w). A caller can take the
    #    occupied half of the axis by conjugation instead of running a second
    #    tau sweep, so the symmetry is pinned here rather than left implicit.
    om_in = minimax_frequency_grid(22, *rW)[0]
    tau = 0.5 * minimax_time_grid(22, *rS)[0]
    W22 = solve_screening_imaginary_axis(lr, nocc, om_in)
    s_plus = self_energy_matrix_imaginary_time_df(
        df_coeff, W22, eps, nocc, tau, om_in, query, mu=mu, ranges=(rW, rS))
    s_minus = self_energy_matrix_imaginary_time_df(
        df_coeff, W22, eps, nocc, tau, om_in, -query, mu=mu, ranges=(rW, rS))
    conj_diff = np.max(np.abs(s_minus - np.conj(s_plus)))
    ok_conj = conj_diff < 1e-10
    all_ok &= ok_conj
    print(f'3. Sigma(-i.w) == conj Sigma(+i.w): maxdiff={conj_diff:.2e}  '
          f"{'OK' if ok_conj else 'FAIL'}")

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.analyticalContinuation import (
    thiele_coefficients, pade_eval, greedy_pade_order, pade_continue,
    thiele_coefficients_matrix, pade_eval_matrix, greedy_pade_order_matrix, pade_continue_matrix,
)

if __name__ == '__main__':
    all_ok = True

    # A Pade approximant built from exact samples of a rational function must
    # reproduce that function exactly (up to floating-point roundoff) at any
    # other point -- the strongest correctness check available for this code.
    def f(z):
        return 1.0 / (z - 2.0) + 1.0 / (z + 3.0) - 0.5

    z_fit = np.array([0.1j, 0.5j, 1.0j, 2.0j, 3.0j, 5.0j, 8.0j, 12.0j], dtype=complex)
    f_fit = f(z_fit)

    a = thiele_coefficients(z_fit, f_fit)
    z_query = np.array([0.3j, 4.0j, 10.0j, 1.0 + 0.5j])
    got = pade_eval(z_query, z_fit, a)
    expected = f(z_query)
    ok = np.allclose(got, expected, atol=1e-6)
    all_ok &= ok
    print(f"thiele/pade_eval reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got-expected)):.2e})")

    got_greedy = pade_continue(z_fit, f_fit, z_query, greedy=True)
    ok = np.allclose(got_greedy, expected, atol=1e-6)
    all_ok &= ok
    print(f"pade_continue(greedy=True) reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_greedy-expected)):.2e})")

    got_naive = pade_continue(z_fit, f_fit, z_query, greedy=False)
    ok = np.allclose(got_naive, expected, atol=1e-6)
    all_ok &= ok
    print(f"pade_continue(greedy=False) reproduces exact rational function: {'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_naive-expected)):.2e})")

    order = greedy_pade_order(z_fit, f_fit)
    ok = sorted(order.tolist()) == list(range(len(z_fit)))
    all_ok &= ok
    print(f"greedy_pade_order returns a valid permutation: {'OK' if ok else 'FAIL'}")

    # Matrix Pade: a matrix resolvent F(z) = (z*I - H)^{-1} is genuinely matrix-valued
    # and rational in z (mirrors the strongest correctness check used above) -- an exact
    # Thiele reconstruction from samples must reproduce it exactly at other points.
    rng = np.random.default_rng(0)
    d = 4
    H_rand = rng.standard_normal((d, d))
    H = H_rand + H_rand.T  # symmetric, real eigenvalues away from the imaginary axis

    def F(z):
        z = np.atleast_1d(z)
        return np.array([np.linalg.inv(zi * np.eye(d) - H) for zi in z])

    z_fit_m = np.array([0.1j, 0.5j, 1.0j, 2.0j, 3.0j, 5.0j, 8.0j, 12.0j, 20.0j, 35.0j], dtype=complex)
    F_fit = F(z_fit_m)

    a_m = thiele_coefficients_matrix(z_fit_m, F_fit)
    z_query_m = np.array([0.3j, 4.0j, 10.0j, 1.0 + 0.5j])
    got_m = pade_eval_matrix(z_query_m, z_fit_m, a_m)
    expected_m = F(z_query_m)
    ok = np.allclose(got_m, expected_m, atol=1e-6)
    all_ok &= ok
    print(f"thiele_coefficients_matrix/pade_eval_matrix reproduces exact matrix resolvent: "
          f"{'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_m - expected_m)):.2e})")

    got_m_greedy = pade_continue_matrix(z_fit_m, F_fit, z_query_m, greedy=True)
    ok = np.allclose(got_m_greedy, expected_m, atol=1e-6)
    all_ok &= ok
    print(f"pade_continue_matrix(greedy=True) reproduces exact matrix resolvent: "
          f"{'OK' if ok else 'FAIL'} (max err={np.max(np.abs(got_m_greedy - expected_m)):.2e})")

    order_m = greedy_pade_order_matrix(z_fit_m, F_fit)
    ok = sorted(order_m.tolist()) == list(range(len(z_fit_m)))
    all_ok &= ok
    print(f"greedy_pade_order_matrix returns a valid permutation: {'OK' if ok else 'FAIL'}")

    # d=1 must reduce exactly to the scalar algorithm.
    f_scalar_as_matrix = f_fit.reshape(-1, 1, 1)
    a_1 = thiele_coefficients_matrix(z_fit, f_scalar_as_matrix)
    got_1 = pade_eval_matrix(z_query, z_fit, a_1)[:, 0, 0]
    expected_1 = pade_eval(z_query, z_fit, thiele_coefficients(z_fit, f_fit))
    ok = np.allclose(got_1, expected_1, atol=1e-10)
    all_ok &= ok
    print(f"matrix Pade at d=1 matches scalar Pade exactly: {'OK' if ok else 'FAIL'} "
          f"(max err={np.max(np.abs(got_1 - expected_1)):.2e})")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")

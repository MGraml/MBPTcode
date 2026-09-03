"""TimeFrequencyGrid: one container, two backends, and the honest diagnostics.

Checks:
  1. The ported minimax cosine transform reproduces the model pair it is fitted
     for: Pi(i.tau) = e^{-x tau}  ->  Pi(i.omega) = 2x/(x^2 + omega^2).
  2. The minimax forward/backward pair is NOT a dual -- the property that makes
     four separate matrices necessary rather than one plus an inverse.
  3. The IR backend, by contrast, DOES round-trip, because its transform goes
     through basis coefficients.
  4. gauss_legendre carries a frequency axis only and refuses to transform
     rather than returning nonsense.
  5. Both backends present the identical interface, so a consumer never
     branches on `method`.

Usage: python tests/test_time_frequency_grid.py (no pytest, per project convention).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.utils.time_frequency import TimeFrequencyGrid

E_MIN, E_MAX = 0.4, 40.0


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'  ({detail})' if detail else ''))
    return bool(ok)


def test_minimax_transforms_the_model_pair():
    g = TimeFrequencyGrid.minimax(14, E_MIN, E_MAX)
    ok = True
    worst = 0.0
    for x in (0.5, 2.0, 11.0, 37.0):
        got = g.to_omega(np.exp(-x * g.tau_points), parity='even')
        want = 2 * x / (x**2 + g.omega_points**2)
        rel = np.abs((got - want) / want).max()
        worst = max(worst, rel)
    ok &= check(worst < 5e-2, 'cosine tau->omega reproduces 2x/(x^2+w^2)',
                f'worst rel err {worst:.2e} over x in [e_min, e_max]')
    ok &= check(max(g.fit_errors.values()) < 1e-3,
                'every transform fit converged',
                ', '.join(f'{k}={v:.1e}' for k, v in g.fit_errors.items()))
    return ok


def test_transform_matrices_are_not_inverses():
    """Why four matrices -- and why |A B - I| is the wrong acceptance test."""
    ok = True
    duality = {n: TimeFrequencyGrid.minimax(n, E_MIN, E_MAX).duality_error()
               for n in (10, 14, 18)}
    ok &= check(all(e > 1e-3 for e in duality.values()),
                'as MATRICES the forward/backward pair is not an inverse pair',
                ', '.join(f'n={k}: {v:.2e}' for k, v in duality.items()))
    g = TimeFrequencyGrid.minimax(14, E_MIN, E_MAX)
    sv = np.linalg.svd(g.cosft_wt, compute_uv=False)
    ok &= check(sv.max() / sv.min() > 1e2, 'nor is either one orthogonal',
                f'singular values {sv.min():.2e} .. {sv.max():.2e}')
    # ...but on the subspace every physical Pi(i.tau) lives in, it round-trips.
    rt = {n: TimeFrequencyGrid.minimax(n, E_MIN, E_MAX).roundtrip_error()
          for n in (10, 14, 18)}
    ok &= check(all(e < 1e-4 for e in rt.values()),
                'yet the round trip on sums of exponentials is accurate',
                ', '.join(f'n={k}: {v:.2e}' for k, v in rt.items()))
    return ok


def test_minimax_usable_window():
    """The usable n is set by the ENERGY RANGE, not by n alone.

    GreenX's Remez generator only converges once the window is wide enough --
    its own tables record the floor climbing from e_max/e_min = 10 at n=14 to
    9649 at n=34, with the number of convergent windows collapsing after n=24.
    The downstream transform fit hits the same wall, so a grid can be too DENSE
    for a narrow range as easily as too sparse for a wide one.
    """
    import warnings
    from src.Base.utils.time_frequency import minimax_convergence_floor
    ok = True

    floors = [minimax_convergence_floor(n) for n in (14, 20, 24, 30, 34)]
    ok &= check(floors == sorted(floors) and floors[0] < floors[-1],
                'the tabulated Remez floor rises monotonically with n',
                ' -> '.join(f'{f:g}' for f in floors))

    # too dense for a narrow range, and too sparse for a wide one, both caught
    cases = [(14, 1e2, False), (20, 1e2, False),      # good
             (24, 1e2, True),  (30, 1e2, True),       # too dense for 1e2
             (14, 1e4, True),                          # too sparse for 1e4
             (24, 1e3, False), (30, 1e4, False)]       # good again
    for n, rng, want_warn in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            g = TimeFrequencyGrid.minimax(n, E_MIN, E_MIN * rng)
        got_warn = any('minimax transform fit' in str(w.message) for w in caught)
        ok &= check(got_warn == want_warn,
                    f'n={n:2d}, range={rng:.0e}: '
                    f"{'flagged' if want_warn else 'accepted'}",
                    f'fit err {g.fit_errors["cosft_wt"]:.1e}')
    return ok


def test_ir_does_round_trip():
    g = TimeFrequencyGrid.ir(beta=50.0, omega_max=5.0, eps=1e-10, statistics='boson')
    ok = check(g.roundtrip_error() < 1e-6,
               'IR round-trips through basis coefficients',
               f'roundtrip = {g.roundtrip_error():.2e}, L = {g.meta["size"]}')
    # a bosonic Pi(i.tau) = e^{-x tau} + e^{-x (beta - tau)} round-trips
    beta, x = 50.0, 1.3
    f = np.exp(-x * g.tau_points) + np.exp(-x * (beta - g.tau_points))
    back = g.to_tau(g.to_omega(f, 'even'), 'even')
    rel = np.abs(back - f).max() / np.abs(f).max()
    ok &= check(rel < 1e-6, 'and a bosonic model function survives the round trip',
                f'max rel err {rel:.2e}')
    return ok


def test_gauss_legendre_refuses_to_transform():
    g = TimeFrequencyGrid.gauss_legendre(20, w0=0.5)
    ok = check(g.nfreq == 20 and g.ntau == 0,
               'gauss_legendre is frequency-only', f'{g!r}')
    try:
        g.to_omega(np.zeros(20))
        ok &= check(False, 'to_omega raises instead of returning nonsense')
    except ValueError as exc:
        ok &= check('carries no' in str(exc),
                    'to_omega raises instead of returning nonsense',
                    str(exc)[:60])
    return ok


def test_identical_interface():
    grids = [TimeFrequencyGrid.minimax(14, E_MIN, E_MAX),
             TimeFrequencyGrid.ir(beta=50.0, omega_max=5.0, statistics='boson')]
    fields = ('tau_points', 'tau_weights', 'omega_points', 'omega_weights',
              'cosft_wt', 'cosft_tw', 'sinft_wt', 'sinft_tw')
    ok = True
    for g in grids:
        present = all(getattr(g, f) is not None and len(np.shape(getattr(g, f)))
                      for f in fields)
        shapes_ok = (g.cosft_wt.shape == (g.nfreq, g.ntau)
                     and g.cosft_tw.shape == (g.ntau, g.nfreq))
        ok &= check(present and shapes_ok,
                    f"method={g.method!r} exposes the full interface with consistent shapes",
                    f'ntau={g.ntau} nfreq={g.nfreq}')
    return ok


if __name__ == '__main__':
    all_ok = True
    print('\n-- 1. minimax reproduces its model pair')
    all_ok &= test_minimax_transforms_the_model_pair()
    print('\n-- 2. matrices are not inverses, but the round trip works')
    all_ok &= test_transform_matrices_are_not_inverses()
    print('\n-- 2b. the usable n depends on the energy range')
    all_ok &= test_minimax_usable_window()
    print('\n-- 3. IR does round-trip')
    all_ok &= test_ir_does_round_trip()
    print('\n-- 4. gauss_legendre is frequency-only')
    all_ok &= test_gauss_legendre_refuses_to_transform()
    print('\n-- 5. one interface, both backends')
    all_ok &= test_identical_interface()
    print('\n' + ('All TimeFrequencyGrid checks passed.' if all_ok else 'FAILURES above.'))
    sys.exit(0 if all_ok else 1)

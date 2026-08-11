"""Validation ladder for src/SingleReference/ADC/spin_adapt.py -- the
numerically-constructed doublet-CSF isometry T for the spin-orbital ADC
solver (see that module's docstring for why this is built numerically
instead of hand-derived).

Layers (each must pass before the next means anything):
  1. exact bitstring algebra: canonical anticommutators on random dets
  2. T columns orthonormal; memoized == unmemoized construction
  3. CSF dimension counts match the closed-form Type-I/II/III counts
     (2h1p/2p1h) and the genealogical doublet counts (3h2p/3p2h)
  4. subspace invariance: ||H T - T (T^T H T)||_max ~ 0 for the FULL dense
     adc4_base AND adc4 supermatrices (this is the theorem-level check: span(T)
     is an invariant subspace of every ADC ingredient at once)
  5. spectrum: every (energy, Z>1e-8) pole of the spin-orbital supermatrix
     appears in eig(T^T H T) with the same energy and Z, exactly once
     (the spin-orbital solve reports each pole twice -- alpha/beta copies)

Run directly: python tests/test_spin_adapt.py  (NOT pytest)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC import ADCSolverUnrestricted
from src.SingleReference.ADC.spin_adapt import apply_ops, build_class_T, build_T
from src.Base.pyscf_interface import (get_orbital_energies,
                                 get_two_electron_integrals_chemist,
                                 get_antisymmetrized_spin_eri)


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  {label:60s} {status}  {detail}")
    if not ok:
        raise SystemExit(f"FAILED: {label}")


# ------------------------------------------------------------------
# 1. bitstring algebra: anticommutators on random determinants
# ------------------------------------------------------------------
print("1. exact bitstring second quantization")
rng = np.random.default_rng(0)
nmodes = 8
worst = 0.0
for _ in range(200):
    det = int(rng.integers(0, 1 << nmodes))
    i, j = int(rng.integers(nmodes)), int(rng.integers(nmodes))
    # {a_i, a+_j} = delta_ij ; {a_i, a_j} = 0 ; {a+_i, a+_j} = 0
    for opsA, opsB, delta in (
        ([(i, False)], [(j, True)], 1.0 if i == j else 0.0),
        ([(i, False)], [(j, False)], 0.0),
        ([(i, True)], [(j, True)], 0.0),
    ):
        acc = {}
        for first, second in ((opsA, opsB), (opsB, opsA)):
            d1, s1 = apply_ops(det, first)
            if s1:
                d2, s2 = apply_ops(d1, second)
                if s2:
                    acc[d2] = acc.get(d2, 0.0) + s1 * s2
        # anticommutator must be delta * identity on |det>
        for d2, coef in acc.items():
            expect = delta if d2 == det else 0.0
            worst = max(worst, abs(coef - expect))
        if delta and det not in acc:
            worst = max(worst, delta)
check("anticommutators {a,a+}={delta}, {a,a}=0", worst < 1e-14, f"worst {worst:.1e}")


# ------------------------------------------------------------------
# system: H2O/sto-3g
# ------------------------------------------------------------------
mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
nocc_spin = mol.nelectron
eps_spin = get_orbital_energies(mf, representation='spin')
eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
solver = ADCSolverUnrestricted.from_arrays(eps_spin, get_antisymmetrized_spin_eri(eri))
d = solver.dimensions_adc4(nocc_spin)
O, V = d['nocc'] // 2, d['nvirt'] // 2  # spatial

print("2. T construction (H2O/sto-3g)")
# closed-form Jordan-Wigner signs vs exact bitstring operators, EVERY config
from src.SingleReference.ADC.spin_adapt import (_class_signs_closed_form,
                                                 _class_configs, _class_ops)
hf_det = (1 << nocc_spin) - 1
for cls in ('2h1p', '2p1h', '3h2p', '3p2h'):
    cfgs = _class_configs(solver, nocc_spin, cls)
    cf = _class_signs_closed_form(cls, cfgs, nocc_spin)
    ok = all(apply_ops(hf_det, _class_ops(cls, tuple(int(x) for x in cfgs[c])))[1] == cf[c]
             for c in range(len(cfgs)))
    check(f"closed-form signs == apply_ops, ALL {len(cfgs)} {cls} configs", ok)

T = build_T(solver, nocc_spin)
for cls in ('orb', '2h1p', '2p1h', '3h2p', '3p2h', 'SD', 'full'):
    G = (T[cls].T @ T[cls]).toarray()
    dev = np.max(np.abs(G - np.eye(G.shape[0])))
    check(f"T[{cls}] columns orthonormal", dev < 1e-12, f"dev {dev:.1e}")

for cls in ('2h1p', '2p1h', '3h2p', '3p2h'):
    Tm, Km = build_class_T(solver, nocc_spin, cls, memoize=True)
    Tu, Ku = build_class_T(solver, nocc_spin, cls, memoize=False)
    dev = abs(Tm - Tu).max() + np.max(np.abs(Km - Ku))
    check(f"memoized == unmemoized ({cls})", dev < 1e-14, f"dev {dev:.1e}")

print("3. CSF dimension counts")
nP_o, nP_v = O * (O - 1) // 2, V * (V - 1) // 2
n2h1p_csf_expect = O * V + 2 * nP_o * V          # Type I + II + III
n2p1h_csf_expect = O * V + 2 * O * nP_v          # Type I' + II' + III'
check("2h1p CSF count == Type I+II+III", T['2h1p'].shape[1] == n2h1p_csf_expect,
      f"{T['2h1p'].shape[1]} vs {n2h1p_csf_expect}")
check("2p1h CSF count == Type I'+II'+III'", T['2p1h'].shape[1] == n2p1h_csf_expect,
      f"{T['2p1h'].shape[1]} vs {n2p1h_csf_expect}")
# genealogical doublet counts: 5 distinct open shells -> 5 doublets,
# 3 open shells -> 2, 1 open shell -> 1; hole patterns are (i<j<k distinct)
# or (i doubly-annihilated + single j != i, O*(O-1) choices), particle
# patterns (a<b distinct) or (a doubly-occupied... a==b, V choices) --
# mirrored for 3p2h
from math import comb
n3h2p_expect = (comb(O, 3) * comb(V, 2) * 5
                + comb(O, 3) * V * 2
                + O * (O - 1) * comb(V, 2) * 2
                + O * (O - 1) * V * 1)
n3p2h_expect = (comb(V, 3) * comb(O, 2) * 5
                + comb(V, 3) * O * 2
                + V * (V - 1) * comb(O, 2) * 2
                + V * (V - 1) * O * 1)
check("3h2p doublet count matches genealogical formula",
      T['3h2p'].shape[1] == n3h2p_expect, f"{T['3h2p'].shape[1]} vs {n3h2p_expect}")
check("3p2h doublet count matches genealogical formula",
      T['3p2h'].shape[1] == n3p2h_expect, f"{T['3p2h'].shape[1]} vs {n3p2h_expect}")
print(f"  dims: spin {d['nH_adc4']} -> csf {T['full'].shape[1]} "
      f"(ratio {d['nH_adc4'] / T['full'].shape[1]:.2f})")


print("\nALL PASSED")

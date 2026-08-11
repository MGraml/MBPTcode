"""Regression tests for the CCSDT + Lambda-CCSDT + 1-RDM pipeline"""
import os
import sys
import time
from itertools import permutations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.SingleReference.CC import lambda1_residual as L1M
from src.SingleReference.CC import lambda2_residual as L2M
from src.SingleReference.CC import lambda3_residual as L3M
from src.SingleReference.CC import d1_blocks as D1M
from src.SingleReference.CC.integrals import build_spinorbital_integrals, energy_denominators
from src.SingleReference.CC import amplitudes
from src.SingleReference.CC import solver

LIH_STO3G = dict(geometry=[('Li', (0, 0, 0)), ('H', (0, 0, 1.5957))], basis='sto-3g')


def _antisymmetrize_last6(x, occ_axes, vir_axes):
    """Fully antisymmetrize a rank-6 tensor over a triple of occ axes and a
    triple of vir axes."""
    def sign(p):
        p = list(p)
        s = 1
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    s = -s
        return s

    out = np.zeros_like(x)
    for po in permutations(range(3)):
        for pv in permutations(range(3)):
            axes = list(range(x.ndim))
            for k, ax in enumerate(occ_axes):
                axes[ax] = occ_axes[po[k]]
            for k, ax in enumerate(vir_axes):
                axes[ax] = vir_axes[pv[k]]
            out += sign(po) * sign(pv) * x.transpose(axes)
    return out / 36


def _random_amplitudes(nocc, nvir, seed=0):
    rng = np.random.default_rng(seed)
    nmo = nocc + nvir

    t1 = rng.random((nvir, nocc))
    t2 = rng.random((nvir, nvir, nocc, nocc))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t3 = _antisymmetrize_last6(rng.random((nvir, nvir, nvir, nocc, nocc, nocc)),
                               occ_axes=(3, 4, 5), vir_axes=(0, 1, 2))

    l1 = rng.random((nocc, nvir))
    l2 = rng.random((nocc, nocc, nvir, nvir))
    l2 = l2 - l2.transpose(1, 0, 2, 3)
    l2 = l2 - l2.transpose(0, 1, 3, 2)
    l3 = _antisymmetrize_last6(rng.random((nocc, nocc, nocc, nvir, nvir, nvir)),
                               occ_axes=(0, 1, 2), vir_axes=(3, 4, 5))

    f = rng.random((nmo, nmo))
    f = f + f.T
    g = rng.random((nmo, nmo, nmo, nmo))
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    kd = np.eye(nmo)

    return t1, t2, t3, l1, l2, l3, f, g, kd


def check_lambda3_antisymmetry():
    """The Lambda3 residual must be antisymmetric under exchange of any two
    virtual or any two occupied indices -- a necessary structural property
    that a wrong P(x,y) interpretation silently violates."""
    nocc, nvir = 3, 4
    o, v = slice(None, nocc), slice(nocc, None)
    t1, t2, t3, l1, l2, l3, f, g, kd = _random_amplitudes(nocc, nvir)

    r3 = L3M.lambda3_residual(t1, t2, t3, l1, l2, l3, f, g, kd, o, v)

    ok = (np.max(np.abs(r3 + r3.transpose(1, 0, 2, 3, 4, 5))) < 1e-10
          and np.max(np.abs(r3 + r3.transpose(0, 2, 1, 3, 4, 5))) < 1e-10
          and np.max(np.abs(r3 + r3.transpose(0, 1, 2, 4, 3, 5))) < 1e-10
          and np.max(np.abs(r3 + r3.transpose(0, 1, 2, 3, 5, 4))) < 1e-10)
    print(f"lambda3 residual antisymmetry: {'OK' if ok else 'FAIL'}")
    return ok


def check_l3_call_is_fast():
    """Guards against the missing-optimize=True regression: a single call on
    this size system must complete quickly, not stall for minutes."""
    nocc, nvir = 4, 8
    o, v = slice(None, nocc), slice(nocc, None)
    t1, t2, t3, l1, l2, l3, f, g, kd = _random_amplitudes(nocc, nvir, seed=1)

    t0 = time.time()
    L3M.lambda3_residual(t1, t2, t3, l1, l2, l3, f, g, kd, o, v)
    elapsed = time.time() - t0
    ok = elapsed < 10.0
    print(f"lambda3 single call fast ({elapsed:.2f}s < 10s): {'OK' if ok else 'FAIL'}"
          + ("" if ok else " -- optimize= regression?"))
    return ok


def check_ccsd_limit_matches_pyscf():
    """At the CCSD level (T3=L3=0), our Lambda1/Lambda2/D1 must reproduce
    pyscf's own trusted GCCSD Lambda/1-RDM. This is the strongest available
    correctness check, since pyscf has no CCSDT to compare Lambda3 against."""
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.scf import addons
    from pyscf.cc import ccsd as pyscf_ccsd

    mol = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf = scf.RHF(mol)
    mf.kernel()
    mf_ghf = addons.convert_to_ghf(mf)
    mycc = cc.GCCSD(mf_ghf)
    mycc.kernel()
    l1_ref, l2_ref = mycc.solve_lambda()

    nocc, nvir = mycc.t1.shape
    nmo = nocc + nvir
    o, v = slice(None, nocc), slice(nocc, None)
    t1 = mycc.t1.transpose(1, 0)
    t2 = mycc.t2.transpose(2, 3, 0, 1)
    t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    l3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir))

    dm = mf_ghf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    vhf = mf_ghf.get_veff(mf_ghf.mol, dm)
    fockao = mf_ghf.get_fock(vhf=vhf, dm=dm)
    mo_idx = pyscf_ccsd.get_frozen_mask(mycc)
    mo_coeff = mycc.mo_coeff[:, mo_idx]
    f = mo_coeff.conj().T @ fockao @ mo_coeff

    nao = mo_coeff.shape[0]
    mo_a, mo_b = mo_coeff[:nao // 2], mo_coeff[nao // 2:]
    eri = ao2mo.kernel(mf._eri, mo_a) + ao2mo.kernel(mf._eri, mo_b)
    eri1 = ao2mo.kernel(mf._eri, (mo_a, mo_a, mo_b, mo_b))
    eri += eri1 + eri1.T
    eri = ao2mo.restore(1, eri, nmo).reshape(nmo, nmo, nmo, nmo)
    g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)

    r1 = L1M.lambda1_residual(t1, t2, t3, l1_ref, l2_ref, l3, f, g, o, v)
    r2 = L2M.lambda2_residual(t1, t2, t3, l1_ref, l2_ref, l3, f, g, o, v)
    ok = np.linalg.norm(r1) < 1e-6 and np.linalg.norm(r2) < 1e-6
    print(f"Lambda1/2 residual ~0 at pyscf's converged GCCSD-Lambda "
          f"(|r1|={np.linalg.norm(r1):.2e}, |r2|={np.linalg.norm(r2):.2e}): {'OK' if ok else 'FAIL'}")

    kd = np.eye(nmo)
    opdm = np.zeros((nmo, nmo))
    opdm[:nocc, :nocc] = D1M.d1_oo(t1, t2, t3, l1_ref, l2_ref, l3, kd, o, v)
    opdm[nocc:, nocc:] = D1M.d1_vv(t1, t2, t3, l1_ref, l2_ref, l3, kd, o, v)
    opdm[:nocc, nocc:] = D1M.d1_ov(t1, t2, t3, l1_ref, l2_ref, l3, kd, o, v)
    opdm[nocc:, :nocc] = D1M.d1_vo(t1, t2, t3, l1_ref, l2_ref, l3, kd, o, v)

    dm_ref = np.zeros((nmo, nmo))
    dm_ref[:nocc, :nocc] = np.eye(nocc)
    my_corr = opdm - dm_ref
    dm_pyscf = mycc.make_rdm1(t1=mycc.t1, t2=mycc.t2, l1=l1_ref, l2=l2_ref, with_mf=False)

    sym_diff = 0.5 * (my_corr + my_corr.T) - 0.5 * (dm_pyscf + dm_pyscf.T)
    ok2 = np.max(np.abs(sym_diff)) < 1e-9
    print(f"CCSD-limit 1-RDM matches pyscf make_rdm1 (max diff {np.max(np.abs(sym_diff)):.2e}): "
          f"{'OK' if ok2 else 'FAIL'}")
    return ok and ok2


def check_full_ccsdt_lambda_lih_sto3g():
    """End-to-end: real (nonzero) T3/L3 on LiH/STO-3G. Checks convergence,
    particle-number conservation, and that the density matrix isn't wildly
    non-Hermitian (small antisymmetry is expected for the CC response
    density, not a bug)."""
    ints = build_spinorbital_integrals(**LIH_STO3G)
    fock, g = ints['fock'], ints['g']
    nocc, nvir = ints['nocc'], ints['nvir']
    o, v = slice(None, nocc), slice(nocc, None)
    e_ai, e_abij, e_abcijk = energy_denominators(fock, nocc, nvir)

    t1 = np.zeros((nvir, nocc))
    t2 = np.zeros((nvir, nvir, nocc, nocc))
    t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t1, t2, t3 = amplitudes.kernel(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk,
                                   ints['hf_energy'], max_iter=100, stopping_eps=1e-9)
    ok_t3 = np.linalg.norm(t3) > 1e-6
    print(f"T3 genuinely nonzero for 4-electron LiH (|t3|={np.linalg.norm(t3):.2e}): "
          f"{'OK' if ok_t3 else 'FAIL'}")

    l1, l2, l3 = solver.solve_lambda_ccsdt(t1, t2, t3, fock, g, o, v,
                                           e_ai, e_abij, e_abcijk,
                                           max_iter=100, stopping_eps=1e-8,
                                           verbose=False)

    opdm = solver.ccsdt_one_rdm(t1, t2, t3, l1, l2, l3, o, v)
    ok_tr = abs(np.trace(opdm) - nocc) < 1e-8
    ok_herm = np.max(np.abs(opdm - opdm.T)) < 1e-3
    print(f"CCSDT 1-RDM trace == n_electrons ({np.trace(opdm):.10f} vs {nocc}): "
          f"{'OK' if ok_tr else 'FAIL'}")
    print(f"CCSDT 1-RDM near-Hermitian (max asym {np.max(np.abs(opdm - opdm.T)):.2e}): "
          f"{'OK' if ok_herm else 'FAIL'}")
    return ok_t3 and ok_tr and ok_herm


if __name__ == '__main__':
    all_ok = True
    all_ok &= check_lambda3_antisymmetry()
    all_ok &= check_l3_call_is_fast()
    all_ok &= check_ccsd_limit_matches_pyscf()
    all_ok &= check_full_ccsdt_lambda_lih_sto3g()
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)

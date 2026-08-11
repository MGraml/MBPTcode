"""Integration tests for the CCSDT AO density matrix (src/SingleReference/CC/):

1. Exactness check: for a 2-electron system (H2) CCSD is already FCI, so the
   CCSDT AO 1-RDM must match pyscf's own CCSD 1-RDM essentially exactly --
   this exercises the whole new mf-based chain (integrals-from-mf, T, Lambda,
   1-RDM assembly, spin-summing, AO backtransform) against a trusted oracle.
2. compute_ccsd_density_matrix must reproduce the inline construction the
   test sweeps have always used (mo @ make_rdm1 @ mo.T).
3. Structural-zero check: HF/sto-3g has only 2 virtual spin-orbitals, so T3
   (antisymmetric in 3 virtual indices) is identically zero and the CCSDT
   density must agree with the CCSD one to solver tolerance.
4. The CCSDT density plugs into calc_qp_energy(dm_correction=...) exactly
   like the CCSD and GW densities do; on LiH/sto-3g (8 virtual spin-orbitals,
   |t3| ~ 1e-2) the CCSDT and CCSD corrections must be close but genuinely
   distinct.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, cc

from src.SingleReference.CC import (compute_ccsdt_density_matrix,
                                 compute_ccsd_density_matrix)
from src.SingleReference.GW.qp_energy import calc_qp_energy


def tight_ccsd_ao_density(mf):
    """pyscf CCSD AO density with tightened convergence (the defaults,
    conv_tol_normt=1e-6, leave ~5e-7 noise in the density)."""
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.run()
    dm = mf.mo_coeff @ mycc.make_rdm1() @ mf.mo_coeff.T
    return 0.5 * (dm + dm.T)


if __name__ == '__main__':
    all_ok = True

    # --- 1. H2: CCSDT density == pyscf CCSD density (both are FCI for 2e) ---
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g')
    mf = scf.RHF(mol).run()

    dm_ccsdt = compute_ccsdt_density_matrix(mf, verbose=False)
    dm_ccsd_ref = tight_ccsd_ao_density(mf)

    diff = np.max(np.abs(dm_ccsdt - dm_ccsd_ref))
    ok = diff < 1e-7
    all_ok &= ok
    print(f"H2/6-31g: CCSDT AO density == pyscf CCSD (=FCI) density "
          f"(max diff {diff:.2e}): {'OK' if ok else 'FAIL'}")

    nelec = np.trace(dm_ccsdt @ mf.get_ovlp())
    ok = abs(nelec - mol.nelectron) < 1e-8
    all_ok &= ok
    print(f"H2/6-31g: CCSDT AO density integrates to n_electrons "
          f"({nelec:.10f} vs {mol.nelectron}): {'OK' if ok else 'FAIL'}")

    # --- 2. CCSD wrapper == the inline construction used by the sweeps ---
    mol_hf = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf_hf = scf.RHF(mol_hf).run()
    dm_ccsd_hf = compute_ccsd_density_matrix(mf_hf)
    mycc_hf = cc.CCSD(mf_hf).run()
    dm_inline = mf_hf.mo_coeff @ mycc_hf.make_rdm1() @ mf_hf.mo_coeff.T
    diff = np.max(np.abs(dm_ccsd_hf - dm_inline))
    ok = diff < 1e-12
    all_ok &= ok
    print(f"HF/sto-3g: compute_ccsd_density_matrix == inline construction "
          f"(max diff {diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- 3. HF/sto-3g: only 2 virtual spin-orbitals -> T3 structurally 0,
    #        so the CCSDT density must equal the CCSD one to solver tolerance.
    dm_ccsdt_hf = compute_ccsdt_density_matrix(mf_hf, verbose=False)
    diff = np.max(np.abs(dm_ccsdt_hf - 0.5 * (dm_ccsd_hf + dm_ccsd_hf.T)))
    ok = diff < 1e-6
    all_ok &= ok
    print(f"HF/sto-3g: CCSDT == CCSD density when T3 is structurally zero "
          f"(max diff {diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- 4. LiH/sto-3g: real T3; dm_correction wiring + distinctness ---
    mol_lih = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf_lih = scf.RHF(mol_lih).run()
    dm_ccsdt_lih = compute_ccsdt_density_matrix(mf_lih, verbose=False)
    dm_ccsd_lih = tight_ccsd_ao_density(mf_lih)

    qp_ccsdt = calc_qp_energy(mf_lih, selfenergy='GW', polarizability='BSE',
                              df=False, state='homo', dm_correction=dm_ccsdt_lih)
    qp_ccsd = calc_qp_energy(mf_lih, selfenergy='GW', polarizability='BSE',
                             df=False, state='homo', dm_correction=dm_ccsd_lih)
    qp_plain = calc_qp_energy(mf_lih, selfenergy='GW', polarizability='BSE',
                              df=False, state='homo')

    ok = np.isfinite(qp_ccsdt) and qp_ccsdt != qp_plain
    all_ok &= ok
    print(f"LiH/sto-3g: CCSDT dm_correction runs and shifts the QP energy "
          f"(plain {qp_plain:.8f}, CCSD-corr {qp_ccsd:.8f}, CCSDT-corr {qp_ccsdt:.8f} eV): "
          f"{'OK' if ok else 'FAIL'}")

    ok = abs(qp_ccsdt - qp_ccsd) < 0.1 and abs(qp_ccsdt - qp_ccsd) > 1e-9
    all_ok &= ok
    print(f"LiH/sto-3g: CCSDT vs CCSD correction close but distinct "
          f"(|diff| = {abs(qp_ccsdt - qp_ccsd):.3e} eV): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)

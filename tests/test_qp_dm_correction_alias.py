import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto, scf, cc, df

from src.SingleReference.GW.qp_energy import calc_qp_energy

if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    dm_ccsd = mf.mo_coeff @ mycc.make_rdm1() @ mf.mo_coeff.T

    qp_via_ccsd_kw = calc_qp_energy(mf, selfenergy='GW', polarizability='BSE', df=False, state='homo', dm_ccsd=dm_ccsd)
    qp_via_correction_kw = calc_qp_energy(mf, selfenergy='GW', polarizability='BSE', df=False, state='homo', dm_correction=dm_ccsd)
    qp_uncorrected = calc_qp_energy(mf, selfenergy='GW', polarizability='BSE', df=False, state='homo')

    ok = qp_via_ccsd_kw == qp_via_correction_kw
    all_ok &= ok
    print(f"dm_ccsd alias == dm_correction (identical result): {'OK' if ok else 'FAIL'} "
          f"({qp_via_ccsd_kw:.8f} vs {qp_via_correction_kw:.8f})")

    ok = qp_via_ccsd_kw != qp_uncorrected
    all_ok &= ok
    print(f"dm_correction actually changes the result vs uncorrected: {'OK' if ok else 'FAIL'}")

    # dm_correction takes precedence if both are (incorrectly) passed together.
    import numpy as np
    dm_dummy = np.zeros_like(dm_ccsd)
    qp_precedence = calc_qp_energy(mf, selfenergy='GW', polarizability='BSE', df=False, state='homo',
                                    dm_ccsd=dm_dummy, dm_correction=dm_ccsd)
    ok = qp_precedence == qp_via_ccsd_kw
    all_ok &= ok
    print(f"dm_correction takes precedence over dm_ccsd when both given: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")

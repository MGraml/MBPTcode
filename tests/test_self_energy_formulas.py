import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto, scf, df

from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import KNOWN_VERTEX_MODES

if __name__ == '__main__':
    all_ok = True

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g')
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    methods = ['GW', 'GW@RPA', 'GW@BSE', 'GWGammaInf', 'PSD1', 'PSD2', 'PSD4', 'PSD5', 'PSD6', 'PSD7', 'PSD8', 'PSD9']
    for m in methods:
        val = calc_qp_energy(mf, selfenergy=m, polarizability='BSE', df=True, state='homo')
        ok = isinstance(val, float) and val == val and -100 < val < 0  # not NaN, sane occupied-state sign/range
        all_ok &= ok
        print(f"{m:12s}: QP = {val:12.6f} eV  {'OK' if ok else 'FAIL'}")

    try:
        from src.SingleReference.GW.self_energy import SelfEnergySolver
        se = SelfEnergySolver(eps=[-1.0, -0.5, 0.2, 0.8], spin_mode='restricted')
        import numpy as np
        se.calculate_self_energy(0, -1.0, 2, np.array([0.5]), np.zeros((1, 4)), vertex_mode='PSD3')
        print("calculate_self_energy(vertex_mode='PSD3') did NOT raise: FAIL")
        all_ok = False
    except ValueError as e:
        print(f"calculate_self_energy(vertex_mode='PSD3') raises ValueError: OK")

    ok = 'PSD3' not in KNOWN_VERTEX_MODES
    all_ok &= ok
    print(f"'PSD3' absent from KNOWN_VERTEX_MODES: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")

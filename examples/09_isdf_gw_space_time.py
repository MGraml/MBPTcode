"""Cubic-scaling GW and BSE on one ISDF factorization.

The route for a large molecule: the SCF, the GW self-energy and the BSE all run
on the SAME separable (ISDF) factors. Reusing them is not just cheaper -- a
second fit lands in a different auxiliary gauge, which stays self-consistent and
silently moves the spectrum.

    python examples/09_isdf_gw_space_time.py
"""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV
from src.Base.isdf_jk import isdf_jk, separable_factors_from_jk
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
nocc = mol.nelectron // 2

# 1. SCF with an ISDF with_df: J and K from the factors, no cderi stored.
mf = isdf_jk(scf.RHF(mol), auxbasis='cc-pvdz-ri')
mf.kernel()
print(f'ISDF-SCF   E = {mf.e_tot:.8f} Ha')

# 2. Hand the GW and BSE the factorization the SCF already built.
factors = separable_factors_from_jk(mf)

# 3. G0W0, O(N^3). A window of states shares one Sigma.
qp = solve_qp_energy_space_time(mf, mol, nocc, [nocc - 2, nocc - 1],
                                factors=factors)
print('G0W0       HOMO-1, HOMO = '
      + ', '.join(f'{e * HARTREE_TO_EV:.3f}' for e in qp) + ' eV')

# 4. BSE on those same factors.
omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=3, factors=factors)
print('BSE@G0W0   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega) + ' eV')

# Or in one step: solve_bse_isdf builds the fit, runs the GW and puts every
# quasiparticle energy on the BSE diagonal itself.
omega1, _, _, _ = solve_bse_isdf(mf, mol, nocc, nroots=3)
print('one step   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega1) + ' eV')

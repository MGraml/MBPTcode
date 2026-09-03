"""Cubic-scaling GW and BSE on one ISDF factorization.

The route for a large molecule: the SCF, the GW self-energy and the BSE all run
on the SAME separable (ISDF) factors. Reusing them is not just cheaper -- a
second fit lands in a different auxiliary gauge, which stays self-consistent and
silently moves the spectrum.

`solve_bse_isdf` takes the two independently:

    factors=  the ISDF fit ONLY. Omit it and one is built; it says nothing
              about whether a GW has been run.
    qp=       what goes on the BSE diagonal. 'G0W0' (the default) runs the GW
              itself, an ARRAY of per-orbital energies uses yours, False gives
              BSE@mean-field.

The screened interaction is never passed: W is always built internally at the
MEAN-FIELD energies, which is what the G0W0-BSE split means. When the GW runs
inside, W is reused from its frequency axis; supply `qp=` and it is rebuilt.

    python examples/09_isdf_gw_space_time.py
"""
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.constants import HARTREE_TO_EV
from src.Base.isdf_jk import isdf_jk, separable_factors_from_jk
from src.SingleReference.GW.space_time import (solve_qp_diagonal_space_time,
                                               solve_qp_energy_space_time)
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

# 3. G0W0 for the states you want. One self-energy serves a whole window.
qp = solve_qp_energy_space_time(mf, mol, nocc, [nocc - 2, nocc - 1],
                                factors=factors)
print('G0W0       HOMO-1, HOMO = '
      + ', '.join(f'{e * HARTREE_TO_EV:.3f}' for e in qp) + ' eV')

# 4a. BSE in two steps: the whole QP diagonal, then the BSE on it. The diagonal
#     needs every orbital, so ask for it explicitly and pass it as qp=.
eps_qp, _ = solve_qp_diagonal_space_time(mf, mol, nocc, factors=factors)
omega, X, Y, info = solve_bse_isdf(mf, mol, nocc, nroots=3, qp=eps_qp,
                                   factors=factors)
print('BSE@G0W0   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega) + ' eV')

# 4b. Or in one call, which runs that same GW itself.
omega1, _, _, _ = solve_bse_isdf(mf, mol, nocc, nroots=3, factors=factors)
print('one call   ' + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega1) + ' eV')

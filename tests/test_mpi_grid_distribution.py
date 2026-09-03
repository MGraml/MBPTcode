"""Distributed tau loops must reproduce the serial answer exactly.

    python tests/test_mpi_grid_distribution.py           # partition algebra
    mpirun -n 3 python tests/test_mpi_grid_distribution.py   # the real check
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import numpy as np
warnings.simplefilter('ignore')
from pyscf import gto, scf
from src.Base.utils.mpi_grid import grid_comm, partition
from src.SingleReference.GW.space_time import (solve_qp_energy_space_time,
                                               separable_factors)
from src.Base.constants import HARTREE_TO_EV

comm, rank, size = grid_comm()


def check(ok, label, detail=''):
    if rank == 0:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'  ({detail})' if detail else ''))
    return bool(ok)


all_ok = True
if rank == 0:
    print(f'\n-- partition covers every index exactly once')
for n in (14, 18, 40):
    for s in (1, 2, 3, 5, 8, 64):
        idx = np.sort(np.concatenate([partition(n, r, s) for r in range(s)]))
        all_ok &= check(np.array_equal(idx, np.arange(n)), f'n={n}, {s} ranks')

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri'); mf.kernel()
nocc = mol.nelectron // 2
F = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
serial = solve_qp_energy_space_time(mf, mol, nocc, nocc-1, factors=F) * HARTREE_TO_EV
if rank == 0:
    print(f'\n-- QP energy, {size} rank(s)')
if size > 1:
    dist = solve_qp_energy_space_time(mf, mol, nocc, nocc-1, factors=F,
                                      distribute=True) * HARTREE_TO_EV
    d = abs(dist - serial)
    all_ok &= check(d < 1e-9, f'distributed over {size} ranks == serial',
                    f'{serial:.9f} vs {dist:.9f}, |d| = {d:.2e} eV')
else:
    check(True, 'serial reference', f'{serial:.9f} eV -- rerun under mpirun to compare')

if rank == 0:
    print('\n' + ('All checks passed.' if all_ok else 'FAILURES above.'))
sys.exit(0 if all_ok else 1)

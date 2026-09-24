"""The dense Casida solve on ELPA, over MPI ranks.

    srun -n 4 python examples/15_distributed_eigensolve.py     # or mpirun -n 4
    python examples/15_distributed_eigensolve.py               # one process

Rank 0 runs the calculation alone; the other ranks wait in
serve_distributed_solves() to take part in its eigensolves. A solve of
dimension `threshold` or more goes to ELPA whenever mpi4py and pyelpa import,
otherwise to scipy.linalg.eigh; the numbers are the same either way. The
threshold is 5000 by default and lowered here so that water, 95 pair states
at cc-pVDZ, takes the distributed route. Setup and Slurm example: the README's
"Distributed eigensolve" section.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.utils.linearAlgebra.diagonalization import (serve_distributed_solves,
                                                          release_workers)

# before anything imports mpi4py: pyelpa refuses to load after it
if serve_distributed_solves():
    sys.exit(0)                     # a worker rank, released by rank 0 at the end

from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

THRESHOLD = 50

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
nocc = mol.nelectron // 2
mf = scf.RHF(mol)
mf.kernel()

eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
resp = LinearResponseSolver(mf.mo_energy, eri_chemist=eri, spin_mode='restricted')
A, B = resp.build_casida_matrices(nocc)
n_pairs = A.shape[0]
res = CasidaSolver(A, B).solve(threshold=THRESHOLD)
omega, X, Y = res                   # X and Y whole, on rank 0

print(f'pair states = {n_pairs}   distributed = {res.is_distributed}')
print('RPA excitation energies: '
      + ', '.join(f'{w * HARTREE_TO_EV:.3f}' for w in omega[:4]) + ' eV')

release_workers()

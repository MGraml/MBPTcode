import os

import numpy as np
import scipy.linalg as la

# mpi4py/ELPA are imported ON DEMAND, never at module import time.
#
# WHY (this is not a style preference): `from mpi4py import MPI` runs MPI_Init at
# import. If the interconnect is unavailable -- e.g. a compute node whose IB
# device returns I/O errors -- MPI ABORTS THE PROCESS from C. It does not raise,
# so the try/except that used to wrap this import could never catch it, and
# merely importing this module killed the job with a bare
#     Abort(...) Fatal error in internal_Init_thread ... ucx function returned
#     with failed status
# and no Python traceback. That reached the periodic RPA driver via
# casida.py -> diagonalization.py, code that never uses ELPA at all.
#
# Deferring the import is the whole fix: code that never asks for a large
# diagonalization (the periodic RPA path) now never initializes MPI, so it cannot
# abort. Code that DOES cross the ELPA threshold still gets ELPA automatically,
# exactly as before -- no behaviour change where ELPA was actually wanted.
# Set WICKS_USE_ELPA=0 to force the scipy path even above the threshold (useful
# on a node with a broken interconnect, where MPI_Init would abort).
MPI = None
ElpaEigensolver = None
HAS_MPI = False
_MPI_TRIED = False


def _try_init_mpi():
    """Import mpi4py + ELPA on first real use. Returns True if usable."""
    global MPI, ElpaEigensolver, HAS_MPI, _MPI_TRIED
    if _MPI_TRIED:
        return HAS_MPI
    _MPI_TRIED = True
    if os.environ.get('WICKS_USE_ELPA', '').lower() in ('0', 'false', 'no'):
        return False                                  # explicit opt-OUT only
    try:
        from mpi4py import MPI as _MPI                       # may MPI_Init here
        from elpa.elpa import ElpaEigensolver as _Elpa
        MPI, ElpaEigensolver, HAS_MPI = _MPI, _Elpa, True
    except (ImportError, RuntimeError):
        HAS_MPI = False
    return HAS_MPI

def get_global_indices_1d(local_size, block_size, grid_dim, process_coord):
    """Generates global coordinates mapping for 2D block-cyclic layout."""
    local_indices = np.arange(local_size)
    block_num_local = local_indices // block_size
    offset_in_block = local_indices % block_size
    global_block_num = block_num_local * grid_dim + process_coord
    global_indices = global_block_num * block_size + offset_in_block
    return global_indices

def gather_block_cyclic(Z_local, global_N, solver, comm):
    """Gathers distributed block-cyclic matrix Z_local to Rank 0."""
    rank = comm.Get_rank()
    global_i = get_global_indices_1d(Z_local.shape[0], solver.Nb, solver.Pr, solver.my_prow)
    global_j = get_global_indices_1d(Z_local.shape[1], solver.Nb, solver.Pc, solver.my_pcol)

    local_data = (global_i, global_j, Z_local)
    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        Z_full = np.empty((global_N, global_N), dtype=Z_local.dtype)
        for idx_i, idx_j, chunk in all_data:
            Z_full[np.ix_(idx_i, idx_j)] = chunk
        return Z_full
    return None

def scatter_block_cyclic(matrix_full, solver, comm):
    """Scatters global matrix on Rank 0 to all ranks as block-cyclic chunks."""
    rank = comm.Get_rank()
    if rank == 0:
        send_data = []
        for target_rank in range(comm.Get_size()):
            target_prow = target_rank % solver.Pr
            target_pcol = target_rank // solver.Pr

            local_rows = solver._numroc(solver.global_N, solver.Nb, target_prow, 0, solver.Pr)
            local_cols = solver._numroc(solver.global_N, solver.Nb, target_pcol, 0, solver.Pc)

            idx_i = get_global_indices_1d(local_rows, solver.Nb, solver.Pr, target_prow)
            idx_j = get_global_indices_1d(local_cols, solver.Nb, solver.Pc, target_pcol)

            chunk = matrix_full[np.ix_(idx_i, idx_j)]
            send_data.append(chunk)
    else:
        send_data = None

    local_chunk = comm.scatter(send_data, root=0)
    return local_chunk

def diagonalize_matrix(M, threshold=5000):
    """Diagonalize symmetric M: distributed ELPA if dim >= threshold and MPI available, else local scipy.linalg.eigh.

    Returns (eigenvalues, Z, is_distributed, solver, comm); Z is a local chunk if distributed, else full.
    """
    global_N = M.shape[0]

    # size check FIRST, so MPI is never initialized for small matrices
    if global_N >= threshold and _try_init_mpi():
        try:
            comm = MPI.COMM_WORLD
            solver = ElpaEigensolver(global_N=global_N, block_size=64, comm=comm)

            global_i = get_global_indices_1d(solver.local_rows, solver.Nb, solver.Pr, solver.my_prow)
            global_j = get_global_indices_1d(solver.local_cols, solver.Nb, solver.Pc, solver.my_pcol)

            M_local = M[np.ix_(global_i, global_j)]
            eigenvalues, Z_local = solver.solve(M_local)
            return eigenvalues, Z_local, True, solver, comm
        except Exception:
            pass

    eigenvalues, Z = la.eigh(M, driver='evd')
    return eigenvalues, Z, False, None, None

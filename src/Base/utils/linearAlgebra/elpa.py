import numpy as np
from mpi4py import MPI
import elpa

class ElpaEigensolver:
    """Distributed eigensolver for symmetric matrices using ELPA and ScaLAPACK 2D block-cyclic data distribution."""
    def __init__(self, global_N, block_size=64, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.global_N = global_N
        self.Nb = block_size

        # 2D process grid (Pr x Pc)
        self.Pc = int(np.floor(np.sqrt(self.size)))
        while self.size % self.Pc != 0:
            self.Pc -= 1
        self.Pr = self.size // self.Pc

        self.my_prow = self.rank % self.Pr
        self.my_pcol = self.rank // self.Pr

        self.local_rows = self._numroc(self.global_N, self.Nb, self.my_prow, self.Pr)
        self.local_cols = self._numroc(self.global_N, self.Nb, self.my_pcol, self.Pc)

        self.elpa_ctx = elpa.Elpa()
        self._configure_elpa()

    def _numroc(self, n, nb, iproc, isrcproc, nprocs):
        """ScaLAPACK's NUMROC (Number of Rows/Cols) utility function."""
        mydist = (nprocs + iproc - isrcproc) % nprocs
        nblocks = n // nb
        return (nblocks // nprocs) * nb + min(max(n - (nblocks * nb) - mydist * nb, 0), nb)

    def _configure_elpa(self):
        """Sets up the ELPA parameters before memory is allocated."""
        self.elpa_ctx.set("na", self.global_N)
        self.elpa_ctx.set("local_nrows", self.local_rows)
        self.elpa_ctx.set("local_ncols", self.local_cols)
        self.elpa_ctx.set("nblk", self.Nb)

        self.elpa_ctx.set("mpi_comm_parent", MPI._addressof(self.comm))
        self.elpa_ctx.set("process_row", self.my_prow)
        self.elpa_ctx.set("process_col", self.my_pcol)

        self.elpa_ctx.setup()

        self.elpa_ctx.set("solver", elpa.SOLVER_2STAGE)

        try:
            self.elpa_ctx.set("nvidia-gpu", 1)
        except Exception:
            pass

    def solve(self, M_local):
        """Diagonalize the local block-cyclic chunk M_local; returns (eigenvalues, Z_local)."""
        if M_local.shape != (self.local_rows, self.local_cols):
            raise ValueError(f"Rank {self.rank}: Expected M_local shape "
                             f"({self.local_rows}, {self.local_cols}), got {M_local.shape}")

        M_local_f = np.asfortranarray(M_local, dtype=np.float64)
        Z_local_f = np.zeros_like(M_local_f, order='F')
        eigenvalues = np.zeros(self.global_N, dtype=np.float64)

        self.elpa_ctx.eigenvectors(M_local_f, eigenvalues, Z_local_f)
        return eigenvalues, Z_local_f

    def destroy(self):
        """Clean up context."""
        if hasattr(self, 'elpa_ctx'):
            del self.elpa_ctx
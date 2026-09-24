"""MPI transfers: exact results, and no single message over the count cap.

An MPI count is a C int, so one message holds at most 2**31 - 1 elements; above
that the library raises MPI_ERR_ARG. Two call sites carry arrays that grow with
the system and therefore go in pieces of at most `_MAX_MESSAGE` elements:

  * the block-cyclic scatter and gather of the ELPA route: a rank's chunk is about
    N^2 / P elements, so 8 ranks crossed the bound at N = 131072 (a Casida matrix
    of 134640 pair states);
  * `reduce_sum` of the space-time GW grid loops: a chi0 of nfreq x naux^2
    crosses it near naux = 8500 at 30 frequencies.

No MPI here: eight threads stand in for eight ranks, joined by a comm that
refuses any message over a limit. With the cap lowered to that limit, an array
that went whole would be refused, so an exact result is the proof that every
message was split.

Run: python tests/test_mpi_message_cap.py
"""
import os
import queue
import sys
import threading
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# elpa.py imports pyelpa and mpi4py at module scope; neither is needed for the
# transfers, and pyelpa is not installable from PyPI, so stub them.
for name, attrs in (('pyelpa', {'Elpa': object}),
                    ('mpi4py', {'MPI': types.SimpleNamespace(COMM_WORLD=None)})):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

from src.Base.utils import mpi_grid as mg
from src.Base.utils.linearAlgebra import diagonalization as dg
from src.Base.utils.linearAlgebra.elpa import ElpaEigensolver

INT_MAX = 2**31 - 1


class World:
    """What the thread ranks share: mailboxes, a barrier, a reduction table."""

    def __init__(self, size, limit):
        self.size, self.limit, self.sizes = size, limit, []
        self.boxes = {(a, b): queue.Queue()
                      for a in range(size) for b in range(size) if a != b}
        self.barrier = threading.Barrier(size)
        self.table = {}


class ThreadComm:
    """Blocking transfers between thread ranks; refuses a message over the limit."""

    def __init__(self, rank, world):
        self.rank, self.w = rank, world

    def Get_rank(self):
        """This thread's rank."""
        return self.rank

    def Get_size(self):
        """The number of thread ranks."""
        return self.w.size

    def bcast(self, obj, root=0):
        """Hand `obj` from root to every other rank, as mpi4py's bcast."""
        if self.rank == root:
            for dest in range(self.w.size):
                if dest != root:
                    self.w.boxes[root, dest].put(('bcast', obj))
            return obj
        return self.w.boxes[root, self.rank].get(timeout=5)[1]

    def _count(self, buf):
        if buf.size > self.w.limit:
            raise RuntimeError(f'{buf.size} elements in one message, '
                               f'limit {self.w.limit}')
        self.w.sizes.append(buf.size)

    def Send(self, buf, dest, tag):
        """Post a copy of `buf` to `dest`; refused over the limit."""
        buf = np.asarray(buf)
        self._count(buf)
        self.w.boxes[self.rank, dest].put((tag, buf.copy()))

    def Recv(self, buf, source, tag):
        """Fill `buf` from the next message of `source`, which must match."""
        if buf.size > self.w.limit:
            raise RuntimeError(f'{buf.size} elements in one message, '
                               f'limit {self.w.limit}')
        got_tag, data = self.w.boxes[source, self.rank].get(timeout=5)
        if got_tag != tag or data.shape != buf.shape:
            raise RuntimeError(f'expected tag {tag} shape {buf.shape}, '
                               f'got tag {got_tag} shape {data.shape}')
        buf[...] = data

    def Allreduce(self, sendbuf, recvbuf, op):
        """Sum `recvbuf` over all ranks in place; refused over the limit."""
        # in place, as reduce_sum calls it; the op is taken to be the sum
        if sendbuf is not mg.MPI.IN_PLACE:
            raise RuntimeError('reduce_sum is expected to reduce in place')
        try:
            self._count(recvbuf)
        except RuntimeError:
            self.w.barrier.abort()
            raise
        self.w.table[self.rank] = recvbuf.copy()
        self.w.barrier.wait(timeout=5)
        total = sum(self.w.table[r] for r in range(self.w.size))
        self.w.barrier.wait(timeout=5)
        recvbuf[...] = total


def on_threads(size, limit, body):
    """Run body(rank, comm) on `size` thread ranks; (results, first error)."""
    world, out = World(size, limit), {}

    def rank_main(rank):
        try:
            out[rank] = body(rank, ThreadComm(rank, world))
        except Exception as exc:
            out[rank] = exc

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = [f'rank {r}: {v}' for r, v in sorted(out.items())
              if isinstance(v, Exception)]
    return out, (errors[0] if errors else ''), world.sizes


def solver_stub(n, nb, pr, pc):
    """What `_chunk_indices` reads off an ElpaEigensolver, without ELPA."""
    return types.SimpleNamespace(
        global_N=n, Nb=nb, Pr=pr, Pc=pc,
        _numroc=lambda *a: ElpaEigensolver._numroc(None, *a))


def round_trip(n, nb, pr, pc, limit):
    """Scatter an n x n matrix over pr*pc thread ranks and gather it back."""
    M = np.arange(n * n, dtype=np.float64).reshape(n, n)
    solver = solver_stub(n, nb, pr, pc)

    def body(rank, comm):
        local = dg.scatter_block_cyclic(M if rank == 0 else None, solver, comm)
        return local, dg.gather_block_cyclic(local, n, solver, comm)

    out, err, sizes = on_threads(pr * pc, limit, body)
    if err:
        return False, err, sizes
    chunks_ok = all(
        np.array_equal(out[r][0], M[np.ix_(*dg._chunk_indices(solver, r))])
        for r in range(pr * pc))
    return chunks_ok and np.array_equal(out[0][1], M), '', sizes


def all_reduce(size, limit, make):
    """reduce_sum of make(rank) over thread ranks; True when every rank has the sum."""
    want = sum(make(r) for r in range(size))

    def body(rank, comm):
        a = make(rank)
        back = mg.reduce_sum(a, comm)
        return back is a, a

    out, err, sizes = on_threads(size, limit, body)
    if err:
        return False, err, sizes
    same = all(out[r][0] for r in range(size))
    exact = all(np.array_equal(out[r][1], want) for r in range(size))
    return same and exact, '' if same else 'reduce_sum did not return its input', sizes


def with_cap(cap, fn, *args):
    """Call fn with both modules' _MAX_MESSAGE lowered to cap.

    Code without the cap does not read it, so there the call sends whole arrays
    and the limit refuses them: a FAIL, not a crash.
    """
    saved = {m: getattr(m, '_MAX_MESSAGE', None) for m in (dg, mg)}
    for m in saved:
        m._MAX_MESSAGE = cap
    try:
        return fn(*args)
    finally:
        for m, value in saved.items():
            if value is None:
                del m._MAX_MESSAGE
            else:
                m._MAX_MESSAGE = value


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def main():
    """Run every section; 0 when all pass, else 1."""
    ok = True
    mg.MPI = types.SimpleNamespace(IN_PLACE=object(), SUM=object())

    print('\n=== 1. the default cap fits an MPI count ===')
    caps = {m.__name__.split('.')[-1]: getattr(m, '_MAX_MESSAGE', None)
            for m in (dg, mg)}
    ok &= check(all(c is not None and 0 < c <= INT_MAX for c in caps.values()),
                '_MAX_MESSAGE <= 2**31 - 1 where each transfer reads it', f'{caps}')

    limit = 97
    print('\n=== 2. scatter and gather under a message limit, 8 ranks on 4 x 2 ===')
    # 203 = 25 blocks of 8 and a remainder of 3: every chunk shape differs, and
    # rows of 104 and 99 elements are wider than the limit
    passed, err, sizes = with_cap(limit, round_trip, 203, 8, 4, 2, limit)
    ok &= check(passed, 'every chunk and the gathered matrix exact',
                err or f'{len(sizes)} messages')
    ok &= check(passed and max(sizes) <= limit and len(sizes) > 2 * 7,
                'no message over the cap, and chunks were split',
                f'largest {max(sizes) if sizes else 0} of {limit} elements')

    print('\n=== 3. the default cap sends a small chunk whole ===')
    passed, err, sizes = round_trip(203, 8, 4, 2, INT_MAX)
    ok &= check(passed and len(sizes) == 2 * 7,
                'one Send per worker chunk each way',
                err or f'{len(sizes)} messages for 7 workers')

    print('\n=== 4. reduce_sum under a message limit, 8 ranks ===')
    # 3 x 17 x 23 = 1173 elements, 13 pieces with a short last one
    def real_t(r):
        return (np.arange(1173.0).reshape(23, 17, 3) * (r + 1)).T

    def cplx(r):
        return np.arange(1173.0).reshape(3, 17, 23) * (1 + 1j) ** r

    for label, make in (('real, not contiguous (the copy-back path)', real_t),
                        ('complex, contiguous', cplx)):
        passed, err, sizes = with_cap(limit, all_reduce, 8, limit, make)
        ok &= check(passed and max(sizes) <= limit and len(sizes) > 8,
                    f'every rank holds the exact sum, {label}',
                    err or f'{len(sizes)} messages, largest {max(sizes)}')

    print('\n=== 5. the default cap reduces a small array in one call ===')
    passed, err, sizes = all_reduce(8, INT_MAX, cplx)
    ok &= check(passed and len(sizes) == 8, 'one Allreduce per rank',
                err or f'{len(sizes)} messages for 8 ranks')

    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

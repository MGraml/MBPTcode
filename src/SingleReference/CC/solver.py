"""Lambda-CCSDT iterative solve + CCSDT 1-RDM assembly, given converged T1/T2/T3.

lambdaK_residual() outputs shape (nv,..,no,..) (matching T's convention), but
l1/l2/l3 themselves are consumed elsewhere (e.g. d1_blocks.py) in the generator's
canonical (no,..,nv,..) order -- transpose each residual before use.

Plain fixed-point iteration on the Lambda equations is not contractive at
this system size; DIIS is not optional here.
"""
import numpy as np

from . import lambda1_residual as L1M
from . import lambda2_residual as L2M
from . import d1_blocks as D1M
from .diis import DIIS

try:
    from . import lambda3_residual as L3M
    HAVE_L3 = True
except ImportError:
    HAVE_L3 = False


def solve_lambda_ccsdt(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk,
                       max_iter=300, stopping_eps=1e-8, verbose=True,
                       diis_size=15, diis_start_cycle=2):
    nv, no = t1.shape

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)

    le_ai = e_ai.transpose(1, 0)
    lfock_e_ai = fock_e_ai.transpose(1, 0)
    le_abij = e_abij.transpose(2, 3, 0, 1)
    lfock_e_abij = fock_e_abij.transpose(2, 3, 0, 1)
    if HAVE_L3:
        le_abcijk = e_abcijk.transpose(3, 4, 5, 0, 1, 2)
        lfock_e_abcijk = fock_e_abcijk.transpose(3, 4, 5, 0, 1, 2)

    # standard CC-Lambda starting point
    l1 = t1.transpose(1, 0).copy()
    l2 = t2.transpose(2, 3, 0, 1).copy()
    l3 = t3.transpose(3, 4, 5, 0, 1, 2).copy() if HAVE_L3 else np.zeros((no, no, no, nv, nv, nv))
    kd = np.eye(no + nv)

    l1_dim = l1.size
    l2_dim = l2.size
    diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
    old_vec = np.hstack((l1.ravel(), l2.ravel(), l3.ravel()))

    if verbose:
        # flush=True: same batch-scheduler buffering reason as amplitudes.kernel
        print("    ==> Lambda-CCSDT amplitude equations (DIIS) <==", flush=True)
        print("     Iter          |dL1|          |dL2|          |dL3|", flush=True)

    for it in range(max_iter):
        r1 = L1M.lambda1_residual(t1, t2, t3, l1, l2, l3, fock, g, o, v).transpose(1, 0)
        r2 = L2M.lambda2_residual(t1, t2, t3, l1, l2, l3, fock, g, o, v).transpose(2, 3, 0, 1)
        if HAVE_L3:
            r3 = L3M.lambda3_residual(t1, t2, t3, l1, l2, l3, fock, g, kd, o, v).transpose(3, 4, 5, 0, 1, 2)
        else:
            r3 = np.zeros_like(l3)

        l1_new = (r1 + lfock_e_ai * l1) * le_ai
        l2_new = (r2 + lfock_e_abij * l2) * le_abij
        l3_new = (r3 + lfock_e_abcijk * l3) * le_abcijk if HAVE_L3 else l3

        d1 = np.linalg.norm(l1_new - l1)
        d2 = np.linalg.norm(l2_new - l2)
        d3 = np.linalg.norm(l3_new - l3) if HAVE_L3 else 0.0

        vec = np.hstack((l1_new.ravel(), l2_new.ravel(), l3_new.ravel()))
        error_vec = old_vec - vec
        try:
            vec = diis_update.compute_new_vec(vec, error_vec)
        except np.linalg.LinAlgError:
            # tiny systems can converge so fast the DIIS error vectors become
            # exactly degenerate (singular B-matrix) -- fall back to the
            # undamped update for this step rather than crashing.
            pass
        old_vec = vec

        l1 = vec[:l1_dim].reshape(l1.shape)
        l2 = vec[l1_dim:l1_dim + l2_dim].reshape(l2.shape)
        l3 = vec[l1_dim + l2_dim:].reshape(l3.shape)

        if verbose:
            print(f"     {it:4d}   {d1:.3e}      {d2:.3e}      {d3:.3e}", flush=True)

        if d1 < stopping_eps and d2 < stopping_eps and d3 < stopping_eps:
            break
    else:
        raise ValueError("Lambda-CCSDT iterations did not converge")

    return l1, l2, l3


def ccsdt_one_rdm(t1, t2, t3, l1, l2, l3, o, v):
    nv, no = t1.shape
    nmo = no + nv
    kd = np.eye(nmo)

    opdm = np.zeros((nmo, nmo))
    opdm[:no, :no] = D1M.d1_oo(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[no:, no:] = D1M.d1_vv(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[:no, no:] = D1M.d1_ov(t1, t2, t3, l1, l2, l3, kd, o, v)
    opdm[no:, :no] = D1M.d1_vo(t1, t2, t3, l1, l2, l3, kd, o, v)
    return opdm

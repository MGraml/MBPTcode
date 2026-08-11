# GENERATED CODE -- CCSDT 1-RDM blocks (oo/vv/ov/vo), machine-generated.
# l1/l2/l3 are consumed in the canonical (no,..,nv,..) index order
# (unlike the residual outputs -- see solver.py). Do not edit by hand.
import numpy as np
from numpy import einsum


def d1_oo(t1, t2, t3, l1, l2, l3, kd, o, v):
    nv, no = t1.shape
    opdm_oo = np.zeros((no, no))
    opdm_oo +=  1.00 * einsum('mn->mn', kd[o, o])
    opdm_oo += -1.00 * einsum('am,na->mn', t1, l1)
    opdm_oo += -0.50 * einsum('baim,inba->mn', t2, l2)
    opdm_oo += -0.083333333333333 * einsum('cbaijm,ijncba->mn', t3, l3)
    return opdm_oo


def d1_vv(t1, t2, t3, l1, l2, l3, kd, o, v):
    nv, no = t1.shape
    opdm_vv = np.zeros((nv, nv))
    opdm_vv +=  1.00 * einsum('fi,ie->ef', t1, l1)
    opdm_vv +=  0.50 * einsum('faij,ijea->ef', t2, l2)
    opdm_vv +=  0.083333333333333 * einsum('fbaijk,ijkeba->ef', t3, l3)
    return opdm_vv


def d1_ov(t1, t2, t3, l1, l2, l3, kd, o, v):
    nv, no = t1.shape
    opdm_ov = np.zeros((no, nv))
    opdm_ov +=  1.00 * einsum('em->me', t1)
    opdm_ov += -1.00 * einsum('eaim,ia->me', t2, l1)
    opdm_ov +=  0.250 * einsum('ebaijm,ijba->me', t3, l2)
    opdm_ov += -0.50 * einsum('baim,ej,ijba->me', t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm_ov += -0.50 * einsum('eaij,bm,ijba->me', t2, t1, l2, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm_ov += -0.083333333333333 * einsum('cbaijm,ek,ijkcba->me', t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm_ov += -0.083333333333333 * einsum('ebaijk,cm,ijkcba->me', t3, t1, l3, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm_ov += -1.00 * einsum('am,ei,ia->me', t1, t1, l1, optimize=['einsum_path', (0, 2), (0, 1)])
    opdm_ov += -0.250 * einsum('baim,ecjk,ijkcba->me', t2, t2, l3, optimize=['einsum_path', (0, 2), (0, 1)])
    return opdm_ov


def d1_vo(t1, t2, t3, l1, l2, l3, kd, o, v):
    nv, no = t1.shape
    opdm_vo = np.zeros((nv, no))
    opdm_vo +=  1.00 * einsum('me->em', l1)
    return opdm_vo

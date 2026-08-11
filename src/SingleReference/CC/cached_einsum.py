"""Path-caching drop-in replacement for numpy.einsum, used by generated CC
residual/density code.

Every generated function calls numpy.einsum(..., optimize=True) many times
per DIIS iteration on operands whose shapes never change across iterations
(no/nv are fixed for a given solve). Plain numpy re-runs its full
einsum_path search on every single call regardless of that -- profiling
showed this costs ~20% of restricted CCSDT/Lambda3 wall time on Ne/cc-pVDZ.
This wrapper memoizes the contraction path by (subscripts, operand shapes)
and reuses it; the arithmetic performed is identical to plain
optimize=True, just without recomputing the path each call.

'optimal' path search is combinatorial in operand count (brute-force over
every contraction order) -- fine for the <=4-operand contractions ordinary
generated CC/density code has, but laplace_codegen.py's Laplace-fused terms
routinely carry 7-9 operands (a couple of real tensors plus one Oe/Ve
dressing vector per amplitude external index) and 'optimal' search there
was measured to hang (still not finished after 2+ minutes) where 'greedy'
finds a path in under a millisecond. Only the ONE-TIME path-finding cost
differs between the two strategies (memoized afterward same as before);
'greedy' can in principle pick a very slightly less optimal contraction
order, but for these shapes the difference is not measurable and "finishes
at all" dominates.

Path search uses the standalone opt_einsum package's contract_path, NOT
numpy's own built-in einsum_path -- found (profiling DF-ADC(2)-x
MP3-density rollout on CO/aug-cc-pvdz) that numpy's own 'optimal' path
search silently returns a badly wrong (non-optimal, despite the name) path
for "triangle"-connectivity contractions: three operands where each PAIR
shares exactly one summed index but no single index is shared by all three
(e.g. 'Qac,Qbd,cdij->abij', the DF-dressed vvvv-type term df_codegen.py
produces -- two B-factors sharing the auxiliary axis Q, each also sharing a
different axis with the amplitude). Measured on that exact term
(naux=144, nv=39, no=7): numpy's own optimizer reports "Optimized scaling"
identical to "Naive scaling" (no improvement found at all) and picks a
single 5e10-FLOP direct contraction, timing at 12+ seconds; opt_einsum's
own contract_path finds the correct 2-step factorization (8.9e8 FLOPs, a
measured 55x reduction) in milliseconds. This pattern never arose before
the DF rewrite (a dense g_xxxx factor was always a single
pre-formed operand, never two separately-contracted DF factors sharing an
auxiliary axis), so it was never exercised by earlier CC/CCSD(T) profiling.
opt_einsum.contract_path's returned path is a bare list of index-pair
tuples (NOT numpy's own ['einsum_path', ...]-prefixed format) -- prepend
'einsum_path' before handing it to np.einsum(optimize=path); verified this
combination reproduces opt_einsum.contract's own numeric result exactly and
runs at opt_einsum's (not numpy's) speed.
"""
import os

import numpy as np
import opt_einsum

_PATH_CACHE = {}
_OPTIMAL_MAX_OPERANDS = 4

# Debug-only rank ceiling, gated by an env var so it costs nothing (and
# doesn't need opt_einsum) when unset -- set WICKS_MAX_EINSUM_RANK=4 to
# enforce, e.g., the MP2/MP3 density paths' "no rank>4 array" invariant
# (see src/SingleReference/DensityMatrix/mpn_density_driver_restricted.py/
# mpn_density_driver_unrestricted.py's Laplace-fusion docstrings). Checks
# BOTH the ndim of every operand passed in AND the ndim of every intermediate
# opt_einsum's own contraction path would materialize -- a fused multi-
# operand einsum whose optimal path happens to materialize a larger
# intermediate than any single operand is just as much a violation as a
# literal rank>4 array showing up as a Python variable. Only runs once per
# distinct (subscripts, shapes) key, the same lifetime as the path cache
# itself, so it doesn't add per-call overhead in a hot loop.
_ASSERT_MAX_RANK = os.environ.get('WICKS_MAX_EINSUM_RANK')


def _assert_ranks_ok(subscripts, operands, strategy, max_rank):
    for op in operands:
        if op.ndim > max_rank:
            raise AssertionError(
                f"cached_einsum: operand of rank {op.ndim} > {max_rank} in '{subscripts}'")
    _, info = opt_einsum.contract_path(subscripts, *operands, optimize=strategy)
    for step in info.contraction_list:
        einsum_str = step[2]
        out_rank = len(einsum_str.split('->')[1])
        if out_rank > max_rank:
            raise AssertionError(
                f"cached_einsum: intermediate of rank {out_rank} > {max_rank} in "
                f"'{subscripts}' (contraction step '{einsum_str}')")


def einsum(subscripts, *operands, optimize=True):
    if optimize is True and len(operands) > 1:
        key = (subscripts, tuple(op.shape for op in operands))
        path = _PATH_CACHE.get(key)
        if path is None:
            strategy = 'optimal' if len(operands) <= _OPTIMAL_MAX_OPERANDS else 'greedy'
            oe_path, _ = opt_einsum.contract_path(subscripts, *operands, optimize=strategy)
            path = ['einsum_path'] + list(oe_path)
            _PATH_CACHE[key] = path
            if _ASSERT_MAX_RANK is not None:
                _assert_ranks_ok(subscripts, operands, strategy, int(_ASSERT_MAX_RANK))
        optimize = path
    return np.einsum(subscripts, *operands, optimize=optimize)

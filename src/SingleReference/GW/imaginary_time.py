"""GW self-energy in imaginary time.

The standard space-time construction: Sigma = i G W is a pointwise product in
imaginary time, so the self-energy costs no convolution.

References
----------
Rojas, Godby and Needs, Phys. Rev. Lett. 74, 1827 (1995) -- the space-time
method itself.
Foerster and Visscher, J. Chem. Theory Comput. 16, 7381 (2020) -- the same
construction in a localized basis, with PADF as the factorization.
Duchemin and Blase, J. Chem. Phys. 150, 174120 (2019) and J. Chem. Theory
Comput. 17, 2383 (2021) -- the separable RI (ISDF) used here in place of PADF,
which is what makes the full self-energy build O(N^3).
"""
import os

import numpy as np

from src.Base.utils.time_frequency import (minimax_transform_weights,
                                          minimax_points_for_accuracy,
                                          COSINE_TW, COSINE_WT, SINE_TW)
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.space_time import (
    polarizability_projected_tau)


def greens_function_imaginary_time(X, eps, nocc, tau, mu=None):
    """
    (Ghat_lesser, Ghat_greater) projected onto the THC grid, each (M, M).
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    X_o, X_v = X[:, occ], X[:, virt]
    e_o, e_v = eps[occ] - mu, eps[virt] - mu
    G_lesser = (X_o * np.exp(e_o * tau)) @ X_o.T          # occupied branch
    G_greater = -(X_v * np.exp(-e_v * tau)) @ X_v.T       # virtual branch
    return G_lesser, G_greater


def self_energy_imaginary_time(X, D, W_tilde_aux_tau, eps, nocc, tau_points,
                               mu=None):
    """Sigma^c(i.tau) in the AO/MO basis X was built in, shape (ntau, n, n) x 2.

    W_tilde_aux_tau : (ntau, naux, naux) the correlation part of the screened
        interaction, W - V, already on the imaginary-time axis.

    Returns (Sigma_lesser, Sigma_greater). The frequency-axis self-energy
    follows by transforming their SUM with the cosine kernel and their
    DIFFERENCE with the sine kernel -- the even and odd parts of Sigma(i.tau)
    respectively. Both transforms are carried by TimeFrequencyGrid, which
    carries the reference for them.
    """
    n = X.shape[1]
    ntau = len(tau_points)
    sig_l = np.empty((ntau, n, n))
    sig_g = np.empty((ntau, n, n))
    for k, tau in enumerate(tau_points):
        G_l, G_g = greens_function_imaginary_time(X, eps, nocc, tau, mu=mu)
        Zt = D @ W_tilde_aux_tau[k] @ D.T                 # (M, M)
        sig_l[k] = X.T @ (Zt * G_l) @ X
        sig_g[k] = X.T @ (Zt * G_g) @ X
    return sig_l, sig_g



def self_energy_fit_ranges(eps, nocc, mu=None):
    """
    The two exponential-decay ranges the space-time self-energy needs.

    * Wt(i.tau): the RPA screened interaction has spectral weight BELOW the
      smallest independent-particle transition -- collective excitations sit
      under the HOMO-LUMO gap -- so a range starting at the gap misfits exactly
      where Wt is largest. Measured on the Wt(i.w) -> tau -> Wt(i.w) round trip
      at ntau=18: 1.3e-3 with [gap, e_max], 5.0e-8 with the widened range.
    * Sigma(i.tau) = -G Wt is a PRODUCT, so its decay rates are SUMS
      |eps_m - mu| + Omega_S and reach far beyond either factor's range.

    Returns ((w_lo, w_hi), (sig_lo, sig_hi)).
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    w_lo = eps[virt].min() - eps[occ].max()
    w_hi = eps[virt].max() - eps[occ].min()
    dG = np.abs(eps - mu)
    return ((0.3 * w_lo, 3.0 * w_hi),
            (0.3 * (dG.min() + w_lo), 3.0 * (dG.max() + w_hi)))


#: Kaltak-Klimes-Kresse residual that buys ~0.1 meV on the quasiparticle
#: energy. Calibrated on naphthalene/cc-pVDZ at R_Sigma = 486, against the
#: ntau -> infinity limit of the space-time HOMO:
#:     eta 1.2e-08 -> 1.96 meV     eta 6.0e-10 -> 0.32 meV
#:     eta 4.8e-11 -> 0.08 meV     eta 3.7e-12 -> 0.03 meV
#: The mapping is empirical -- eta bounds the quadrature, not the self-energy
#: it is used to integrate -- so this is a calibrated default, not a bound.
DEFAULT_TAU_TARGET = 1e-10


def minimax_points_for_gw(eps, nocc, mu=None, target=DEFAULT_TAU_TARGET,
                          npoints_max=34):
    """Smallest minimax time grid that resolves every range the route integrates.

    The space-time route uses ONE ntau for three different fits, over three
    different energy ranges, and the widest one binds:

        chi0(i.tau) -> chi0(i.omega)   [e_min, e_max]        the bare gap ratio
        W(i.omega)  -> W(i.tau)        rW, widened below the gap
        Sigma(i.tau)-> Sigma(i.omega)  rS, widest -- Sigma is a PRODUCT, so its
                                       decay rates are SUMS (see
                                       `self_energy_fit_ranges`)

    R = e_max/e_min grows as the gap closes, so this necessarily returns more
    points for a longer acene than a shorter one at fixed accuracy: naphthalene
    needs 16 where hexacene needs 18. A hardcoded ntau is therefore wrong at one
    end or the other of any size series.

    Returns (npoints, worst_error). If no tabulated size reaches `target` the
    best available is returned instead, and the second value is the accuracy
    actually obtained -- callers should not assume `target` was met.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = self_energy_fit_ranges(eps, nocc, mu=mu)
    ratios = ((eps[virt].max() - eps[occ].min())
              / (eps[virt].min() - eps[occ].max()),
              rW[1] / rW[0],
              rS[1] / rS[0])

    npoints, worst = 0, 0.0
    for R in ratios:
        n, err = minimax_points_for_accuracy(1.0, R, target=target,
                                             npoints_max=npoints_max)
        if n is None:                       # nothing tabulated resolved this R
            return npoints_max, float('inf')
        npoints, worst = max(npoints, n), max(worst, err)
    return npoints, worst


def _transform_screened(Ctw, W_omega, chunk_bytes=2 << 30):
    """
    sum_omega Ctw[.,omega] (W(i.omega) - I), without ever copying all of W.

    `W_omega - np.eye(naux)[None]` reads innocently but allocates a second
    (nfreq, naux, naux) -- 170 GB at 10k basis functions, on top of the W it is
    copying. Chunking over frequency bounds the temporary to `chunk_bytes`, and
    the identity comes off the diagonal in place on the chunk rather than from a
    broadcast np.eye. Pure: W_omega is not touched.
    """
    nfreq, naux = W_omega.shape[0], W_omega.shape[-1]
    step = max(1, min(nfreq, int(chunk_bytes // max(naux * naux * 8, 1))))
    dg = np.diag_indices(naux)
    out = np.zeros(Ctw.shape[:1] + (naux, naux))
    for k0 in range(0, nfreq, step):
        k1 = min(k0 + step, nfreq)
        blk = W_omega[k0:k1].copy()
        blk[(slice(None),) + dg] -= 1.0
        out += np.tensordot(Ctw[:, k0:k1], blk, axes=(1, 0))
        del blk
    return out

def screened_interaction_imaginary_time(W_omega, grid):
    """
    Wt(i.tau) = cosine transform of the correlation part W(i.omega) - I.
    """
    return _transform_screened(grid.cosft_tw, W_omega)


def screened_interaction_tau_blocked(X, D, eps, nocc, grid, Ctw, mu=None,
                                     freq_block=None, scratch_dir=None,
                                     tile_memory_gb=4.0, wt_scratch=None,
                                     static_index=None, static_out=None):
    """
    Wt(i.tau) = sum_w Ctw[.,w] ( [I - chi0(i.w)]^-1 - I ), in one blocked pass.

    The in-core route builds all of chi0 (nfreq, naux, naux), inverts it in
    place, then transforms -- so the whole frequency axis is live from the first
    tau to the last. It does not have to be. Frequency is never coupled to
    itself: the Dyson step is diagonal in omega, and the two transforms are
    fan-ins in opposite directions with W(i.omega) as the pivot. A block of
    frequencies can therefore be built, inverted, folded into Wt(i.tau), and
    dropped. Peak becomes

        Wt(tau)           ntau_out x naux^2     (unavoidable -- it is the result)
      + one freq block    freq_block x naux^2

    instead of nfreq x naux^2. At 10k basis functions, where naux^2 is 5.3 GB,
    that is 95 + 21 GB against 170.

    The catch is that every block needs all of proj(tau) again, and rebuilding
    those IS the N^3 cost of the method -- so recomputing them costs a factor
    nfreq/freq_block in time. `scratch_dir` avoids that by caching them: written
    on the first sweep, memory-mapped on the rest. It is OPT-IN, and off by
    default, because it puts ntau x naux^2 on disk (2.9 GB at undecacene, 95 GB
    at 10k) and only pays off when freq_block is genuinely smaller than nfreq.

    `wt_scratch` puts Wt ITSELF on disk, memory-mapped. freq_block bounds the
    frequency axis but Wt is the result and stays resident -- ntau x naux^2,
    which is 55 GB at the 476-atom hexamer/cc-pVDZ and ~135 GB at cc-pVTZ,
    where it does not fit alongside anything else. Every consumer touches it as
    Wt[k] inside a tau loop, exactly once per k, so a memmap streams it at no
    algorithmic cost.

    static_index / static_out: capture [I - chi0(i.omega_k)]^-1 for one
    frequency, before the identity is removed. Used to carry omega = 0 for a
    BSE's static screening on the axis the self-energy is already building.
    THE CALLER MUST ZERO THAT COLUMN OF Ctw: this routine folds every frequency
    it is given into Wt, and a passenger that is not part of the Sigma
    quadrature must contribute nothing to it.

    Returns Wt(i.tau) with shape (Ctw.shape[0], naux, naux), matching what
    `self_energy_frequency_from_time` builds internally.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    X_o = np.ascontiguousarray(X[:, occ])
    X_v = np.ascontiguousarray(X[:, virt])
    e_o, e_v = eps[occ] - mu, eps[virt] - mu

    naux, nfreq, ntau = D.shape[1], grid.nfreq, grid.ntau
    nb = int(freq_block or nfreq)
    if wt_scratch is not None:
        os.makedirs(os.path.dirname(wt_scratch) or '.', exist_ok=True)
        Wt = np.lib.format.open_memmap(
            wt_scratch, mode='w+', dtype=np.float64,
            shape=(Ctw.shape[0], naux, naux))
        Wt[:] = 0.0
    else:
        Wt = np.zeros((Ctw.shape[0], naux, naux))
    eye = np.eye(naux)
    dg = np.diag_indices(naux)

    cache, path = None, None
    if scratch_dir is not None:
        os.makedirs(scratch_dir, exist_ok=True)
        path = os.path.join(scratch_dir, f'proj_{ntau}_{naux}.npy')
        cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                          shape=(ntau, naux, naux))
    cached = False
    try:
        for k0 in range(0, nfreq, nb):
            k1 = min(k0 + nb, nfreq)
            blk = np.zeros((k1 - k0, naux, naux))
            for j in range(ntau):
                if cached:
                    proj = cache[j]
                else:
                    proj = polarizability_projected_tau(
                        X_o, X_v, e_o, e_v, D, grid.tau_points[j],
                        tile_memory_gb=tile_memory_gb)
                    if cache is not None:
                        cache[j] = proj
                blk += grid.cosft_wt[k0:k1, j, None, None] * proj
            if cache is not None:
                cached = True
            for m in range(k1 - k0):
                b = blk[m]
                b[:] = np.linalg.inv(eye - b)
                if static_index is not None and k0 + m == static_index:
                    static_out['w_static'] = b.copy()      # before -I
                b[dg] -= 1.0                  # the correlation part, W - I
            # ONE OUTPUT TAU AT A TIME. `Wt += tensordot(Ctw[:, k0:k1], blk)`
            # materializes a temporary the size of Wt itself, which is the very
            # array wt_scratch exists to keep off the heap -- and it is the
            # largest object in the route either way.
            for t in range(Ctw.shape[0]):
                Wt[t] += np.tensordot(Ctw[t, k0:k1], blk, axes=(0, 0))
            del blk
    finally:
        if cache is not None:
            del cache
            if path and os.path.exists(path):
                os.remove(path)
    if wt_scratch is not None:
        # Hand back an in-memory copy only if it fits is NOT the contract here:
        # the caller asked for Wt on disk precisely because it does not. Flush
        # and re-open read-only, so the consumer streams it and the file is the
        # caller's to remove -- it is the one object this routine cannot clean
        # up itself, since it is the return value.
        Wt.flush()
        Wt = np.lib.format.open_memmap(wt_scratch, mode='r')
    return Wt


def self_energy_frequency_from_time(X, D, W_omega, eps, nocc, tau_points,
                                    omega_in, omega_out, p_state, mu=None,
                                    ranges=None, tau_indices=None,
                                    Wt_tau=None):
    """
    Sigma^c_pp(i.omega) by the full space-time route
    tau kept through the
    self-energy, with a second Fourier transform back.

        Wt(i.omega) --cos--> Wt(i.tau)                    (even)
        Zt_PQ(tau)  = D Wt(tau) D^T
        Sigma^{<,>}_pq(tau) = sum_PQ X[P,p] (Zt * Ghat^{<,>})_PQ X[Q,q]
        Sigma(i.omega) = cos[Sigma^> + Sigma^<] + i sin[Sigma^> - Sigma^<]

    the last line being Foerster and Visscher eq 54 (JCTC 2020, 16, 7381),
    which follows from splitting

        Sigma(i.omega) = int_0^inf dtau [ e^{i omega tau} Sigma^>(tau)
                                        + e^{-i omega tau} Sigma^<(tau) ]

    into its even and odd parts. Both transform matrices are fitted to
    2 int_0^inf dtau cos/sin(omega tau) e^{-x tau}, i.e. they carry a factor 2
    relative to the bare half-line integral, hence the 1/2 below.

    See also from the Kresse group:
    Liu et al. PRB, 94, 165109 (2016)
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = ranges or self_energy_fit_ranges(eps, nocc, mu=mu)

    if Wt_tau is None:
        Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, omega_in, *rW)
        Wt_tau = _transform_screened(Ctw, W_omega)
    # else the caller built it blockwise and W_omega was never formed at all

    ntau = len(tau_points)
    # zeros, not empty: with tau_indices set only the owned entries are filled
    # and the rest must contribute nothing to the transform, which is a sum.
    states = np.atleast_1d(p_state)
    scalar = np.ndim(p_state) == 0
    sig_l = np.zeros((ntau, len(states)))
    sig_g = np.zeros((ntau, len(states)))
    Xp = np.ascontiguousarray(X[:, states])            # (M, nstates)
    X_o, X_v = X[:, occ], X[:, virt]
    e_o, e_v = eps[occ] - mu, eps[virt] - mu
    which = range(ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        tau = tau_points[k]
        G_l = (X_o * np.exp(e_o * tau)) @ X_o.T
        G_g = -(X_v * np.exp(-e_v * tau)) @ X_v.T
        Zt = D @ Wt_tau[k] @ D.T
        G_l *= Zt                                       # in place; Zt kept
        sig_l[k] = np.einsum('Pp,Pp->p', Xp, G_l @ Xp)
        G_g *= Zt
        sig_g[k] = np.einsum('Pp,Pp->p', Xp, G_g @ Xp)
        del Zt

    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS)
    out = -0.5 * ((sig_g + sig_l).T @ C.T + 1j * ((sig_g - sig_l).T @ S.T))
    return out[0] if scalar else out


def _morton_order(coords):
    """Z-order interpolation points so a contiguous block is spatially compact.

    The grid is built atom by atom, and atom order in a geometry file is not
    spatial, so contiguous index blocks otherwise straddle the whole molecule
    and no distance screen can bite.
    """
    c = coords - coords.min(axis=0)
    scale = c.max()
    if scale <= 0:
        return np.arange(len(c))
    bits = 21                                     # 3 * 21 fits one uint64
    q = np.minimum((c / scale * (2**bits - 1)).astype(np.uint64), 2**bits - 1)
    key = np.zeros(len(c), dtype=np.uint64)
    for b in range(bits):
        for d in range(3):
            key |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + d)
    return np.argsort(key)


def self_energy_matrix_imaginary_time(X_ao, D, W_omega, mo_coeff, eps, nocc,
                                      tau_points, omega_in, omega_out,
                                      mu=None, ranges=None, tau_indices=None,
                                      Wt_tau=None, block_memory_gb=4.0,
                                      coords=None, screen_r_cut=None):
    """Full Sigma^c_{mu nu}(i.omega) in the AO basis, shape (nfreq, nao, nao).

    The matrix counterpart of `self_energy_frequency_from_time`, which returns
    only one MO-diagonal element. Same chain and the same two fitted ranges;
    what changes is that the outer contraction keeps both AO indices:

        Sigma_{mu nu}(tau) = sum_PQ X_ao[P,mu] (Zt_PQ * Ghat_PQ) X_ao[Q,nu]

    Two collocation matrices are involved and they are NOT interchangeable.
    Ghat needs the occupied/virtual split, so it is built from the MO
    collocation X_mo = X_ao @ mo_coeff; the outer indices are AO, so they use
    X_ao. Passing an MO-basis X for both silently returns Sigma in the MO basis
    instead.

    Cost per tau is O(M^2 N + M N^2): two GEMMs and a Hadamard product for the
    WHOLE matrix, not per element. That is the O(N^3)-for-everything claim --
    the frequency route needs O(N^3) per state, so it only matches this for a
    handful of states.

    Wt_tau: the screened interaction already on the time grid. Pass it when the
    caller built W blockwise (`screened_interaction_tau_blocked`) and W_omega --
    (nfreq, naux, naux), the largest array on this route -- was never formed at
    all; W_omega is then ignored. The peer parameter on
    `self_energy_frequency_from_time`, so the low-memory sweep serves both.

    block_memory_gb caps the per-block working set. It does not change the
    answer or the flop count, only the peak allocation.

    coords + screen_r_cut (Bohr) drop (P, Q) block pairs whose bounding spheres
    are further apart than the cutoff. Sigma^{<,>}_PQ decays in |r_P - r_Q| --
    G at a rate set by the gap, Wt because it is screened -- so the surviving
    fraction falls with system size, which is where the cubic scaling is
    actually realized. A magnitude bound is NOT usable here: Cauchy-Schwarz on
    ||G[P]|| ||G[Q]|| discards the row overlap that carries the decay, and
    measured on benzene it drops 34% of pairs where only 3.7% are negligible.

    tau_indices selects a subset of tau points and returns that subset's
    contribution; the full Sigma is the sum over subsets, since the tau -> omega
    transform is a sum over tau. Distributing tau across ranks and reducing is
    therefore exact, with each rank threading its own GEMMs -- the same
    two-level split `chi0_imaginary_frequency` supports.

    Memory is (len(omega_out), nao, nao) complex -- the returned array and
    little else; the time points are folded in as they are built. Project it
    down with one of

        sigma_ao_to_mo           full MO matrix     -> qsGW, scGW
        sigma_ao_to_mo_diagonal  MO diagonal        -> evGW
        sigma_ao_to_mo_element   one MO element     -> G0W0

    and discard, or loop over frequency blocks, if that does not fit.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = ranges or self_energy_fit_ranges(eps, nocc, mu=mu)

    if Wt_tau is None:
        Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, omega_in, *rW)
        Wt_tau = _transform_screened(Ctw, W_omega)
    # else the caller built it blockwise and W_omega was never formed at all

    X_mo = X_ao @ mo_coeff
    X_o, X_v = X_mo[:, occ], X_mo[:, virt]
    e_o, e_v = eps[occ] - mu, eps[virt] - mu

    ntau, nao = len(tau_points), X_ao.shape[1]
    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS)

    # STREAMED OVER TAU, not staged. The tau -> omega transform is a sum over
    # tau, so each time point can be folded into the output as soon as it is
    # built and never stored. Staging Sigma^{<,>}(tau) first would hold two
    # (ntau, nao, nao) arrays, and forming `sig_g +/- sig_l` for the transform
    # two more -- and with both grids on 'auto' nfreq == ntau, so those are the
    # same size as the result itself. Measured peak was 6.1 (n, nao, nao)
    # float64 stacks against the 2 this array actually needs.
    # zeros, not empty: with tau_indices set only the owned points contribute
    # and the rest must add nothing, since the caller reduces over subsets.
    out = np.zeros((len(omega_out), nao, nao), dtype=complex)
    # BLOCKED OVER THE INTERPOLATION INDEX. Zt, G_l and G_g are each (M, M),
    # and the unblocked form holds three of them plus the Hadamard temporary --
    # 159 GB at M = 70448 (the 476-atom hexamer/cc-pVDZ), which the OOM killer
    # ended. Every term is a sum over P, so cutting that index costs nothing:
    # the three GEMMs become (b, M) row slabs, the outer contraction
    # accumulates into a (nao, M) buffer and multiplies by X_ao once at the end.
    # Flop counts are identical term by term; only the working set changes.
    M, naux = X_ao.shape[0], D.shape[1]
    # b from the per-block working set: one (b, naux) and two (b, b).
    b = int((-naux + np.sqrt(naux**2 + 8 * block_memory_gb * 1e9 / 8)) / 4)
    b = max(1, min(M, b))
    edges = list(range(0, M, b)) + [M]
    pairs = list(zip(edges[:-1], edges[1:]))
    # Spatial order first, then block: the screen is geometric, so the blocks
    # have to be. Permuting P is a relabelling of a summation index and leaves
    # Sigma unchanged.
    far = None
    if coords is not None and screen_r_cut:
        idx = _morton_order(np.asarray(coords))
        X_ao = np.ascontiguousarray(X_ao[idx])
        D = np.ascontiguousarray(D[idx])
        X_mo = X_ao @ mo_coeff
        X_o, X_v = X_mo[:, occ], X_mo[:, virt]
        P = np.asarray(coords)[idx]
        cen = np.array([P[p0:p1].mean(axis=0) for p0, p1 in pairs])
        rad = np.array([np.linalg.norm(P[p0:p1] - c, axis=1).max()
                        for (p0, p1), c in zip(pairs, cen)])
        sep = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
        far = sep - rad[:, None] - rad[None, :] > screen_r_cut

    which = range(ntau) if tau_indices is None else np.atleast_1d(tau_indices)
    for k in which:
        tau = tau_points[k]
        Wk = Wt_tau[k]
        Xo_t = X_o * np.exp(e_o * tau)
        Xv_t = X_v * np.exp(-e_v * tau)

        # Cauchy-Schwarz bound per interpolation point, so a block pair whose
        # product cannot reach the tolerance is skipped without being built.
        # |G^<_PQ| <= ||X_o[P] e^{e tau/2}|| ||X_o[Q] e^{e tau/2}|| and
        # |Zt_PQ| <= ||D[P]|| ||Wt||_F ||D[Q]||, the Frobenius norm standing in
        # for the spectral one so the bound stays cheap. Rigorous, not heuristic:
        # a skipped pair is bounded, never estimated.
        T_l = np.zeros((nao, M))
        T_g = np.zeros((nao, M))
        for ip, (p0, p1) in enumerate(pairs):
            DW = D[p0:p1] @ Wk
            Xa_p = X_ao[p0:p1].T
            for iq, (q0, q1) in enumerate(pairs):
                if far is not None and far[ip, iq]:
                    continue
                Zt = DW @ D[q0:q1].T
                G = Xo_t[p0:p1] @ X_o[q0:q1].T
                G *= Zt                              # in place; Zt is reused
                T_l[:, q0:q1] += Xa_p @ G
                G = Xv_t[p0:p1] @ X_v[q0:q1].T
                G *= Zt
                T_g[:, q0:q1] -= Xa_p @ G
                del Zt, G
            del DW
        s_l = T_l @ X_ao
        s_g = T_g @ X_ao
        del T_l, T_g
        even_k, odd_k = s_g + s_l, s_g - s_l
        # One frequency at a time: np.multiply.outer(C[:, k], even_k) would be
        # correct but allocates a whole (nfreq, nao, nao) temporary per tau,
        # which is the thing being avoided.
        for w in range(len(omega_out)):
            out.real[w] += C[w, k] * even_k
            out.imag[w] += S[w, k] * odd_k
    out *= -0.5
    return out


def self_energy_matrix_imaginary_time_df(df_coeff, W_omega, eps, nocc,
                                         tau_points, omega_in, omega_out,
                                         mu=None, ranges=None, Wt_tau=None):
    """Full Sigma^c_pq(i.omega) from a DENSE three-index tensor, via imaginary time.

    The peer of `self_energy_matrix_imaginary_time` for a caller that has a
    dense (naux, norb, norb) DF tensor rather than the separable pair (X, D).
    Everything above the contraction is shared -- the same two fitted ranges,
    the same W(i.omega) -> W(i.tau) cosine transform, the same even/odd
    recombination -- so the two agree on the physics and differ only in how the
    aux index is carried.

    WHY A SECOND KERNEL EXISTS. The separable route's O(N^3) rests on the
    Hadamard product `Zt * G` between two (M, M) matrices, which needs the
    co-densities factorized as X_P,mu X_P,nu. A dense B has no such structure,
    so the aux index cannot be folded away and the cost per tau is
    O(naux^2 norb^2 + naux norb^3) -- the same per-point cost as the
    frequency-axis form it replaces. What imaginary time buys here is therefore
    NOT a better exponent: it is the point count, ~24 minimax tau against the
    96 Gauss-Legendre frequencies that route needed, because Sigma(i.tau) is a
    product where Sigma(i.omega) was a convolution.

    Sigma_pq(i.tau) = sum_m G_m(i.tau) sum_PQ B_P,pm Wc_PQ(i.tau) B_Q,mq

    with G split into its lesser (occupied) and greater (virtual) parts, which
    is what makes the tau -> omega transform separate into a cosine piece on
    the even combination and a sine piece on the odd one.

    Passing eps - mu with mu = 0 returns Sigma_pq(i.omega), matching the
    convention of the frequency-axis form; passing raw eps and mu returns
    Sigma_pq(mu + i.omega), matching `self_energy_matrix_imaginary_time`.

    Wt_tau: W already on the time grid, as on the separable kernels; W_omega is
    then ignored.
    """
    occ, virt = get_occ_virt_indices(eps, nocc)
    if mu is None:
        mu = 0.5 * (eps[occ].max() + eps[virt].min())
    rW, rS = ranges or self_energy_fit_ranges(eps, nocc, mu=mu)

    if Wt_tau is None:
        Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, omega_in, *rW)
        Wt_tau = _transform_screened(Ctw, W_omega)

    naux, norb, _ = df_coeff.shape
    B_flat = df_coeff.reshape(naux, -1)
    e_o, e_v = eps[occ] - mu, eps[virt] - mu

    ntau = len(tau_points)
    sig_l = np.zeros((ntau, norb, norb))
    sig_g = np.zeros((ntau, norb, norb))
    for k in range(ntau):
        tau = tau_points[k]
        # (W B)_Q,pm then contract the aux index against B_Q,mq. Building A once
        # per tau and reweighting its middle index is the same factorization the
        # frequency-axis form used across query points -- here the reweighting
        # is the Green's function exponential instead.
        WB = (Wt_tau[k] @ B_flat).reshape(naux, norb, norb)
        A = np.einsum('Qpm,Qmq->pmq', WB, df_coeff, optimize=True)
        sig_l[k] = np.einsum('pmq,m->pq', A[:, occ, :], np.exp(e_o * tau),
                             optimize=True)
        sig_g[k] = -np.einsum('pmq,m->pq', A[:, virt, :], np.exp(-e_v * tau),
                              optimize=True)

    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS)
    even = np.tensordot(C, sig_g + sig_l, axes=(1, 0))
    odd = np.tensordot(S, sig_g - sig_l, axes=(1, 0))
    return -0.5 * (even + 1j * odd)


def sigma_ao_to_mo(sigma_ao, mo_coeff):
    """Full MO-basis Sigma, (nfreq, nmo, nmo) -- what qsGW and scGW need."""
    return np.einsum('mp,wmn,nq->wpq', mo_coeff, sigma_ao, mo_coeff,
                     optimize=True)


def sigma_ao_to_mo_diagonal(sigma_ao, mo_coeff, states=None):
    """MO-diagonal Sigma_pp, (nfreq, nstates) -- what evGW needs.

    Never forms the full MO matrix, so it stays O(nfreq nao^2 nstates).
    """
    C = mo_coeff if states is None else mo_coeff[:, states]
    return np.einsum('mp,wmn,np->wp', C, sigma_ao, C, optimize=True)


def sigma_ao_to_mo_element(sigma_ao, mo_coeff, p, q=None):
    """A single MO element Sigma_pq, (nfreq,) -- what G0W0 needs (q = p).

    Named for both bases like its siblings: the input is AO, the output is one
    MO element. A bare `sigma_mo_element` would read as if it took MO input.
    """
    cp = mo_coeff[:, p]
    cq = cp if q is None else mo_coeff[:, q]
    return np.einsum('m,wmn,n->w', cp, sigma_ao, cq, optimize=True)
